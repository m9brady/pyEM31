"""
Processing functions for the EM31
J.King 2022

Many thanks to Christian Haas for the help with this
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pynmea2

# Constants
HAAS_2010 = [0.98229, 13.404, 1366.4]


def text_to_bits(text, encoding="windows-1252", errors="surrogatepass"):
    bits = bin(int.from_bytes(text.encode(encoding, errors), "big"))[2:]
    return bits.zfill(8 * ((len(bits) + 7) // 8))


def read_r31(filename, gps_tol=1, encoding="windows-1252"):
    """
    Load R31 files output from the EM31
        Input:
            filename: Path to the R31 file
            gps_tol: GPS time tolerance (seconds)
            encoding: Encoding of the file
        Output:

    """
    with open(filename, "r", encoding=encoding) as r31_file:
        r31_dat = r31_file.read().splitlines()
    # Get header data
    header = r31_dat[0]
    if header.startswith("E"):
        h_ident = header[0:7].strip()  # System ident
        h_ver = header[7:12]  # Software version
        h_type = header[12:15]  # GPS or GRD (Grid)
        h_units = int(header[15:16])  # 0 = meter, 1 = feet
        h_dipole = int(header[16:17])  # 0 = vertical, 1 = horizontal
        h_mode = int(header[17:18])  # 0 = auto, 1 = wheel, 2 = manual
        h_component = int(header[18:19])  # 0 = Both, 1 = Iphase
        h_computer = int(header[22:23])  # no info on what this is
    else:
        print("Error: First line is not a header")
        return 1
    # Get file data fields
    file_meta = r31_dat[1]
    if file_meta.startswith("H"):
        h_file = file_meta[2:11].strip()
        h_time = float(file_meta[13:18])
    else:
        print("Error: Data fields missing")
        return 1
    # Get file start stamp
    time_meta = r31_dat[5]
    if time_meta.startswith("Z"):
        h_date = time_meta[1:9]
        h_time = time_meta[10:18]
    else:
        print("Error: Start stamp missing")
        return 1
    # Get timer relation
    epoch_meta = r31_dat[6]
    if epoch_meta.startswith("*"):
        epoch_time = epoch_meta[1:13]
        epoch_ms = int(epoch_meta[13:23])
        epoch_ts = datetime.strptime(f"{h_date} {epoch_time}", "%d%m%Y %H:%M:%S.%f")
    # Extract the measurements and gps
    meas_df = extract_measurements(r31_dat, epoch_ms, epoch_ts, encoding=encoding)
    gps_df = extract_gps(r31_dat, epoch_ms, epoch_ts)
    # Merge based on time
    em31_merged = pd.merge_asof(
        meas_df,
        gps_df[["lat", "lon", "time_gps", "time_sys"]],
        left_on="time_ds",
        right_on="time_sys",
        direction="nearest",
    )
    # Remove measurements where GPS is desynced
    # TODO: Make sure time_sys is what we think it is
    time_diff = em31_merged["time_ds"] - em31_merged["time_sys"]
    em31_merged = em31_merged.loc[time_diff < timedelta(seconds=gps_tol)]
    return em31_merged


def parse_data(text, encoding="windows-1252"):
    """
    Given a line of EM31 measurement data, extract some information
    """
    bits = text_to_bits(text[1], encoding=encoding)
    meas_range3 = int(bits[5])
    meas_range2 = int(bits[6])
    meas_read1 = float(text[2:7])
    meas_read2 = float(text[7:12])
    meas_time = int(text[13:23])
    if (meas_range2 == 1) and (meas_range3 == 1):
        c_factor = -0.25
        i_factor = -0.0025
        # use 6-degree precision to avoid float issues
        app_cond = round(meas_read1 * c_factor, 6)
        in_phase = round(meas_read2 * i_factor, 6)
    else:
        c_factor = np.nan
        i_factor = np.nan
        app_cond = np.nan
        in_phase = np.nan
    return meas_time, bits, meas_range2, meas_range3, c_factor, app_cond, in_phase


def extract_measurements(r31_dat, epoch_ms, epoch_ts, encoding):
    # Read the measurements into DF
    meas_idx = [idx for idx, line in enumerate(r31_dat) if line.startswith("T")]
    meas_data = np.array(
        [parse_data(r31_dat[idx], encoding=encoding) for idx in meas_idx]
    )
    meas_df = pd.DataFrame(
        data={
            "l_num": pd.Series(meas_idx, dtype="uint32"),
            "time_ms": pd.Series(meas_data[:, 0]).astype("uint32"),
            "flags": pd.Series(meas_data[:, 1]).astype(pd.StringDtype()),
            "range2": pd.Series(meas_data[:, 2]).astype("uint32"),
            "range3": pd.Series(meas_data[:, 3]).astype("uint32"),
            "c_factor": pd.Series(meas_data[:, 4]).astype("float32"),
            "appcond": pd.Series(meas_data[:, 5]).astype("float32"),
            "inph": pd.Series(meas_data[:, 6]).astype("float32"),
        }
    )
    # Create measurement time stamps
    meas_df["time_relative"] = meas_df["time_ms"] - epoch_ms
    meas_df["time_ds"] = epoch_ts
    meas_df["time_ds"] += pd.Series(
        [timedelta(milliseconds=rel) for rel in meas_df["time_relative"]]
    )
    return meas_df


def parse_gps(gps_data, idx_of_em31):
    """
    Given a cleaned line of EM31 GPS data and its line number in the EM31 datafile, return a NMEA0183 sentence object
    """
    try:
        gps_msg = pynmea2.parse(gps_data)
    except pynmea2.ChecksumError:
        print(f'GPS NMEA0183 checksum error with line {idx_of_em31}: "{gps_data}"')
        return
    except pynmea2.ParseError:
        print(f'GPS NMEA0183 parse error with line {idx_of_em31}: "{gps_data}"')
        return
    return gps_msg


# TODO interpolate the GPS data instead of taking nearest
def extract_gps(r31_dat, epoch_ms, epoch_ts):
    """
    Extract the GPS information from a given EM31 dataset
    """
    # detect where the GPS data chunks are in the EM31 data file
    gps_starts = [idx for idx, line in enumerate(r31_dat) if line.startswith("@")]
    gps_ends = [idx for idx, line in enumerate(r31_dat) if line.startswith("!")]
    # these can vary in length but are no longer than 6
    gps_data = [r31_dat[start:end] for start, end in zip(gps_starts, gps_ends)]
    gps_times = [int(r31_dat[end : end + 1][0].split(" ")[-1]) for end in gps_ends]
    assert all([len(data) <= 6 for data in gps_data])
    # drop the first character in each line and join the remainder chunks into 1 string
    gps_data_clean = ["".join([d[1:] for d in data]).strip() for data in gps_data]
    # extract NMEA0183 objects
    nmea = [
        parse_gps(clean_string, em31_idx)
        for clean_string, em31_idx in zip(gps_data_clean, gps_starts)
    ]
    # we only want GGA NMEA messages apparently
    gga_idx = [idx for idx in range(len(nmea)) if isinstance(nmea[idx], pynmea2.GGA)]
    gga_msgs = [nmea[idx] for idx in gga_idx]
    # need to add the gps_time to the epoch from EM31 header
    gga_times = [
        epoch_ts + timedelta(milliseconds=gps_times[idx] - epoch_ms) for idx in gga_idx
    ]
    gga_data = np.array(
        [
            (
                tstamp,
                msg.timestamp,
                msg.gps_qual,
                msg.num_sats,
                msg.horizontal_dil,
                msg.altitude,
                msg.latitude,
                msg.lat_dir,
                msg.longitude,
                msg.lon_dir,
            )
            for tstamp, msg in zip(gga_times, gga_msgs)
        ]
    )
    gps_df = pd.DataFrame(
        data={
            "time_sys": pd.Series(gga_data[:, 0]).astype("datetime64[ns]"),
            "time_gps": pd.Series(gga_data[:, 1]),
            "fix": pd.Series(gga_data[:, 2]).astype(bool),
            "nsats": pd.Series(gga_data[:, 3]).astype("uint16"),
            "hdop": pd.Series(gga_data[:, 4]).astype("float32"),
            "alt": pd.Series(gga_data[:, 5]).astype("float32"),
            "lat": pd.Series(gga_data[:, 6]).astype("float64"),
            "lat_dir": pd.Series(gga_data[:, 7]).astype(pd.StringDtype()),
            "lon": pd.Series(gga_data[:, 8]).astype("float64"),
            "lon_dir": pd.Series(gga_data[:, 9]).astype(pd.StringDtype()),
        }
    )
    # isolate only the records with 0 < fix < 6
    subset = gps_df.loc[(gps_df["fix"] > 0) & (gps_df["fix"] < 6)]
    return subset


def thickness(em31_df, inst_height, coeffs=HAAS_2010):
    """
    Estimate total thickness from apparent conductivity
        input:
            em31_df: pyEM31 dataframe
            inst_height: height of the instrument above the snow surface
            coeffs: 3 element list of retrieval coefficients
        output:
            em31_df: pyEM31 dataframe with total thickness
    """
    em31_df["ttem"] = (
        -1 / coeffs[0] * np.log((em31_df["appcond"] - coeffs[1]) / coeffs[2])
    )
    em31_df["ttem"] -= inst_height
    return em31_df
