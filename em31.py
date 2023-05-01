"""
Processing functions for the EM31
J.King 2022

Many thanks to Christian Haas for the help with this
"""

import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pynmea2

LOGGER = logging.getLogger("pyem31")
LOGGER.addHandler(logging.NullHandler())

# These coefficients for estimating ice thickness from measured apparent conductivity
# are derived from Haas et. al (2017): http://dx.doi.org/10.1002/2017GL075434
# Supplementary figure S2
HAAS_2010 = [0.98229, 13.404, 1366.4]


def text_to_bits(text, encoding="windows-1252", errors="surrogatepass"):
    """
    Convert instrument data to something useful
    """
    bits = bin(int.from_bytes(text.encode(encoding, errors), "big"))[2:]
    return bits.zfill(8 * ((len(bits) + 7) // 8))


def read_r31(filename, gps_tol=1, encoding="windows-1252"):
    """
    Load R31 files output from the EM31

    input:
        filename: Path to the R31 file
        gps_tol: GPS time tolerance (seconds)
        encoding: Encoding of the file
    output:
        em31_merged: Pandas dataframe containing parsed EM31 measurement and GPS data
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
        LOGGER.error("First line is not a header")
        return 1
    # Get file data fields
    file_meta = r31_dat[1]
    if file_meta.startswith("H"):
        h_file = file_meta[2:11].strip()
        h_time = float(file_meta[13:18])
    else:
        LOGGER.error("Data fields missing")
        return 1
    # Get file start stamp
    time_meta = r31_dat[5]
    if time_meta.startswith("Z"):
        h_date = time_meta[1:9]
        h_time = time_meta[10:18]
    else:
        LOGGER.error("Start stamp missing")
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
        gps_df,
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
    """
    Given an in-memory list of raw EM31 data, attempt to parse the sensor measurement data

    input:
        r31_dat: list of data lines from R31 data file
        epoch_ms: ?
        epoch_ts: ?
        encoding: file encoding (default "windows-1252")
    output:
        meas_df: Pandas dataframe containing the parsed data
    """
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

    input:
        gps_data: single line of NMEA0183 GPS data
        idx_of_em31: line number in R31 file where gps_data resides
    output:
        gps_msg: parsed GPS message in pynmea object
    """
    try:
        gps_msg = pynmea2.parse(gps_data)
    except pynmea2.ChecksumError:
        LOGGER.warning("NMEA0183 checksum error with line %d" % idx_of_em31)
        LOGGER.debug("%r" % gps_data)
        return
    except pynmea2.ParseError:
        LOGGER.warning("NMEA0183 parse error with line %d" % idx_of_em31)
        LOGGER.debug("%r" % gps_data)
        return
    return gps_msg


# TODO interpolate the GPS data instead of taking nearest
def extract_gps(r31_dat, epoch_ms, epoch_ts):
    """
    Extract the GPS information from a given EM31 dataset

    input:
        r31_dat: list of data lines from R31 data file
        epoch_ms: ?
        epoch_ts: ?
    output:
        meas_df: Pandas dataframe containing the parsed data
    """
    # detect where the GPS data chunks are in the EM31 data file
    gps_starts = [idx for idx, line in enumerate(r31_dat) if line.startswith("@")]
    gps_ends = [idx for idx, line in enumerate(r31_dat) if line.startswith("!")]
    assert len(gps_starts) == len(gps_ends)
    # gps messages can vary in length but are no longer than 6
    gps_data = [r31_dat[start:end] for start, end in zip(gps_starts, gps_ends)]
    # detect where EM31 measurements have been logged inside GPS messages
    bad_logs_idx = [
        gps_data.index(data)
        for data in gps_data
        if any([line.startswith("T") for line in data])
    ]
    if len(bad_logs_idx) != 0:
        LOGGER.debug(
            "Detected %d instances where EM31 data overrides GPS data"
            % len(bad_logs_idx)
        )
        for bad_idx in bad_logs_idx:
            em31_line_idx = [
                gps_data[bad_idx].index(line)
                for line in gps_data[bad_idx]
                if line.startswith("T")
            ]
            # remove the em31 data from gps_data by pop-from-list
            for line_idx in em31_line_idx:
                _ = gps_data[bad_idx].pop(line_idx)
    assert all([len(data) <= 6 for data in gps_data])
    gps_times = [int(r31_dat[end : end + 1][0].split(" ")[-1]) for end in gps_ends]
    # drop the first character in each line and join the remainder chunks into 1 string
    gps_data_clean = ["".join([d[1:] for d in data]).strip() for data in gps_data]
    # extract NMEA0183 objects
    nmea = [
        parse_gps(clean_string, em31_idx)
        for clean_string, em31_idx in zip(gps_data_clean, gps_starts)
    ]
    # we only want GGA NMEA messages apparently (Time, position, and fix related data)
    # https://receiverhelp.trimble.com/alloy-gnss/en-us/NMEA-0183messages_GGA.html
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
                msg.horizontal_dil,  # Horizontal Dilution of Precision (HDOP)
                msg.altitude,
                msg.latitude,
                msg.lat_dir,
                msg.longitude,
                msg.lon_dir,
            )
            for tstamp, msg in zip(gga_times, gga_msgs)
        ]
    )
    # attempt to use RMC (Position, velocity and time) messages for the useful speed-over-ground/course-made-good
    # https://receiverhelp.trimble.com/alloy-gnss/en-us/NMEA-0183messages_RMC.html
    rmc_idx = [idx for idx in range(len(nmea)) if isinstance(nmea[idx], pynmea2.RMC)]
    try:
        assert len(rmc_idx) == len(gga_idx)
        use_rmc = True
    except AssertionError:
        LOGGER.warning(
            "n_RMC (%d) does not equal n_GGA (%d) -> omitting sog/cmg from output"
            % (len(rmc_idx), len(gga_msgs))
        )
        use_rmc = False
    if use_rmc:
        rmc_msgs = [nmea[idx] for idx in rmc_idx]
        rmc_data = np.array(
            [
                (
                    datetime(
                        msg.datestamp.year,
                        msg.datestamp.month,
                        msg.datestamp.day,
                        msg.timestamp.hour,
                        msg.timestamp.minute,
                        msg.timestamp.second,
                        msg.timestamp.microsecond,
                        tzinfo=timezone.utc,  # always UTC according to Trimble
                    ),
                    msg.status,
                    msg.latitude,
                    msg.lat_dir,
                    msg.longitude,
                    msg.lon_dir,
                    msg.spd_over_grnd,  # sog, knots
                    msg.true_course,  # cmg, degrees from TRUE NORTH
                    msg.mag_variation,
                    msg.mag_var_dir,
                    msg.mode_indicator,
                    msg.nav_status,
                )
                for msg in rmc_msgs
            ]
        )
    # edge cases where some gps message contents are empty
    # nsats
    gga_data[:, 3] = np.where(gga_data[:, 3] == "", "00", gga_data[:, 3])
    # hdop
    gga_data[:, 4] = np.where(gga_data[:, 4] == "", np.nan, gga_data[:, 4])
    gps_df = pd.DataFrame(
        data={
            "time_sys": pd.Series(gga_data[:, 0]).astype("datetime64[ns]"),
            "time_gps": pd.Series(gga_data[:, 1]),
            "fix": pd.Series(gga_data[:, 2]).astype("uint8"),
            "nsats": pd.Series(gga_data[:, 3]).astype("uint16"),
            "hdop": pd.Series(gga_data[:, 4]).astype("float32"),
            "alt": pd.Series(gga_data[:, 5]).astype("float32"),
            "lat": pd.Series(gga_data[:, 6]).astype("float64"),
            "lat_dir": pd.Series(gga_data[:, 7]).astype(pd.StringDtype()),
            "lon": pd.Series(gga_data[:, 8]).astype("float64"),
            "lon_dir": pd.Series(gga_data[:, 9]).astype(pd.StringDtype()),
        }
    )
    # return only a subset of columns based on Josh's prior work
    cols = ["lat", "lon", "time_gps", "time_sys"]
    if use_rmc:
        gps_df["sog"] = pd.Series(rmc_data[:, 6]).astype("float32")
        gps_df["cmg"] = pd.Series(rmc_data[:, 7]).astype("float32")
        cols.extend(["sog", "cmg"])
    # isolate only the records with 0 < gps_quality < 6
    # different types of GGA quality indicators: https://receiverhelp.trimble.com/alloy-gnss/en-us/NMEA-0183messages_GGA.html
    subset = gps_df.loc[(gps_df["fix"] > 0) & (gps_df["fix"] < 6), cols]
    return subset


def thickness(em31_df, inst_height, coeffs=HAAS_2010):
    """
    Estimate total thickness from apparent conductivity

    input:
        em31_df: pyEM31 dataframe
        inst_height: height of the instrument above the snow surface (meters?)
        coeffs: 3 element list of retrieval coefficients
    output:
        em31_df: pyEM31 dataframe with total thickness

    TODO: instead of passing dataframe and modifying, take apparent conductivity as input and produce
    thickness as output
    TODO: instead of list of coefficients, split into separate variables for readability
    """
    # modify the instrument measurements using the coefficients
    mod_app_cond = (em31_df["appcond"] - coeffs[1]) / coeffs[2]
    # fill negative values with np.nan to avoid np.log warnings
    mod_app_cond[mod_app_cond < 0] = np.nan
    # estimate total thickness from instrument
    em31_df["ttem"] = -1 / coeffs[0] * np.log(mod_app_cond)
    # account for instrument height above snow surface
    em31_df["ttem"] -= inst_height
    return em31_df


if __name__ == "__main__":
    """
    If this file is run, attempt to process everything in ./data/em31/ with console-logging
    """
    from pathlib import Path

    console_log = logging.StreamHandler()
    console_log.setFormatter(
        logging.Formatter(
            "%(asctime)s.%(msecs)03d | %(name)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    LOGGER.addHandler(console_log)
    LOGGER.setLevel(logging.INFO)

    src_dir = Path("./data/em31")
    dst_dir = Path("./data/output")
    src_dir.mkdir(exist_ok=True)
    dst_dir.mkdir(exist_ok=True)
    for data_file in sorted(src_dir.glob("???????*.R31")):
        target = dst_dir / f"{data_file.stem}.ttem.csv"
        if target.exists():
            LOGGER.info("Skipping existing file: %s" % target.as_posix())
            continue
        data_size_MB = data_file.stat().st_size / 1024**2
        LOGGER.info("Processing %s (~%.2f MB)" % (data_file.as_posix(), data_size_MB))
        df = read_r31(data_file)
        df = thickness(df, 0.15)
        df.to_csv(target, index=False, na_rep="NaN")
        LOGGER.info("Saved to CSV: %s" % target.as_posix())
