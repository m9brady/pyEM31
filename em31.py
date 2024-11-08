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


# instrument constants and lookup-tables
SURVEY_UNITS = {
    0: "meters",
    1: "feet"
}
DIPOLE_MODES = {
    0: "vertical",
    1: "horizontal",
    2: "both"
}
SURVEY_MODES = {
    0: "auto",
    1: "wheel",
    2: "manual"
}
EM31_COMPONENTS = {
    0: "both",
    1: "inphase",
    2: "conductivity"
}
EM31_SUBTYPES = {
    0: "standard",
    1: "short 2m"
}
NMEA_TYPES = {
    0: "GGA/GSA",
    1: "GGA",
    2: "POS",
    3: "LLK",
    4: "LLQ",
    5: "GLL",
    6: "GGK",
    7: "Leica TPS"
}


def text_to_bits(text, encoding="windows-1252", errors="surrogatepass"):
    """
    Convert instrument data to something useful
    """
    bits = bin(int.from_bytes(text.encode(encoding, errors), "big"))[2:]
    return bits.zfill(8 * ((len(bits) + 7) // 8))


def read_data(filename, gps_tol=1, encoding="windows-1252"):
    """
    Load R31/H31 files output from the EM31

    input:
        filename: Path to the R31/H31 file
        gps_tol: GPS time tolerance (seconds)
        encoding: Encoding of the file
    output:
        em31_merged: Pandas dataframe containing parsed EM31 measurement and GPS data

    note: most of the index-related stuff is from the EM31 documentation PDFs found online
    """
    with open(filename, "r", encoding=encoding) as r31_file:
        r31_dat = r31_file.read().splitlines()
    # make use of a crude cursor index since we might have to manipulate our index position
    # based on the type of data file
    idx = 0
    # Header is always first row and seems to be either 2 (R31) or 3 (H31) rows long
    header_1 = r31_dat[idx]
    instrument = header_1[:7].strip()
    version = header_1[8:12]  # Software version
    survey_type = header_1[12:15]  # GPS or GRD (Grid)
    unit_type = SURVEY_UNITS.get(int(header_1[15]))
    dipole_mode = DIPOLE_MODES.get(int(header_1[16]))
    survey_mode = SURVEY_MODES.get(int(header_1[17]))
    em_component = EM31_COMPONENTS.get(int(header_1[18]))
    # only the H31 data seems to have the instrument subtype
    if instrument == "NAV31":
        em_subtype = EM31_SUBTYPES.get(int(header_1[19]))
    idx += 1 # done with row 1
    # second header row
    header_2 = r31_dat[idx]
    if not header_2.startswith("H "):
        LOGGER.error("Missing header 2nd row")
        return 1
    file_label = header_2[2:11].strip()
    tws = header_2[11:18] # Time/Wheel/Samples depends on survey_mode
    if survey_mode in ["auto", "wheel"]:
        # auto: time increment in seconds
        # wheel: wheel increment in user units
        tws = float(tws) 
    elif survey_mode == "manual":
        # manual: samples/reading (int?)
        tws = int(tws)
    # extra flag for NAV31
    if instrument == "NAV31":
        file_tag = "original" if int(header_2[21]) == 0 else "possibly-modified"
    idx += 1 # done with row 2
    # 3rd header row if NAV31
    if instrument == "NAV31":
        header_3 = r31_dat[idx]
        if not header_3.startswith("G"):
            LOGGER.error("Missing header 3rd row for datafile type NAV31")
            return 1
        gps_xoffset = float(header_3[1:8]) # Offset of GPS Antenna in X direction
        gps_yoffset = float(header_3[8:15]) # Offset of GPS Antenna in Y direction
        nmea_type = NMEA_TYPES.get(int(header_3[19]))
        idx += 1 # done with row 3
    # after the header rows the survey line metadata starts and is usually 4 lines
    survey_meta = r31_dat[idx:idx+4]
    # the first chars in each line put together must be LBAZ
    assert "".join([line[0] for line in survey_meta]) == "LBAZ"
    # we don't really care about anything except the Z (time) information
    time_meta = survey_meta[-1]
    if not time_meta.startswith("Z"):
        LOGGER.error("Missing survey-time metadata")
        return 1
    ddmmyyyy = time_meta[1:9]
    # hhmmss = time_meta[10:20].strip() #TODO: check if not useful
    idx += 4 # done with survey metadata
    # after the survey metadata is the timer information (epoch)
    epoch_meta = r31_dat[idx]
    if not epoch_meta.startswith("*"):
        LOGGER.error("Epoch information missing from header")
        return 1
    epoch_time = epoch_meta[1:13] # precise computer time (local timezone probably)
    epoch_ms = int(epoch_meta[13:23]) # datalogger reference epoch (?)
    epoch_ts = datetime.strptime(f"{ddmmyyyy} {epoch_time}", "%d%m%Y %H:%M:%S.%f")
    # Extract the measurements and gps
    meas_df = extract_measurements(r31_dat, epoch_ms, epoch_ts, em_component, instrument, encoding=encoding)
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


def parse_data(text, em_component, instrument, encoding="windows-1252"):
    """
    Given a line of EM31 measurement data and the desired component, extract some information
    """
    bits = text_to_bits(text[1], encoding=encoding)
    # measurement flags
    meas_range3 = int(bits[5])
    meas_range2 = int(bits[6])
    # measurement data
    # NB: in the H31 manual there is a gap between reading1 and reading2
    # but in the example H31 provided by Marios there is no gap
    meas_read1 = float(text[2:7])
    meas_read2 = float(text[7:12])
    meas_time = int(text[13:23])
    # based on the manual, there are several multiplication factors
    # depending on the em components and flags in the data
    if em_component == "both":
        # inphase multiplcation factor is constant in BOTH mode
        inphase_factor = -0.025
        # conductivity factor in BOTH mode depends on range flags
        if meas_range2 == 1:
            if meas_range3 == 1:
                conductivity_factor = -0.25
            elif meas_range3 == 0:
                conductivity_factor = -0.0025
        elif meas_range2 == 0:
            if meas_range3 == 1:
                conductivity_factor = -0.025
            elif meas_range3 == 0:
                conductivity_factor = np.nan
        # use 6-degree precision to avoid float issues
        # in BOTH mode, reading1 is conductivity and reading2 is inphase
        apparent_conductivity = round(meas_read1 * conductivity_factor, 6)
        inphase = round(meas_read2 * inphase_factor, 6)
    elif em_component == "inphase":
        # conductivity is not measured in INPHASE mode
        apparent_conductivity = np.nan
        if meas_range2 == 1:
            if meas_range3 == 1:
                inphase_factor = -0.0625
            elif meas_range3 == 0:
                inphase_factor = -0.000625
        elif meas_range2 == 0:
            if meas_range3 == 1:
                inphase_factor = -0.00625
            elif meas_range3 == 0:
                # no mention in manual of what to do when both range2 and range3 are 0
                inphase_factor = np.nan
        # use 6-degree precision to avoid float issues
        # in INPHASE mode, reading1 is inphase
        inphase = round(meas_read2 * inphase_factor, 6)    
    # according to manual, if short EM31 being used then divide inphase by 3.35
    if instrument == "EM31-SH":
        inphase /= 3.35
    return meas_time, bits, meas_range2, meas_range3, conductivity_factor, apparent_conductivity, inphase


def extract_measurements(r31_dat, epoch_ms, epoch_ts, em_component, instrument, encoding):
    """
    Given an in-memory list of raw EM31 data, attempt to parse the sensor measurement data

    input:
        r31_dat: list of data lines from R31 data file
        epoch_ms: datalogger epoch referencing start-of-data-recording (?)
        epoch_ts: datalogger timestamp (computer time) at point of epoch_ms (?)
        em_component: the chosen em31 surveying mode ("both", "inphase" or "conductivity")
        instrument: if EM31-SH, additional multiplcation factor needed
        encoding: file encoding (default "windows-1252")
    output:
        meas_df: Pandas dataframe containing the parsed data
    """
    # Read the measurements into DF
    # NB: "T" works for "auto" mode, but manual mode can include a "2"
    meas_idx = [idx for idx, line in enumerate(r31_dat) if line.startswith("T")]
    meas_data = np.array(
        [parse_data(r31_dat[idx], em_component, instrument, encoding=encoding) for idx in meas_idx]
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
    for data_file in sorted(src_dir.glob("???????*.?31")):
        target = dst_dir / f"{data_file.stem}.ttem.csv"
        if target.exists():
            LOGGER.info("Skipping existing file: %s" % target.as_posix())
            continue
        data_size_MB = data_file.stat().st_size / 1024**2
        LOGGER.info("Processing %s (~%.2f MB)" % (data_file.as_posix(), data_size_MB))
        df = read_data(data_file)
        df = thickness(df, 0.15)
        df.to_csv(target, index=False, na_rep="NaN")
        LOGGER.info("Saved to CSV: %s" % target.as_posix())
