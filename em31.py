"""
Processing functions for the EM31
J.King 2022

Many thanks to Christian Haas for the help with this
"""

import logging
from bisect import bisect_left
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pynmea2
import pyproj

LOGGER = logging.getLogger("pyem31")
LOGGER.addHandler(logging.NullHandler())

# These coefficients for estimating ice thickness from measured apparent conductivity
# are derived from Haas et. al (2017): http://dx.doi.org/10.1002/2017GL075434
# Supplementary figure S2
HAAS_2017 = [0.98229, 13.404, 1366.4]

# in Pandas 3.0+ the default timestamp resolution is MICROseconds
if int(pd.__version__.split(".")[0]) >= 3:
    USE_MICROSECONDS = True
else:
    USE_MICROSECONDS = False

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
# Pandas dtypes (instead of instantiating a new one each time)
PD_STR = pd.StringDtype()
PD_UINT8 = pd.UInt8Dtype()
PD_UINT16 = pd.UInt16Dtype()
PD_UINT32 = pd.UInt32Dtype()
PD_FLOAT32 = pd.Float32Dtype()
PD_FLOAT64 = pd.Float64Dtype()

class EM31GPSError(ValueError):
    """Raised when no valid GPS Positioning ($GNGGA) data found"""
    def __init__(self, message):
        super().__init__(message)
    def __str__(self):
        return super().__str__()
    

def text_to_bits(text, encoding="latin-1", errors="surrogatepass"):
    """
    Convert instrument data to something useful
    """
    bits = bin(int.from_bytes(text.encode(encoding, errors), "big"))[2:]
    return bits.zfill(8 * ((len(bits) + 7) // 8))

def get_measurements_before_after_comment(measurement_indices: list, comment_line_num: int) -> tuple:
    """
    When Pete makes a comment while surveying, he wants the measurement before and after the comment
    in order to get a good location fix

    Assumptions:
        - measurement_indices is sorted
    """
    list_idx = bisect_left(measurement_indices, comment_line_num)
    # if the comment is before any measurement data, we return None for the left-index
    if list_idx == 0:
        before = None
        after = measurement_indices[list_idx]
    # same idea if it is at the very end (without any subsequent measurements)
    elif list_idx == len(measurement_indices) - 1:
        before = measurement_indices[list_idx - 1]
        after = None
    else:
        before = measurement_indices[list_idx - 1]
        after = measurement_indices[list_idx]
    return (before, after)


def read_data(filename, gps_tol=1, encoding="latin-1"):
    """
    Load R31/H31 files output from the EM31

    input:
        filename: Path to the R31/H31 file
        gps_tol: GPS time tolerance (seconds)
        encoding: Encoding of the file (default 'latin-1')
    output:
        em31_merged: Pandas dataframe containing parsed EM31 measurement and GPS data

    note: most of the index-related stuff is from the EM31 documentation PDFs found online
    """
    if not isinstance(filename, Path):
        filename = Path(filename)
    if not filename.exists():
        raise FileNotFoundError(f"No such file: {filename.resolve()}")
    with filename.open("r", encoding=encoding) as f:
        raw_data = f.read().splitlines()
    # make use of a crude cursor index since we might have to manipulate our index position
    # based on the type of data file
    idx = 0
    # Header is always first row and seems to be either 2 (R31) or 3 (H31) rows long
    header_1 = raw_data[idx]
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
    header_2 = raw_data[idx]
    if not header_2.startswith("H "):
        LOGGER.error("Missing header 2nd row")
        return 1
    file_label = header_2[2:11].strip()
    # quick check ensure filename hasn't been mashed
    try:
        assert file_label == filename.stem
    except AssertionError:
        LOGGER.warning(f"Filename/Internal label mismatch for file {filename.resolve()}: {filename.stem!r} vs {file_label!r}")
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
        header_3 = raw_data[idx]
        if not header_3.startswith("G"):
            LOGGER.error("Missing header 3rd row for datafile type NAV31")
            return 1
        gps_xoffset = float(header_3[1:8]) # Offset of GPS Antenna in X direction
        gps_yoffset = float(header_3[8:15]) # Offset of GPS Antenna in Y direction
        nmea_type = NMEA_TYPES.get(int(header_3[19]))
        idx += 1 # done with row 3
    # after the header rows the survey line metadata starts and is usually 4 lines
    survey_meta = raw_data[idx:idx+4]
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
    epoch_meta = raw_data[idx]
    if not epoch_meta.startswith("*"):
        LOGGER.error("Epoch information missing from header")
        return 1
    epoch_time = epoch_meta[1:13] # precise computer time (local timezone probably)
    epoch_ms = int(epoch_meta[13:23]) # datalogger reference epoch (?)
    epoch_ts = datetime.strptime(f"{ddmmyyyy} {epoch_time}", "%d%m%Y %H:%M:%S.%f")
    # Extract the measurements and gps
    meas_df = extract_measurements(raw_data, epoch_ms, epoch_ts, em_component, instrument, encoding)
    gps_df = extract_gps(raw_data, epoch_ms, epoch_ts)
    LOGGER.info('Interpolating GPS data...')
    gps_interp = interpolate_gps(gps_df, 3413, 1e-3)
    # new in Pandas 3.0+
    if USE_MICROSECONDS:
        gps_interp['time_sys'] = gps_interp['time_sys'].astype('datetime64[us]')
    # Merge based on time
    LOGGER.info('Merging EM31 data with interpolated GPS data...')
    em31_merged = pd.merge_asof(
        meas_df,
        gps_interp,
        left_on="time_ds",
        right_on="time_sys",
        direction="nearest",
    )
    # Remove measurements where GPS is desynced
    # NB: now that we interpolate by default, there should be zero desync
    # time_sys: datalogger timestamp for GGA-NMEA message
    # time_ds: datalogger timestamp for EM31 data
    time_diff = em31_merged["time_ds"] - em31_merged["time_sys"]
    em31_merged = em31_merged.loc[time_diff < timedelta(seconds=gps_tol)]
    # detect comments, if any are present expose them in the log
    comm_idx = [idx for idx, line in enumerate(raw_data) if line.startswith("C")]
    if len(comm_idx) > 0:
        l_nums = em31_merged["l_num"].to_list()
        for comment_idx in comm_idx:
            # CEMSI EM31 data has corrupted lines that get treated as Comments (length 23 starting with C)
            # so we sneakily try to detect non-integer data (where there absolutely should be int data)
            try:
                int(raw_data[comment_idx][13:])
            except ValueError:
                continue
            before, after = get_measurements_before_after_comment(l_nums, comment_idx)
            LOGGER.info(f"Comment: {raw_data[comment_idx][1:13].strip()!r} data lines: {before, after}")
    return em31_merged


def interpolate_gps(gps_df, projected_crs=3413, resample_freq_seconds=1e-3):
    """
    Given a GPS dataframe containing lats, lons, timestamps from GGA NMEA messages from EM31 
    datafile and a desired projected coordinate reference system epsg code:
     - project lons/lats to passed crs
     - interpolate to passed sample_freq (seconds)
     - convert XY back to LonLat

     NB: currently defaults to NSIDC North Pole Stereographic with 0.001s resample freq
    """
    trans = pyproj.Transformer.from_crs(
        crs_from=pyproj.CRS.from_epsg(4326),
        crs_to=pyproj.CRS.from_epsg(projected_crs),
        always_xy=True
    )
    xs, ys = trans.transform(gps_df['lon'], gps_df['lat'])
    # NB: for longer datafiles and smaller resample_freq_seconds this could
    # produce gigantic tables
    interp_df = gps_df[['lon', 'lat', 'time_sys']].assign(
        lon=xs,
        lat=ys
    ).set_index('time_sys').resample(f'{resample_freq_seconds:f}s').interpolate()
    lons, lats = trans.transform(interp_df['lon'], interp_df['lat'], direction='INVERSE')
    interp_df['lon'] = lons
    interp_df['lat'] = lats
    return interp_df.reset_index()


def parse_data(text, em_component, instrument, encoding="latin-1"):
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
    try:
        meas_read1 = float(text[2:7])
        meas_read2 = float(text[7:12])
        meas_time = int(text[13:23])
    except ValueError:
        # in R31 provided by CEMSI partners, some strange 
        # characters interrupt instrument data so we skip 
        # those rows
        return (pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA)
    # based on the manual, there are several multiplication factors
    # depending on the em components and flags in the data
    if em_component == "both":
        # inphase multiplication factor is constant in BOTH mode
        inphase_factor = -0.025
        # conductivity factor in BOTH mode depends on range flags
        if meas_range2 == 1 and meas_range3 == 1:
            conductivity_factor = -0.25
            # weird edge-case where the inphase factor is actually -0.0025
            inphase_factor = -0.0025
        elif meas_range2 == 0 and meas_range3 == 1:
            conductivity_factor = -0.025
        elif meas_range2 == 1 and meas_range3 == 0:
            conductivity_factor = -0.0025
        elif meas_range2 == 0 and meas_range3 == 0:
            # geonics folks indicate that 0 0 is equivalent to 1 1
            conductivity_factor = -0.25
        # use 6-degree precision to avoid float issues
        # in BOTH mode, reading1 is conductivity and reading2 is inphase
        apparent_conductivity = round(meas_read1 * conductivity_factor, 6)
        inphase = round(meas_read2 * inphase_factor, 6)
    elif em_component == "inphase":
        if meas_range2 == 1 and meas_range3 == 1:
            inphase_factor = -0.0625
        elif meas_range2 == 0 and meas_range3 == 1:
            inphase_factor = -0.00625
        elif meas_range2 == 1 and meas_range3 == 0:
            inphase_factor = -0.000625
        elif meas_range2 == 0 and meas_range3 == 0:
            # geonics folks indicate that 0 0 is equivalent to 1 1
            inphase_factor = -0.0625
        # use 6-degree precision to avoid float issues
        # in INPHASE mode, reading1 is inphase
        inphase = round(meas_read1 * inphase_factor, 6)
        # conductivity is not measured in INPHASE mode
        apparent_conductivity = np.nan
    # according to manual, if short EM31 being used then divide inphase by 3.35
    if instrument == "EM31-SH":
        inphase /= 3.35
    return meas_time, bits, meas_range2, meas_range3, conductivity_factor, apparent_conductivity, inphase


def extract_measurements(raw_data, epoch_ms, epoch_ts, em_component, instrument, encoding):
    """
    Given an in-memory list of raw EM31 data, attempt to parse the sensor measurement data

    input:
        raw_data: list of data lines from R31/H31 data file
        epoch_ms: datalogger epoch referencing start-of-data-recording (?)
        epoch_ts: datalogger timestamp (computer time) at point of epoch_ms (?)
        em_component: the chosen em31 surveying mode ("both", "inphase" or "conductivity")
        instrument: if EM31-SH, additional multiplication factor needed
        encoding: needed for converting text chars into bits in parse_data()
    output:
        meas_df: Pandas dataframe containing the parsed data
    """
    # Read the measurements into DF
    # NB: "T" works for "auto" mode, but manual mode can include a "2"
    meas_idx = [idx for idx, line in enumerate(raw_data) if line.startswith("T")]
    meas_data = np.array(
        [parse_data(raw_data[idx], em_component, instrument, encoding) for idx in meas_idx]
    )
    meas_df = pd.DataFrame(
        data={
            "l_num": pd.Series(meas_idx, dtype=PD_UINT32) + 1,          # we +1 here because of zero-indexing
            "time_ms": pd.Series(meas_data[:, 0], dtype=PD_UINT32),
            "flags": pd.Series(meas_data[:, 1], dtype=PD_STR),
            "range2": pd.Series(meas_data[:, 2], dtype=PD_UINT32),
            "range3": pd.Series(meas_data[:, 3], dtype=PD_UINT32),
            "c_factor": pd.Series(meas_data[:, 4], dtype=PD_FLOAT32),
            "appcond": pd.Series(meas_data[:, 5], dtype=PD_FLOAT32),
            "inph": pd.Series(meas_data[:, 6], dtype=PD_FLOAT32),
        }
    )
    # drop any malformed measurement data (except for appcond which might be truthfully NaN)
    n_nan_meas = len(meas_df.loc[pd.isna(meas_df["time_ms"])])
    if n_nan_meas > 0:
        LOGGER.warning(f"Identified {n_nan_meas} malformed data rows. Dropping from final dataframe.")
        meas_df = meas_df.dropna(subset=["time_ms", "flags", "range2", "range3", "c_factor", "inph"]).reset_index(drop=True)
    # Create measurement time stamps
    meas_df["time_relative"] = meas_df["time_ms"] - epoch_ms
    meas_df["time_ds"] = epoch_ts + pd.Series(
        [pd.Timedelta(milliseconds=float(rel)) for rel in meas_df["time_relative"]]
    )
    # in the CEMSI R31, there are very strange "time_relative" entries that are unsorted?
    # we identify any negative timestamp diffs in sequential measurement rows and drop the offending row
    tdiffs = meas_df['time_ds'].shift(-1) - meas_df['time_ds']
    anomalous_tdiffs = tdiffs.loc[tdiffs < '00:00:00']
    # WALRUS OPERATOR YEAH!
    if (n_anomalous_tdiffs := len(anomalous_tdiffs)) > 0:
        LOGGER.warning(f"Identified {n_anomalous_tdiffs} malformed (negative tdelta in sequential rows) timestamps. Dropping from final dataframe")
        meas_df = meas_df.drop(anomalous_tdiffs.index).reset_index(drop=True)
    return meas_df


def parse_gps(gps_data, idx_of_em31):
    """
    Given a "cleaned" line of EM31 GPS data and its line number in the EM31 datafile, return a NMEA0183 sentence object

    input:
        gps_data: single line of NMEA0183 GPS data
        idx_of_em31: line number in R31/H31 file where gps_data resides
    output:
        gps_msg: parsed GPS message in pynmea object
    """
    try:
        gps_msg = pynmea2.parse(gps_data)
    except pynmea2.ChecksumError:
        LOGGER.warning(f"NMEA0183 checksum error with line {idx_of_em31}")
        LOGGER.debug(f"{gps_data!r}")
        return
    except pynmea2.ParseError:
        LOGGER.warning(f"NMEA0183 parse error with line {idx_of_em31}")
        LOGGER.debug(f"{gps_data!r}")
        return
    try:
        if not gps_msg.is_valid:
            LOGGER.warning(f"NMEA0183 validation failure with line {idx_of_em31}")
            LOGGER.debug(f"{gps_data!r}")
            return
        elif not gps_msg.latitude:
            # apparently .is_valid does not trigger False when msg.latitude = "N"
            return
    except ValueError: # apparently .is_valid can trigger ValueError instead of just returning False?!
        LOGGER.warning(f"NMEA0183 validation failure with line {idx_of_em31}")
        LOGGER.debug(f"{gps_data!r}")        
        return
    except AttributeError: # apparently not all NMEA0183 objects have .is_valid attr?!
        pass
    return gps_msg


def extract_gps(raw_data, epoch_ms, epoch_ts):
    """
    Extract the GPS information from a given EM31 dataset

    input:
        raw_data: list of data lines from R31/H31 data file
        epoch_ms: datalogger epoch referencing start-of-data-recording (?)
        epoch_ts: datalogger timestamp (computer time) at point of epoch_ms (?)
    output:
        meas_df: Pandas dataframe containing the parsed data
    """
    # detect where the GPS data chunks are in the EM31 data file
    gps_starts = [idx for idx, line in enumerate(raw_data) if line.startswith("@")]
    gps_ends = [idx for idx, line in enumerate(raw_data) if line.startswith("!")]
    try:
        assert len(gps_starts) == len(gps_ends)
        # extract only the gps messages from the data file contents
        gps_data = [raw_data[start:end] for start, end in zip(gps_starts, gps_ends)]
    except AssertionError:
        LOGGER.warning(f"N of GPS message-start chars ({len(gps_starts)}) does not equal N of GPS message-end chars ({len(gps_ends)}). Attempting to recover...")
        # we use the smaller of the two sizes, because it seems likely we would have to drop the data from the larger size anyway
        gps_data = []
        if len(gps_starts) > len(gps_ends):
            gps_starts = []
            for end_idx in gps_ends[:]:
                # walk backwards in raw_data from current end_idx, looking for gps_start char
                for line in raw_data[:end_idx][::-1]:
                    if line.startswith("@"):
                        chunk = raw_data[raw_data.index(line):end_idx]
                        # we don't even want to try recovering if the gps chunk isn't less than expected length
                        if len(chunk) <= 6:
                            gps_data.append(chunk)
                            # NB: this assumes <line> only shows up once in the entire datafile
                            gps_starts.append(raw_data.index(line))
                        else:
                            _ = gps_ends.pop(gps_ends.index(end_idx))
                        break
        else:
            gps_ends = []
            for start_idx in gps_starts[:]:
                # walk forwards in raw_data from current start_idx, looking for gps_end char
                for line in raw_data[start_idx:]:
                    if line.startswith("!"):
                        chunk = raw_data[start_idx:raw_data.index(line)]
                        # we don't even want to try recovering if the gps chunk isn't less than expected length
                        if len(chunk) <= 6:
                            gps_data.append(chunk)
                            # NB: this assumes <line> only shows up once in the entire datafile
                            gps_ends.append(raw_data.index(line))
                        else:
                            _ = gps_starts.pop(gps_starts.index(start_idx))
                        break
    # detect where EM31 measurements have been logged inside GPS messages
    bad_logs_idx = [
        gps_data.index(data)
        for data in gps_data
        if any(line.startswith("T") for line in data)
    ]
    if len(bad_logs_idx) != 0:
        LOGGER.warning(f"Detected {len(bad_logs_idx)} instances where EM31 data overrides GPS data")
        for bad_idx in bad_logs_idx:
            em31_line_idx = [
                gps_data[bad_idx].index(line)
                for line in gps_data[bad_idx]
                if line.startswith("T")
            ]
            # remove the em31 data from gps_data by pop-from-list
            for line_idx in em31_line_idx:
                _ = gps_data[bad_idx].pop(line_idx)
    # gps messages can vary in length but are never longer than 6 lines according to documentation
    assert all(len(data) <= 6 for data in gps_data)
    # grab the sys-time immediately after each gps message
    gps_times = []
    for end_idx, end in enumerate(gps_ends[:]):
        try:
            gps_times.append(int(raw_data[end : end + 1][0].split(" ")[-1]))
        except ValueError:
            # ignore the malformed gps_time and drop the data from other gps lists too
            for l in [gps_starts, gps_ends, gps_data]:
                _ = l.pop(end_idx)
    # drop the first character in each line and join the remainder chunks into 1 string
    gps_data_clean = ["".join([d[1:] for d in data]).strip() for data in gps_data]
    # extract NMEA0183 objects
    # NB: any invalid parsed-NMEA-objects will be silently ignored in later steps
    nmea = [
        parse_gps(clean_string, em31_idx)
        for clean_string, em31_idx in zip(gps_data_clean, gps_starts)
    ]
    # we only want GGA NMEA messages apparently (Time, position, and fix related data)
    # https://receiverhelp.trimble.com/alloy-gnss/en-us/NMEA-0183messages_GGA.html
    gga_idx = [idx for idx in range(len(nmea)) if isinstance(nmea[idx], pynmea2.GGA)]
    # if no GGA NMEA messages in data file, raise error
    if len(gga_idx) == 0:
        raise EM31GPSError("No valid GPS positioning data ($GNGGA) detected in file. Aborting...")
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
        LOGGER.warning(f"n_RMC ({len(rmc_idx)}) does not equal n_GGA ({len(gga_msgs)}) -> omitting sog/cmg from output")
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
    # nsats replace with 00
    gga_data[:, 3] = np.where(gga_data[:, 3] == "", "00", gga_data[:, 3])
    # hdop replace with NaN
    gga_data[:, 4] = np.where(gga_data[:, 4] == "", np.nan, gga_data[:, 4])
    gps_df = pd.DataFrame(
        data={
            "time_sys": pd.to_datetime(gga_data[:, 0]).astype("datetime64[ns]"),
            "time_gps": pd.Series(gga_data[:, 1]),
            "fix": pd.Series(gga_data[:, 2], dtype=PD_UINT8),
            "nsats": pd.Series(gga_data[:, 3], dtype=PD_UINT16),
            "hdop": pd.Series(gga_data[:, 4], dtype=PD_FLOAT32),
            "alt": pd.Series(gga_data[:, 5], dtype=PD_FLOAT32),
            "lat": pd.Series(gga_data[:, 6], dtype=PD_FLOAT64),
            "lat_dir": pd.Series(gga_data[:, 7], dtype=PD_STR),
            "lon": pd.Series(gga_data[:, 8], dtype=PD_FLOAT64),
            "lon_dir": pd.Series(gga_data[:, 9], dtype=PD_STR),
        }
    )
    # return only a subset of columns based on Josh's prior work
    cols = ["lat", "lon", "time_gps", "time_sys"]
    if use_rmc:
        gps_df["sog"] = pd.Series(rmc_data[:, 6], dtype=PD_FLOAT32)
        gps_df["cmg"] = pd.Series(rmc_data[:, 7], dtype=PD_FLOAT32)
        cols.extend(["sog", "cmg"])
    # isolate only the records with 0 < gps_quality < 6
    # different types of GGA quality indicators: https://receiverhelp.trimble.com/alloy-gnss/en-us/NMEA-0183messages_GGA.html
    subset = gps_df.loc[(gps_df["fix"] > 0) & (gps_df["fix"] < 6), cols]
    return subset


def thickness(em31_df, inst_height, coeffs=HAAS_2017):
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
    LOGGER.setLevel(logging.DEBUG)

    src_dir = Path("./data/em31")
    dst_dir = Path("./data/output")
    src_dir.mkdir(exist_ok=True)
    dst_dir.mkdir(exist_ok=True)
    for data_file in sorted(src_dir.rglob("???????*.?31")):
        target = dst_dir / f"{data_file.stem}.ttem.csv"
        if target.exists():
            LOGGER.info(f"Skipping existing file: {target.resolve()}")
            continue
        data_size_MB = data_file.stat().st_size / 1024**2
        LOGGER.info(f"Processing {data_file.resolve()} (~{data_size_MB:.2f} MB)")
        df = read_data(data_file)
        df = thickness(df, 0.15)
        df.to_csv(target, index=False, na_rep="NaN")
        LOGGER.info(f"Saved to CSV: {target.resolve()}")
