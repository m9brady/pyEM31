# pyEM31

This codebase was initiated by the late Dr. Josh King at Environment and Climate Change Canada (ECCC) with support from Dr. Christian Haas at the Alfred Wegener Institute (AWI). Efforts are being continued by Mike Brady (ECCC) in order to share Josh's work with the general science community.

## The EM31 Instrument

The Geonics EM31 sensor is designed to measure changes in conductivity through electromagnetic induction. Offered in two configurations (conventional EM31-MK2 and "short" EM31-SH), the device can be mounted on a trailer/toboggan and towed while traversing a given study area. Further information on the EM31 devices can be found at http://www.geonics.com/html/em31-mk2.html

At ECCC, the EM31-SH version is used (pictured below, mounted to a toboggan) to detect changes in conductivity in sea ice in order to estimate total ice thickness for a given area.

<img src="assets/em31sh_eccc.jpg" width=600>

*Supplementary Figure 1 from Haas et. al (2017) https://dx.doi.org/10.1002/2017GL075434*

## What this code does
This codebase takes the raw logger output from the EM31 and converts it into a pandas DataFrame for further analysis. If run as a script, all EM31 data files (`*.R31`) inside `./data/em31/` are converted into comma-separated value (`.csv`) text files containing measurement data and total thickness estimates.

## Quickstart

### Creating the runtime environment
```zsh
# assumes you have python3.10 installed
# create the environment
python3 -m venv .venv

# activate the environment
## linux
source ./.venv/bin/activate
## windows
.\.venv\Scripts\activate

# install requirements
python3 -m pip install -r requirements.txt
```

### Running the code
```zsh
python3 em31.py
```

### Using the functions interactively
```python
from em31 import read_r31, thickness
# gps_tol: acceptable gps time separation in seconds
# encoding: the specific encoding for the input data file (default windows-1252)
df = read_r31('./data/em31/datafile.R31', gps_tol=1, encoding='windows-1252')
# inst_height: height of instrument above surface (meters?)
# coeffs: 3-element list of coefficients for estimating thickness from EM31 measurements
df = thickness(df, inst_height=0.15, coeffs=HAAS_2010)
```


## Caveats
This code attempts to recover for NMEA0183 message corruption where EM31 data-logger lines intersect multi-line NMEA0183 GPS messages. If it finds any other problems - whether a parsing or checksum error for GPS data - it logs a message but makes no attempt to recover. Users can set debug-level logging to log the problematic NMEA0183 messages for further analysis:

```python
logger.setLevel(logging.DEBUG)
```
