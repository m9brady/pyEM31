# pyEM31

This codebase was initiated by the late Dr. Josh King at Environment and Climate Change Canada (ECCC) with support from Dr. Christian Haas at the Alfred Wegener Institute (AWI). This effort is being continued by Mike Brady (ECCC) and being made available to the general science community and the public.

## The EM31 Instrument

...EM31 blurb goes here with picture(s)...
http://www.geonics.com/html/em31-mk2.html
https://www.mathworks.com/help/nav/ref/nmeaparser-system-object.html#mw_b72d25cf-472c-4c4f-8197-85e2a2956480

## What this code does
This codebase takes the raw logger output from the EM31 and converts it into a pandas DataFrame for further analysis. If run as a script, all EM31 data files (`*.R31`) inside `./data/em31/` are converted into comma-separated value (`.csv`) text files containing measurement data and total thickness estimates.

## Quickstart

### Creating the runtime environment
```console
## assumes you have python3.10 installed
## create the environment
python3 -m venv .venv
## activate the environment
### linux
source ./.venv/bin/activate
### windows
.\.venv\Scripts\activate
## install requirements
python3 -m pip install -r requirements.txt
```

### Running the code
```console
python3 em31.py
```

### Using the functions interactively
```ipython
from em31 import read_r31, thickness
# gps_tol: acceptable gps time separation in seconds
# encoding: the specific encoding for the input data file (default windows-1252)
df = read_r31('./data/em31/datafile.R31', gps_tol=1, encoding='windows-1252')

```


## Caveats
This code tries recover for NMEA0183 message corruption where EM31 data-logger lines split NMEA0183 messages. If it finds any other problems - whether a parsing or checksum error for GPS data - it logs a message but makes no attempt to recover. You can use debug-level logging to log the problematic NMEA0183 messages.
