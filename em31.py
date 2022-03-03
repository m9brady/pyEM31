'''
Processing functions for the EM31
J.King 2022

Many thanks to Christian Haas for the help with this
'''

import numpy as np
import pandas as pd
import pynmea2
from datetime import datetime, timedelta

#Constants
HAAS_2010 = [0.98229, 13.404, 1366.4]

def text_to_bits(text, encoding='windows-1252', errors='surrogatepass'):
    bits = bin(int.from_bytes(text.encode(encoding, errors), 'big'))[2:]
    return bits.zfill(8 * ((len(bits) + 7) // 8))

def read_r31(filename, gps_tol = 1, encoding='windows-1252'):
    '''
    Load R31 files output from the EM31
        Input:
            filename: Path to the R31 file
            gps_tol: GPS time tolerance (seconds)
            encoding: Encoding of the file
        Output:

    '''

    r31_file = open(filename, 'r', encoding=encoding)
    r31_dat = r31_file.readlines()

    # Get header data
    if r31_dat[0].startswith('E'):
        h_ident = r31_dat[0][0:7] # System ident
        h_ver = r31_dat[0][8:12] # Software version
        h_type = r31_dat[0][12:15] # GPS or GRD (Grid)
        h_units = int(r31_dat[0][15:16]) # 0 = meter, 1 = feet
        h_dipole = int(r31_dat[0][16:17]) # 0 = vertical, 1 = horizontal
        h_mode = int(r31_dat[0][17:18]) # 0 = auto, 1 = wheel, 2 = manual
        h_component = int(r31_dat[0][18:19]) # 0 = Both, 1 = Iphase
        h_computer = int(r31_dat[0][22:23]) # no info on what this is
    else:
        print('Error: First line is not a header')
        return 1

    # Get file data fields
    if r31_dat[1].startswith('H'):
        h_file = r31_dat[1][2:11]
        h_time = r31_dat[1][13:18]
    else:
       print('Error: Data fields missing')
       return 1 

    # Get file start stamp
    if r31_dat[5].startswith('Z'):
        h_date = r31_dat[5][1:9]
        h_time = r31_dat[5][10:18]
    else:
       print('Error: Start stamp missing')
       return 1 

    # Get timer relation
    if r31_dat[6].startswith('*'):
        epoch_time = r31_dat[6][1:13]
        epoch_ms = int(r31_dat[6][13:23])
        epoch_ts = datetime.strptime(h_date + ' ' + epoch_time, '%d%m%Y %H:%M:%S.%f')
    
    # Extract the measurements and gps
    meas_df = extract_measurements(r31_dat, epoch_ms, epoch_ts, encoding=encoding)
    gps_df = extract_gps(r31_dat, epoch_ms, epoch_ts)

    # Merge based on time
    em31_merged = pd.merge_asof(meas_df, gps_df[['lat', 'lon', 'time_gps', 'time_sys']], 
                            left_on='time_ds', right_on='time_sys', direction='nearest')
    
    # Remove measurements where GPS is desynced
    # TODO: Make sure time_sys is what we think it is
    time_diff = em31_merged['time_ds'] - em31_merged['time_sys']
    em31_merged = em31_merged.loc[time_diff < timedelta(seconds=gps_tol)]
    return em31_merged

def extract_measurements(r31_dat, epoch_ms, epoch_ts, encoding):
   # Read the measurements into DF
    meas_idx = []; meas_df = pd.DataFrame()
    for l_num, line in enumerate(r31_dat):
        if line.startswith('T'):
            meas_idx.append(l_num)
            meas_gn = text_to_bits(line[1], encoding=encoding)
            meas_range3 = int(meas_gn[5])
            meas_range2 = int(meas_gn[6])
            
            meas_read1 = float(line[2:7])
            meas_read2 = float(line[7:12])
            meas_time = int(line[13:23])
            
            if (meas_range2 == 1) and (meas_range3 == 1):
                c_factor = -0.25
                i_factor = -0.0025
                app_cond =  meas_read1 * c_factor
                in_phase = meas_read2 * i_factor
            else:
                app_cond = np.nan
  
            record_df = pd.DataFrame({'l_num': [l_num],
                                    'time_ms': [meas_time],
                                    'flags': [meas_gn],
                                    'range2': [meas_range3],
                                    'range3': [meas_range3],
                                    'c_factor': [c_factor],
                                    'appcond': [app_cond],
                                    'inph': in_phase})
            meas_df = pd.concat([meas_df, record_df])
    
    # Create measurement time stamps
    meas_df['time_relative'] = meas_df['time_ms'] - epoch_ms
    meas_df['time_ds'] = np.repeat(epoch_ts, len(meas_df)) + [timedelta(milliseconds=rel) for rel in meas_df['time_relative']]
    
    return meas_df.reset_index(drop = True)

# TODO interpolate the GPS data instead of taking nearest
def extract_gps(r31_dat, epoch_ms, epoch_ts):
    count = 0
    gps_sentances = []; gps_sys_ts = []
    for num, line in enumerate(r31_dat):
        if line.startswith('@'):
            count = count + 1
            # Find the GPS ending line (No more than 6 away)
            gps_end = [l.startswith('!') for l in r31_dat[num:num+6]]
            gps_end_idx = int(num + np.where(gps_end)[0])
            #Parse the GPS data
            gps_data = r31_dat[num:gps_end_idx]
            gps_data_clean = [dat[1:-1] for dat in gps_data]
            gps_data_str = ''.join(gps_data_clean).strip()
            try:
                gps_msg = pynmea2.parse(gps_data_str)
            except:
                print('bad gps encoding @ line {}'.format(num))
                
            if isinstance(gps_msg, pynmea2.types.talker.GGA):
                gps_end_line = r31_dat[gps_end_idx]
                gps_sys_ms = int(gps_end_line[13:23])
                gps_sys_rel = gps_sys_ms - epoch_ms
                gps_sys_ts.append(epoch_ts + timedelta(milliseconds=gps_sys_rel))
                gps_sentances.append(gps_msg)
        
    gps_df = pd.DataFrame()
    for sidx, msg in enumerate(gps_sentances):
        record_df = pd.DataFrame({'time_sys':[gps_sys_ts[sidx]],
                                'time_gps':[msg.timestamp],
                                'fix': [msg.gps_qual],
                                'nsats': [msg.num_sats],
                                'hdop': [msg.horizontal_dil],
                                'alt': [msg.altitude],
                                'lat': [msg.latitude],
                                'lat_dir': [msg.lat_dir],
                                'lon': [msg.longitude],
                                'lon_dir': [msg.lon_dir]})
        gps_df = pd.concat([gps_df,record_df])
    
    gps_df = gps_df.loc[(gps_df['fix'] > 0) & (gps_df['fix'] <6)]

    return gps_df


def thickness(em31_df, inst_height, coeffs=HAAS_2010):
    '''
    Estaimte total thickness from apparent conductivity
        input:
            em31_df: pyEM31 dataframe
            inst_height: height of the instrument above the snow surface
            coeffs: 3 element list of retrieval coefficients
        output:
            em31_df: pyEM31 dataframe with total thickness
    '''
    em31_df['ttem'] = -1/coeffs[0] * np.log((em31_df['appcond']-coeffs[1])/coeffs[2])
    em31_df['ttem'] -= inst_height
    return em31_df