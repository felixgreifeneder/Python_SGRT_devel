import pandas as pd
from sgrt_devels.extr_TS import extr_SIG0_LIA_ts_GEE as extrts
from matplotlib import pyplot as plt
import datetime as dt
from pytesmo.time_series import anomaly
import ascat
import numpy as np
import h5py


lon = 12.19
lat = 48.43

# SENTINEL1

s1ts = extrts(lon, lat, 6000, maskwinter=False)
#s1ts = extrts(36.0189, 0.425, 50, maskwinter=False)
#s1pd = pd.Series(s1ts['130'][1]['sig0'], index=s1ts['130'][0])
#s1pd.sort_index

s1tracks = s1ts.keys()
firsttrack = s1tracks[0]
ts_df = pd.DataFrame({firsttrack: s1ts[firsttrack][1]['sig0']}, index=s1ts[firsttrack][0])
ts_df.sort_index
ts_df = ts_df.resample('D').mean()

for trackid in s1ts.keys():
    if trackid == firsttrack:
        continue

    tmp = pd.Series(s1ts[trackid][1]['sig0'], index=s1ts[trackid][0], name=trackid)
    tmp.sort_index
    tmp = tmp.resample('D').mean()
    ts_df = ts_df.join(tmp, how='outer')

# ASCAT

ascat_db = ascat.AscatH109_SSM('/mnt/SAT/Workspaces/GrF/01_Data/ASCAT/h109/SM_ASCAT_TS12.5_DR2016/',
                                           '/mnt/SAT/Workspaces/GrF/01_Data/ASCAT/h109_auxiliary/grid/',
                                           grid_info_filename = 'TUW_WARP5_grid_info_2_1.nc',
                                           static_path = '/mnt/SAT/Workspaces/GrF/01_Data/ASCAT/h109_auxiliary/static_layers/')

ascat_series = ascat_db.read_ssm(lon, lat)
valid = np.where((ascat_series.data['proc_flag'] == 0) & (ascat_series.data['ssf'] == 1) & (ascat_series.data['snow_prob'] < 20))
ssm_series = pd.Series(data=ascat_series.data['sm'][valid[0]], index=ascat_series.data.index[valid[0]], name='ASCAT')
ssm_series = ssm_series.resample('D').mean()

ts_df = ts_df.join(ssm_series, how='outer')

# SMAP

#load file stack
h5file = '/mnt/SAT4/DATA/S1_EODC/Sentinel-1_CSAR/IWGRDH/ancillary/datasets/SMAPL4/SMAPL4_SMC_2015.h5'
ssm_stack = h5py.File(h5file, 'r')
ssm_stack.close()
ssm_stack = h5py.File(h5file, 'r')

# find the nearest gridpoint
easelat = ssm_stack['LATS']
easelon = ssm_stack['LONS']
mindist = 10

dist = np.sqrt(np.power(lon - easelon[:,:], 2) + np.power(lat - easelat[:,:], 2))
mindist_loc = np.unravel_index(dist.argmin(), dist.shape)

# stack time series of the nearest grid-point
ssm = np.array(ssm_stack['SM_array'][:,mindist_loc[0], mindist_loc[1]])
ssmmin = np.nanmin(ssm)
ssmmax = np.nanmax(ssm)
ssmrange = ssmmax - ssmmin
ssm = ((ssm - ssmmin) / ssmrange) * 100

# create the time vector
time_sec = np.array(ssm_stack['time'])
time_dt = [dt.datetime(2000,1,1,11,58,55,816) + dt.timedelta(seconds=x) for x in time_sec]
ssm_series = pd.Series(data=ssm, index=time_dt, name='SMAP')

ssm_stack.close()

ssm_series = ssm_series.resample('D').mean()
ts_df = ts_df.join(ssm_series, how='outer')

# Interpolate and Plot
ts_df = ts_df['2015-01-01':'2015-12-31']
ts_df.interpolate(inplace=True)



plt.figure(figsize=(6.3,3.7))
# ts_df['ASCAT'].plot(style='g')
# ts_df['SMAP'].plot(style='b')
# print(s1ts.keys())
# for i in s1ts.keys():
#     ts_df[i].plot(style='r', secondary_y=True)
ts_df.plot(secondary_y=s1ts.keys(), mark_right=False)
#plt.xlabel('Date', fontsize=10)
#plt.ylabel('SMC [m3/m-3]')
#plt.ylim((0,50))
#plt.legend()
plt.show()
plt.savefig('/mnt/SAT/Workspaces/GrF/Processing/S1ALPS/s1_ts_plots/s1_ascat_smap_landshut.png', dpi=300)
plt.close()

# plt.figure(figsize=(6.3,3.7))
# s1pd_anom.plot()
# timex = pd.date_range('30/9/2014', '30/4/2017', freq='D')
# zerolist = pd.Series([0]*len(timex), index=timex)
# zerolist.plot(linestyle='--', color='r')
# #plt.plot(timex, [0]*len(timex), linestyle='--', color='r', marker='*')
# plt.xlabel('Date', fontsize=10)
# plt.ylabel('SMC anomaly')
# plt.show()
# plt.savefig('/mnt/SAT/Workspaces/GrF/02_Documents/ISRSE2017/s1ts_anomaly.png', dpi=300)
# plt.close()
