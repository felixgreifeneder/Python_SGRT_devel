import pytesmo.io.ismn.interface as ismn_interface
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sgrt_devels.extr_TS import extr_SIG0_LIA_ts_GEE as extrts
import ascat
import h5py
import datetime as dt

# outpath
outpath = '/mnt/SAT/Workspaces/GrF/Processing/S1ALPS/ISMN/s1_ascat_smap_comp/'

# initialise available ISMN data
ismn = ismn_interface.ISMN_Interface('/mnt/SAT/Workspaces/GrF/01_Data/InSitu/ISMN/')

# get list of available stations
available_stations = ismn.list_stations()

# initialise S1 SM retrieval
#mlmodel = pickle.load(open('/mnt/SAT/Workspaces/GrF/Processing/S1ALPS/all_alps_plus_se_gee/pwise/mlmodel.p', 'rb'))

# initialse text report
txtrep = open(outpath + 'report.txt', 'w')
txtrep.write('Name, R, RMSE\n')

xyplot = pd.DataFrame()
cntr = 1

# iterate through all available ISMN stations
for st_name in available_stations:
    try:
        station = ismn.get_station(st_name)
        station_vars = station.get_variables()

        if 'soil moisture' not in station_vars:
            continue

        station_depths = station.get_depths('soil moisture')

        if 0.05 not in station_depths[0]:
            continue

        print(st_name)
        sm_sensors = station.get_sensors('soil moisture', depth_from=0.05, depth_to=0.05)
        station_ts = station.read_variable('soil moisture', depth_from=0.05, depth_to=0.05, sensor=sm_sensors[0])

        # SENTINEL 1
        s1ts = extrts(station.longitude, station.latitude, 6000, maskwinter=False)
        # s1ts = extrts(36.0189, 0.425, 50, maskwinter=False)
        # s1pd = pd.Series(s1ts['130'][1]['sig0'], index=s1ts['130'][0])
        # s1pd.sort_index

        s1tracks = s1ts.keys()
        firsttrack = s1tracks[0]
        s1_df = pd.DataFrame({firsttrack: s1ts[firsttrack][1]['sig0']}, index=s1ts[firsttrack][0])
        s1_df.sort_index
        s1_df = s1_df.resample('D').mean()

        for trackid in s1ts.keys():
            if trackid == firsttrack:
                continue

            tmp = pd.Series(s1ts[trackid][1]['sig0'], index=s1ts[trackid][0], name=trackid)
            tmp.sort_index
            tmp = tmp.resample('D').mean()
            s1_df = s1_df.join(tmp, how='outer')

        # ASCAT

        ascat_db = ascat.AscatH109_SSM('/mnt/SAT/Workspaces/GrF/01_Data/ASCAT/h109/SM_ASCAT_TS12.5_DR2016/',
                                       '/mnt/SAT/Workspaces/GrF/01_Data/ASCAT/h109_auxiliary/grid/',
                                       grid_info_filename='TUW_WARP5_grid_info_2_1.nc',
                                       static_path='/mnt/SAT/Workspaces/GrF/01_Data/ASCAT/h109_auxiliary/static_layers/')

        ascat_series = ascat_db.read_ssm(station.longitude, station.latitude)
        valid = np.where((ascat_series.data['proc_flag'] == 0) & (ascat_series.data['ssf'] == 1) & (
        ascat_series.data['snow_prob'] < 20))
        ssm_series = pd.Series(data=ascat_series.data['sm'][valid[0]], index=ascat_series.data.index[valid[0]],
                               name='ASCAT')
        ASCAT_series = ssm_series.resample('D').mean()

        # SMAP

        # load file stack
        h5file = '/mnt/SAT4/DATA/S1_EODC/Sentinel-1_CSAR/IWGRDH/ancillary/datasets/SMAPL4/SMAPL4_SMC_2015.h5'
        ssm_stack = h5py.File(h5file, 'r')
        ssm_stack.close()
        ssm_stack = h5py.File(h5file, 'r')

        # find the nearest gridpoint
        easelat = ssm_stack['LATS']
        easelon = ssm_stack['LONS']
        mindist = 10

        dist = np.sqrt(np.power(station.longitude - easelon[:, :], 2) + np.power(station.latitude - easelat[:, :], 2))
        mindist_loc = np.unravel_index(dist.argmin(), dist.shape)

        # stack time series of the nearest grid-point
        ssm = np.array(ssm_stack['SM_array'][:, mindist_loc[0], mindist_loc[1]])
        #ssmmin = np.nanmin(ssm)
        #ssmmax = np.nanmax(ssm)
        #ssmrange = ssmmax - ssmmin
        #ssm = ((ssm - ssmmin) / ssmrange) * 100

        # create the time vector
        time_sec = np.array(ssm_stack['time'])
        time_dt = [dt.datetime(2000, 1, 1, 11, 58, 55, 816) + dt.timedelta(seconds=x) for x in time_sec]
        ssm_series = pd.Series(data=ssm, index=time_dt, name='SMAP')

        ssm_stack.close()

        SMAP_series = ssm_series.resample('D').mean()


        plotpath = outpath + st_name + '.png'

        station_ts_res = station_ts.data['soil moisture'].resample('D').mean()

        all_df = s1_df
        all_df = all_df.join(station_ts_res*100, how='outer')
        all_df = all_df.join(ASCAT_series, how='outer')
        all_df = all_df.join(SMAP_series*100, how='outer')

        # Interpolate and Plot
        all_df = all_df['2015-05-01':'2015-09-30']
        for k in s1ts.keys():
            all_df[k] = all_df[k].interpolate(method='linear')
        #all_df.fillna(axis=1, inplace=True)

        # plot
        plt.figure(figsize=(18, 6))
        #plt.plot(s1_ts.index, s1_ts, color='b', linestyle='-', marker='+', label='S1')
        #plt.plot(s1_df)
        #plt.plot(station_ts_res.index, station_ts_res*100)
        #plt.plot(ASCAT_series.index, ASCAT_series)
        #plt.plot(SMAP_series.index, SMAP_series)
        #plt.legend()
        #all_df.plot(secondary_y=s1ts.keys(), mark_right=False, linewidth=1.0)
        ax = all_df[['soil moisture', 'ASCAT', 'SMAP']].plot(linewidth=1.0)
        all_df[s1ts.keys()].plot(secondary_y=True, mark_right=False, linewidth=1.0, ax=ax)
        plt.show()
        plt.savefig(plotpath)
        plt.close()
    except:
        print('No data for: ' + st_name)


#xyplot.plot(x='x', y='y', color='r')
#plt.show()
#plt.savefig(outpath + 'scatterplot.png')

txtrep.close()






