import pytesmo.io.ismn.interface as ismn_interface
from sgrt_devels.derive_smc import extract_time_series_gee
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# outpath
outpath = '/mnt/SAT/Workspaces/GrF/Processing/S1ALPS/ASCAT/ISMN/pwise/'

# initialise available ISMN data
ismn = ismn_interface.ISMN_Interface('/mnt/SAT/Workspaces/GrF/01_Data/InSitu/ISMN/')

# get list of available stations
available_stations = ismn.list_stations()

# initialise S1 SM retrieval
mlmodel = pickle.load(open('/mnt/SAT/Workspaces/GrF/Processing/S1ALPS/ASCAT/gee/mlmodel.p', 'rb'))

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

        # get S1 time series
        s1_ts, error_ts = extract_time_series_gee(mlmodel,
                            '/mnt/SAT4/DATA/S1_EODC/',
                            outpath,
                            station.latitude,
                            station.longitude,
                            name=st_name,
                            footprint=1000)

        plotpath = outpath + st_name + '.png'

        #s1_ts = s1_ts[np.where(error_ts == error_ts.min())[0]]

        s1_ts_res = s1_ts.resample('D').mean()
        station_ts_res = station_ts.data.resample('D').mean() * 100.

        # calculate error metrics
        ts_bias = s1_ts_res.subtract(station_ts_res['soil moisture']).mean()
        #s1_ts_res = s1_ts_res - ts_bias
        #s1_ts = s1_ts - ts_bias

        xytmp = pd.concat({'x':s1_ts_res, 'y':station_ts_res}, join='inner', axis=1)
        if cntr == 1:
            xyplot = xytmp
        else:
            xyplot = pd.concat([xyplot, xytmp], axis=0)

        cntr = cntr + 1

        # subset
        #s1_ts_res = s1_ts_res['20150401':'20151031']
        #station_ts_res = station_ts_res['20150401':'20151031']

        ts_cor = s1_ts_res.corr(station_ts_res['soil moisture'])
        ts_rmse = np.sqrt(np.nanmean(np.square(s1_ts_res.subtract(station_ts_res['soil moisture']))))
        print('R: ' + str(ts_cor))
        print('RMSE: ' + str(ts_rmse))
        txtrep.write(st_name + ', ' + str(ts_cor) + ', ' + str(ts_rmse) + '\n')

        # plot
        plt.figure(figsize=(18, 6))
        plt.plot(s1_ts.index, s1_ts, color='b', linestyle='-', marker='+', label='S1')
        plt.plot(station_ts.data.index, station_ts.data['soil moisture'] * 100.)
        plt.legend()
        plt.show()
        plt.savefig(plotpath)
        plt.close()
    except:
        print('No data for: ' + st_name)


xyplot.plot.scatter(x='x', y='y', color='r')
plt.show()
plt.savefig(outpath + 'scatterplot.png')

txtrep.close()






