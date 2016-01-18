__author__ = 'felix'

import ee
import numpy as np
import devel_fgreifen2.extr_TS as extr_TS
import datetime
import os
import fnmatch
import devel_fgreifen2.jdutil as jdutil
from scipy import stats
import matplotlib.pyplot as plt
import random
from progressbar import *
import math
from time import time


def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return(result)


def sample_pixel_area_sig0(lon, lat, pxsize, landcover, asapath, outpath):
    #this function calculates an average sig0 over a certain area by sampling it with 5 random points
    pxlist = []
    #pxlist.append([lon,lat])
    random.seed()
    goodlc = np.array([10,12,13,15,16,17,18,19,20,21,26,27,28,29])

    meanSIG0 = []
    SIG0list = []
    SIG0Nlist = []
    LIAlist = []
    DAYlist = []

    while len(pxlist) <= 10:
        if len(pxlist) == 0:
            rlon = lon
            rlat = lat
        else:
            rlon = random.uniform(lon-(pxsize/3),lon+(pxsize/3))
            rlat = random.uniform(lat-(pxsize/3),lat+(pxsize/3))

        lc = extr_TS.extr_raster_pixel_values(landcover, 1, rlat, rlon, 1)
        if np.any(goodlc == lc):
            tmp = extr_TS.extr_SIG0_LIA_ts(asapath, 'ASAWS', 'SGRT21A01', 'resampled', 75, rlon, rlat)
            tmpSIG0 = tmp[1]
            tmpLIA=tmp[2]
            #normalise SIG0 to a constant incidence angle of 30deg
            mask = np.array((tmpLIA > 2000) & (tmpLIA < 6000) & (tmpSIG0 != -9999)).squeeze()
            slope = stats.linregress(tmpLIA[mask], tmpSIG0[mask])[0]
            # print(slope)
            tmpSIG0norm = tmpSIG0 - (slope*(tmpLIA-3000))
            tmpSIG0norm[np.logical_not(mask)] = -9999
            SIG0list.append(tmp[1])
            SIG0Nlist.append(tmpSIG0norm)
            LIAlist.append(tmp[2])
            DAYlist.append(tmp[0])
            meanSIG0.append(10*math.log10(np.power(10,(np.array(tmpSIG0norm[mask])/10)).mean()))

            pxlist.append([rlon,rlat])

    #find minimum and maximum day
    minDay = min([x for xs in DAYlist for x in xs])
    maxDay = max([x for xs in DAYlist for x in xs])
    ndays = int(maxDay) - int(minDay)
    ndays = ndays + 1

    # create arrays for SIG0 and LIA
    DAYseq = range(int(minDay), int(maxDay)+1)
    SIG0arr = np.zeros((ndays, len(pxlist)+1), dtype=np.float)
    SIG0arr[:,1:] = -9999
    SIG0Narr = np.zeros((ndays, len(pxlist)+1), dtype=np.float)
    SIG0Narr[:,1:] = -9999
    LIAarr = np.zeros((ndays, len(pxlist)+1), dtype=np.float)
    LIAarr[:,1:] = -9999

    for i in range(ndays):
        SIG0arr[i,0] = DAYseq[i]
        SIG0Narr[i,0] = DAYseq[i]
        LIAarr[i,0] = DAYseq[i]

    # fill SIG0 and LIA into array
    for i in range(len(pxlist)):
        for j in range(len(SIG0list[i])):
            SIG0arr[SIG0arr[:,0] == DAYlist[i][j], i+1] = SIG0list[i][j]
            SIG0Narr[SIG0Narr[:,0] == DAYlist[i][j], i+1] = SIG0Nlist[i][j]
            LIAarr[LIAarr[:,0] == DAYlist[i][j], i+1] = LIAlist[i][j]

    # create array with averages
    outSIG0arr = np.zeros((ndays,2))
    outSIG0Narr = np.zeros((ndays,2))
    outLIAarr = np.zeros((ndays,2))

    for i in range(ndays):
        outSIG0arr[i,0] = DAYseq[i]
        outSIG0Narr[i,0] = DAYseq[i]
        outLIAarr[i,0] = DAYseq[i]

    SIG0arr[SIG0arr == -9999] = np.nan
    SIG0Narr[SIG0Narr == -9999] = np.nan
    LIAarr[LIAarr == -9999] = np.nan
    outSIG0arr[:,1] = 10*np.log10(np.nanmean(np.power(10,SIG0arr[:,1:]/10), axis=1))
    outSIG0Narr[:,1] = 10*np.log10(np.nanmean(np.power(10,SIG0Narr[:,1:]/10), axis=1))
    outLIAarr[:,1] = np.nanmean(LIAarr[:,1:], axis=1)
    outSIG0arr[np.isfinite(outSIG0arr) == False] = -9999
    outSIG0Narr[np.isfinite(outSIG0Narr) == False] = -9999
    outLIAarr[np.isfinite(outLIAarr) == False] = -9999

    return (outSIG0arr.astype(np.int), outSIG0Narr.astype(np.int), outLIAarr.astype(np.int), pxlist, np.nanmean(meanSIG0))


def sample_pixel_area_raster(pxlist, rasterpath, dtype, maj):
    pxvals = []
    for i in range(len(pxlist)):
        tmp = extr_TS.extr_raster_pixel_values(rasterpath, 1, pxlist[i][1], pxlist[i][0], dtype)
        pxvals.append(tmp)
    pxvals = np.array(pxvals, dtype=np.float32)
    if maj == True: out_maj = np.argmax(np.bincount(np.array(pxvals,dtype=np.int)))
    pxvals[pxvals == -9999] = np.nan
    pxvals[pxvals == 32767] = np.nan
    out_pxval = np.nanmean(pxvals)
    if out_pxval == np.nan: out_pxval = -9999
    if maj == True:
        return((out_pxval, out_maj))
    else:
        return(out_pxval)


def extract_params(situ_path, point_list, asa_path, lc_path, dem_path, outpath):

    INSITUlist =[]
    DAYlist = []

    # retriev locations to extract training data from
    pointListTab = np.recfromcsv(point_list, delimiter=',', names=True)
    #pointListTab = pointListTab[0:3]

    # retrieve SMC in-situ data from stations
    print("ERA-Land")
    pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=pointListTab.shape[0]).start()

    for i in range(pointListTab.shape[0]):
        tmp = extr_TS.extr_ERA_SMC(situ_path, pointListTab[i][0], pointListTab[i][1])
        INSITUlist.append(tmp[:, 1])
        DAYlist.append(tmp[:, 0])
        # pointListTab[i][0] = tmp[1][0]
        # pointListTab[i][1] = tmp[1][1]
        pbar.update(i+1)
    pbar.finish()

    minDay = min([x for xs in DAYlist for x in xs])
    maxDay = max([x for xs in DAYlist for x in xs])
    ndays = int(maxDay) - int(minDay)
    ndays = ndays + 1

    DAYseq = range(int(minDay), int(maxDay)+1)
    INSITUarr = np.zeros((ndays, pointListTab.shape[0]+1))
    INSITUarr[:,1:] = -9999

    for i in range(ndays):
        INSITUarr[i,0] = DAYseq[i]

    for i in range(pointListTab.shape[0]):
        for j in range(len(INSITUlist[i])):
            tmpsmc = INSITUlist[i][j]
            if tmpsmc == -9999: tmpsmc = -9999
            INSITUarr[INSITUarr[:,0] == DAYlist[i][j], i+1] = tmpsmc

    # ---------------------------------------------------------------------
    # extract SIG0 and LIA
    # ---------------------------------------------------------------------

    print('SIG0 and LIA')

    pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=pointListTab.shape[0]).start()
    SIG0list = []
    SIG0Nlist = []
    LIAlist = []
    DAYlist = []
    pxcoordlist = []
    meanList = []
    for i in range(pointListTab.shape[0]):
        tmp = sample_pixel_area_sig0(pointListTab[i][0], pointListTab[i][1], 0.125, lc_path, asa_path, outpath)
        #tmp = extr_TS.extr_SIG0_LIA_ts(asa_path, 'ASAWS', 'SGRT21A01', 'resampled', 75, pointListTab[i][0], pointListTab[i][1])
        SIG0list.append(list(tmp[0][:,1]))
        SIG0Nlist.append(list(tmp[1][:,1]))
        LIAlist.append(list(tmp[2][:,1]))
        DAYlist.append(list(tmp[0][:,0]))
        pxcoordlist.append(tmp[3])
        if np.isfinite(tmp[4]):
            meanList.append(int(tmp[4]))
        else:
            meanList.append(-9999)
        pbar.update(i+1)
    pbar.finish()

    MEANarr = np.array(meanList)

    minDay = min([x for xs in DAYlist for x in xs])
    maxDay = max([x for xs in DAYlist for x in xs])
    ndays = int(maxDay) - int(minDay)
    ndays = ndays + 1

    # create arrays for SIG0 and LIA
    DAYseq = range(int(minDay), int(maxDay)+1)
    SIG0arr = np.zeros((ndays, pointListTab.shape[0]+1))
    SIG0arr[:,1:] = -9999
    SIG0Narr = np.zeros((ndays, pointListTab.shape[0]+1))
    SIG0Narr[:,1:] = -9999
    LIAarr = np.zeros((ndays, pointListTab.shape[0]+1))
    LIAarr[:,1:] = -9999

    for i in range(ndays):
        SIG0arr[i,0] = DAYseq[i]
        SIG0Narr[i,0] = DAYseq[i]
        LIAarr[i,0] = DAYseq[i]

    # fill SIG0 and LIA into array
    for i in range(pointListTab.shape[0]):
        for j in range(len(SIG0list[i])):
            SIG0arr[SIG0arr[:,0] == DAYlist[i][j], i+1] = SIG0list[i][j]
            SIG0Narr[SIG0Narr[:,0] == DAYlist[i][j], i+1] = SIG0Nlist[i][j]
            LIAarr[LIAarr[:,0] == DAYlist[i][j], i+1] = LIAlist[i][j]

    # -------------------------------------------------------------------
    # Land Cover
    # -------------------------------------------------------------------

    print('Land-Cover')

    pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=pointListTab.shape[0]).start()
    LC = np.zeros((pointListTab.shape[0]))
    for i in range(pointListTab.shape[0]):
        LC[i] = sample_pixel_area_raster(pxcoordlist[i], lc_path, 1, True)[1]
        pbar.update()
    pbar.finish()

    # --------------------------------------------------------------------
    # Terrain
    # --------------------------------------------------------------------
    # the 'dem_path' parameter must point to a file holding elevation information.
    # The same directory has to include an aspect and a slope file withe the appendix
    # _slope and _aspect, respectively

    print('Terrain')

    pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=pointListTab.shape[0]).start()
    slope_path = dem_path[:-4] + '_slope.tif'
    aspect_path = dem_path[:-4] + '_aspect.tif'

    TERRAIN = np.zeros((3,pointListTab.shape[0]))
    TERRAIN[:,:] = -9999
    for i in range(pointListTab.shape[0]):
        TERRAIN[0,i] = sample_pixel_area_raster(pxcoordlist[i], dem_path, 3, False)
        # slope
        TERRAIN[1,i] = sample_pixel_area_raster(pxcoordlist[i], slope_path, 3, False)
        # aspect
        TERRAIN[2,i] = sample_pixel_area_raster(pxcoordlist[i], aspect_path, 3, False)
        pbar.update(i+1)
    pbar.finish()

    # -----------------------------------------------------------------
    # NDVI
    # -----------------------------------------------------------------
    # extract NDVI for each point from MODIS NDVI time-series, retrieved from
    # google earth engine

    # print('NDVI')
    #
    # pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=pointListTab.shape[0]).start()
    # ee.Initialize()
    # modis_ndvi = ee.ImageCollection('MODIS/MOD09GA_NDVI')
    # tchunks = math.trunc((maxDay - minDay + 1) / 100)
    # NDVIlist = []
    #
    # # extract modis time_series in 100 day chuncks
    #
    # for i in range(pointListTab.shape[0]):
    #
    #     tmp_ts_mean = np.zeros(ndays, dtype=np.float32)
    #     tmp_ts_mean[:] = -1
    #     pos = ee.Geometry.MultiPoint(pxcoordlist[i])
    #     cntr = 0
    #
    #     for j in range(1,tchunks+1):
    #         tmp_time = []
    #         tmp_ts =[]
    #         modis_ndvi_filt = modis_ndvi.filterDate(str(datetime.date.fromordinal(minDay+100*(j-1))),
    #                                                 str(datetime.date.fromordinal(minDay+100*j-1)))
    #         tmp = modis_ndvi_filt.getRegion(pos, scale=150).getInfo()
    #         for k in range(1, len(tmp)): tmp_time.append(tmp[k][3])
    #         unique_time = np.unique(np.array(tmp_time))
    #         for k in range(1, len(tmp)): tmp_ts.append(tmp[k][4])
    #         tmp_ts = np.array(tmp_ts, dtype=np.float32)
    #         tmp_ts[tmp_ts == -1] = np.nan
    #         for k in unique_time:
    #             tndvi = np.nanmean(tmp_ts[tmp_time == k])
    #             if np.isfinite(tndvi):
    #                 tmp_ts_mean[cntr] = tndvi
    #             cntr = cntr + 1
    #
    #         del modis_ndvi_filt
    #
    #     if ((maxDay - minDay + 1) % 100) > 0:
    #         tmp_time = []
    #         tmp_ts =[]
    #         modis_ndvi_filt = modis_ndvi.filterDate(str(datetime.date.fromordinal(minDay+100*tchunks)),
    #                                                 str(datetime.date.fromordinal(maxDay)))
    #         tmp = modis_ndvi_filt.getRegion(pos, scale=150).getInfo()
    #         for k in range(1, len(tmp)): tmp_time.append(tmp[k][3])
    #         unique_time = np.unique(np.array(tmp_time))
    #         for k in range(1, len(tmp)): tmp_ts.append(tmp[k][4])
    #         tmp_ts = np.array(tmp_ts, dtype=np.float32)
    #         tmp_ts[tmp_ts == -1] = np.nan
    #         for k in unique_time:
    #             tndvi = np.nanmean(tmp_ts[tmp_time == k])
    #             if np.isfinite(tndvi):
    #                 tmp_ts_mean[cntr] = tndvi
    #             cntr = cntr + 1
    #
    #         del modis_ndvi_filt
    #
    #     NDVIlist.append(list(tmp_ts_mean))
    #     pbar.update(i+1)
    # pbar.finish()


    #
    #
    # modis_ndvi_filt1 = modis_ndvi.filterDate(str(datetime.date.fromordinal(minDay)), str(datetime.date.fromordinal(minDay+1*((maxDay-minDay)/3))))
    # modis_ndvi_filt2 = modis_ndvi.filterDate(str(datetime.date.fromordinal(1+minDay+1*((maxDay-minDay)/3))), str(datetime.date.fromordinal(minDay+2*((maxDay-minDay)/3))))
    # modis_ndvi_filt3 = modis_ndvi.filterDate(str(datetime.date.fromordinal(1+minDay+2*((maxDay-minDay)/3))), str(datetime.date.fromordinal(minDay+3*((maxDay-minDay)/3))))
    #
    # NDVIlist = []
    # for i in range(pointListTab.shape[0]):
    #     #pos = ee.Geometry.Point(pointListTab[i][0], pointListTab[i][1])
    #     pos = ee.Geometry.MultiPoint(pxcoordlist[i])
    #     #first part
    #     tmp = modis_ndvi_filt1.getRegion(pos, scale=150).getInfo()
    #     tmp_ts =[]
    #     tmp_ts_mean = []
    #     tmp_time = []
    #     for j in range(1, len(tmp)): tmp_time.append(tmp[j][3])
    #     unique_time = np.unique(np.array(tmp_time))
    #     for j in range(1, len(tmp)): tmp_ts.append(tmp[j][4])
    #     tmp_ts = np.array(tmp_ts, dtype=np.float)
    #     tmp_ts[tmp_ts == -1] = np.nan
    #     for j in unique_time:
    #         tndvi = np.nanmean(tmp_ts[tmp_time == j])
    #         if np.isfinite(tndvi):
    #             tmp_ts_mean.append(tndvi)
    #         else:
    #             tmp_ts_mean.append(-1)
    #
    #     #second part
    #     tmp_ts =[]
    #     tmp_time = []
    #     tmp = modis_ndvi_filt2.getRegion(pos, scale=150).getInfo()
    #     for j in range(1, len(tmp)): tmp_time.append(tmp[j][3])
    #     unique_time = np.unique(np.array(tmp_time))
    #     for j in range(1, len(tmp)): tmp_ts.append(tmp[j][4])
    #     tmp_ts = np.array(tmp_ts, dtype=np.float)
    #     tmp_ts[tmp_ts == -1] = np.nan
    #     for j in unique_time:
    #         tndvi = np.nanmean(tmp_ts[tmp_time == j])
    #         if np.isfinite(tndvi):
    #             tmp_ts_mean.append(tndvi)
    #         else:
    #             tmp_ts_mean.append(-1)
    #     #third part
    #     tmp_ts =[]
    #     tmp_time = []
    #     tmp = modis_ndvi_filt3.getRegion(pos, scale=150).getInfo()
    #     for j in range(1, len(tmp)): tmp_time.append(tmp[j][3])
    #     unique_time = np.unique(np.array(tmp_time))
    #     for j in range(1, len(tmp)): tmp_ts.append(tmp[j][4])
    #     tmp_ts = np.array(tmp_ts, dtype=np.float)
    #     tmp_ts[tmp_ts == -1] = np.nan
    #     for j in unique_time:
    #         tndvi = np.nanmean(tmp_ts[tmp_time == j])
    #         if np.isfinite(tndvi):
    #             tmp_ts_mean.append(tndvi)
    #         else:
    #             tmp_ts_mean.append(-1)
    #
    #     NDVIlist.append(list(tmp_ts_mean))
    #     pbar.update(i+1)
    # pbar.finish()

    # fill in np array

    # NDVIarr = np.zeros((ndays, pointListTab.shape[0]+1))
    # NDVIarr[:,1:] = -9999
    #
    # for i in range(ndays): NDVIarr[i,0] = DAYseq[i]
    # for i in range(pointListTab.shape[0]):
    #     for j in range(len(NDVIlist[i])):
    #         NDVIarr[j, i+1] = NDVIlist[i][j]


    np.savetxt(outpath + '/SIGarr.csv', SIG0arr, delimiter=',')
    np.savetxt(outpath + '/SIGNarr.csv', SIG0Narr, delimiter=',')
    np.savetxt(outpath + '/LIAarr.csv', LIAarr, delimiter=',')
    np.savetxt(outpath + '/LCarr.csv', LC, delimiter=',')
    np.savetxt(outpath + '/TERRAINarr.csv', TERRAIN, delimiter=',')
    # np.savetxt(outpath + '/NDVIarr.csv', NDVIarr, delimiter=',')
    np.savetxt(outpath + '/MEANarr.csv', MEANarr, delimiter=',')

    np.savez(outpath + '/NParrays.npz', SIG0arr=SIG0arr, SIG0Narr=SIG0Narr, LIAarr=LIAarr, LC=LC, TERRAIN=TERRAIN, INSITUarr=INSITUarr, MEANarr=MEANarr)


def merge_params(inpath, outpath):
    # --------------------------------------------------------------------
    # Create table, combining all parameters
    # --------------------------------------------------------------------
    # parameter order: SMC, SIG0, LIA, NDVI, Height, Slope, Aspect, LC, date, station
    npfiles = np.load(inpath + '/NParrays.npz')
    INSITUarr = npfiles['INSITUarr']
    SIG0arr = npfiles['SIG0arr']
    SIG0Narr = npfiles['SIG0Narr']
    LIAarr = npfiles['LIAarr']
    # NDVIarr = npfiles['NDVIarr']
    MEANarr = npfiles['MEANarr']
    TERRAIN = npfiles['TERRAIN']
    LC = npfiles['LC']

    PARAMarr = np.array([0,1,2,3,4,5,6,7,8,9,10])

    for i in range(1,INSITUarr.shape[1]):
        for j in range(INSITUarr.shape[0]):
            #PARAMrow = []
            # in-Situ SMC
            PARAMrow = np.asarray(INSITUarr[j,i])
            # SIG0 and LIA
            if INSITUarr[j,0] in SIG0arr[:,0]:
                SIG0 = SIG0arr[SIG0arr[:,0] == INSITUarr[j,0], i]
                SIG0N = SIG0Narr[SIG0Narr[:,0] == INSITUarr[j,0], i]
                LIA = LIAarr[LIAarr[:,0] == INSITUarr[j,0], i]
            else:
                SIG0 = -9999
                SIG0N = -9999
                LIA = -9999
            PARAMrow = np.hstack((PARAMrow, SIG0))
            PARAMrow = np.hstack((PARAMrow, SIG0N))
            PARAMrow = np.hstack((PARAMrow, LIA))
            # MEAN SIG0
            PARAMrow = np.hstack((PARAMrow, MEANarr[i-1]))
            #NDVI
            # if INSITUarr[j,0] in NDVIarr[:,0]:
            #     NDVI = NDVIarr[NDVIarr[:,0] == INSITUarr[j,0], i]
            # else:
            #     NDVI = -9999
            # PARAMrow = np.hstack((PARAMrow, NDVI))
            # TERRAIN, height slope, aspect
            PARAMrow = np.hstack((PARAMrow, TERRAIN[0,i-1], TERRAIN[1, i-1], TERRAIN[2, i-1]))
            # LC
            PARAMrow = np.hstack((PARAMrow, LC[i-1]))
            #PARAMrow = np.hstack((PARAMrow, LC[i-1]))
            #date
            PARAMrow = np.hstack((PARAMrow, INSITUarr[j,0]))
            #station number (name)
            PARAMrow = np.hstack((PARAMrow, i))
            PARAMarr = np.vstack([PARAMarr, PARAMrow])

    flt_cond = np.array([PARAMarr[:,0] != -9999] and [PARAMarr[:,1] != -9999] and [PARAMarr[:,2] != -9999]).squeeze()
    PARAMarrflt = PARAMarr[flt_cond, :]
    PARAMarr = PARAMarrflt

    np.savetxt(outpath + '/combtable.csv', PARAMarr, delimiter=',')
    np.savez(outpath + '/comtable.npz', PARAMarr=PARAMarr)





    print('done')


def train_SVR(data_path, outpath):
    from sklearn.svm import SVR
    from sklearn.grid_search import GridSearchCV
    from sklearn.cross_validation import train_test_split
    from sklearn.tree import DecisionTreeRegressor
    from sklearn import ensemble
    from sklearn.utils import shuffle
    from sklearn.metrics import mean_squared_error
    import sklearn.preprocessing
    import cPickle

    # this routines uses data extracted from extract_params and merge_params
    # to train a support vector machine

    tmp = np.load(data_path)
    training_data = tmp['PARAMarr']
    training_data = training_data[1::,:]

    #filter data for months
    dates = []
    for i in range(training_data.shape[0]):
        dates.append(datetime.date.fromordinal(int(training_data[i,9])))
    valid_months = (np.array([d.month for d in dates]) > 4) & (np.array([d.month for d in dates]) < 11) & (training_data[:,0] != -9999) & (training_data[:,1] < -0) & (training_data[:,8] != -1) & (training_data[:,3] > 2000) & (training_data[:,3] < 6000)
    training_data_flt = training_data[valid_months, :]

    #load training data
    y = training_data_flt[:,0]
    param_columns = np.array([1,2,3,4,5,6,7])
    x = training_data_flt[:, param_columns]
    #normalize data
    scaler = sklearn.preprocessing.StandardScaler().fit(x)
    #x = scaler.transform(x)
    #sort data
    x_sort_ind = np.argsort(x[:,0])
    y = y[x_sort_ind]
    x = x[x_sort_ind,:]

    #split into independent training data and test data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, train_size=0.7)

    # # SVR
    # # -------------------------------------------------
    # #epsilon = logspace(-2,-0.5,len=3); gamma = logspace(-2,1,len=3); cost = logspace(-2,2,len=3)
    # dictCV = dict(C=np.logspace(-2, 2, 10),
    #               gamma=np.logspace(-2, 1, 10),
    #               epsilon=np.logspace(-2, -0.5, 10))
    #
    # #specify kernel
    # svr_rbf = SVR(kernel='rbf')
    #
    # #rigorous grid search
    # start = time()
    # gdCV = GridSearchCV(estimator=svr_rbf, param_grid=dictCV)
    # gdCV.fit(x_train, y_train)
    # print(gdCV.best_score_)
    # print(gdCV.best_params_)
    # print(gdCV.best_estimator_)
    # print(time() - start)
    #
    # #prediction
    # y_CV_rbf = gdCV.predict(x_test)
    #
    n_est = 1000
    #
    # # Ada Boost
    # # ------------------------------------------------
    # params1 = [{'max_depth': np.arange(5)+1}]
    # params2 = [{'learning_rate': np.logspace(-2,0,10)}]
    #
    # dtr = DecisionTreeRegressor()
    # dtr_cv = GridSearchCV(dtr, params1)
    # dtr_cv.fit(x_train, y_train)
    #
    # clf_cv = ensemble.AdaBoostRegressor(dtr_cv.best_estimator_, n_estimators=n_est)
    # #clf_2 = ensemble.AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=n_est)
    # clf_2 = GridSearchCV(clf_cv, params2)
    # start = time()
    # clf_2.fit(x_train, y_train)
    # print(time() - start)
    #
    # #predict
    # clf_2 = clf_2.best_estimator_
    # y_ada = clf_2.predict(x_test)
    # with open(outpath + '/ada_model.pkl', 'wb') as fid:
    #     cPickle.dump(clf_2, fid)

    # Gradient Boost
    # --------------------------------------------
    params = [{'max_depth': np.arange(5)+1,
               'min_samples_split': np.arange(5)+1,
               'learning_rate': np.logspace(-2,0,10)}]

    clf_cv = ensemble.GradientBoostingRegressor(n_estimators=n_est, loss='ls')
    clf_grad = GridSearchCV(clf_cv, params)
    #clf_grad.

    # clf_grad = ensemble.GradientBoostingRegressor(n_estimators=n_est, max_depth=4, min_samples_split=1,
    #                                               learning_rate=0.01, loss='ls')
    clf_grad.fit(x_train, y_train)
    clf_grad = clf_grad.best_estimator_

    #predict
    y_grad = clf_grad.predict(x_test)
    with open(outpath + '/grad_model.pkl', 'wb') as fid:
        cPickle.dump(clf_grad, fid)

    # plt training deviance
    test_score = np.zeros((n_est,), dtype=np.float64)

    for i, y_pred in enumerate(clf_grad.staged_decision_function(x_test)):
        test_score[i] = clf_grad.loss_(y_test, y_pred)

    plt.figure(figsize=(12,6))
    plt.subplot(1, 2, 1)
    plt.title('Deviance')
    plt.plot(np.arange(n_est) + 1, clf_grad.train_score_, 'b-', label='Training Set Deviance')
    plt.plot(np.arange(n_est) + 1, test_score, 'r-', label='Test Set Deviance')
    plt.legend(loc='upper right')
    plt.xlabel('Boosting Iterations')
    plt.ylabel('Deviance')
    #plt.savefig('/mnt/SAT/Workspaces/Grf/Processing/ASAR_ALPS/gradboost_dev.png')
    #plt.close()

    # plot feature importance
    feature_importance = clf_grad.feature_importances_
    feature_importance = 100.0 * (feature_importance/feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, ['SIG0', 'SIG0norm', 'LIA', 'Mean SIG', 'Height', 'Slope', 'Aspect'])
    #plt.yticks(pos, ['SIG0norm', 'Mean SIG', 'Height', 'Slope', 'Aspect'])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()
    plt.savefig(outpath + '/gradboost_fimp.png')
    plt.close()

    #-------------------------------------------

    # print("SVR")
    # print(np.corrcoef(y_test, y_CV_rbf))
    # print(np.sqrt(np.sum(np.square(y_test-y_CV_rbf)) / len(y_test)))

    # print("Ada Boost")
    # print(np.corrcoef(y_test, y_ada))
    # #print(np.sqrt(np.sum(np.square(y_test-y_ada)) / len(y_test)))
    # print(mean_squared_error(y_test, y_ada))

    print("Gradient Boost")
    print(np.corrcoef(y_test, y_grad))
    print(mean_squared_error(y_test, y_grad))

    # plt.figure(figsize=(6,6))
    # plt.scatter(y_test, y_ada, c='g', label='true vs est')
    # plt.xlim(0.2,0.45)
    # plt.ylim(0.2,0.45)
    # plt.savefig(outpath + '/figAda2.png')
    # plt.close()

    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_grad, c="", label='true vs est')
    plt.xlim(0.2,0.45)
    plt.ylim(0.2,0.45)
    plt.plot([0.2, 0.45], [0.2, 0.45], 'k--')
    plt.xlabel('True SMC [m3/m-3]')
    plt.ylabel('Estimated SMC [m3/m-3]')
    plt.savefig(outpath + '/figgrad2.png')
    plt.close()

    # plt.scatter(y_test, y_CV_rbf, c='g', label='true vs est')
    # plt.xlim(0,0.5)
    # plt.ylim(0,0.5)
    # plt.savefig(outpath + '/figSVR2.png')
    # plt.close()
