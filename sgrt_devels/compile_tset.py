__author__ = 'usergre'

import sgrt.common.recursive_filesearch as rsearch
import h5py
import os
import glob
import sklearn.preprocessing
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sgrt.common.grids.Equi7Grid import Equi7Tile
from sgrt.common.grids.Equi7Grid import Equi7Grid
from sgrt.common.utils.SgrtTile import SgrtTile
from osgeo import gdal, gdalconst
from sklearn.svm import SVR
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
from sklearn.cross_validation import train_test_split
from time import time
import math
from scipy.ndimage import median_filter
import datetime as dt

def stackh5(root_path=None, out_path=None):


    #find h5 files to create time series
    filelist = rsearch.search_file(root_path, '*T15*.he5')

    #get file size
    ftmp = h5py.File(filelist[0], 'r')
    ftmp_smc = ftmp['HDFEOS/GRIDS/Analysis_Data/Data Fields/sm_surface_analysis']

    z_dim = len(filelist)
    col_dim = ftmp_smc.shape[1]
    row_dim = ftmp_smc.shape[0]

    #initiate output stack
    SMAPstack = h5py.File(root_path + '/SMAPL4SMC_2015.h5','w')

    smcstack = SMAPstack.create_dataset("SMCstack_2015", (z_dim, row_dim, col_dim), dtype="<f4")
    time = SMAPstack.create_dataset("time", (z_dim,), dtype="<f8")
    latarr = SMAPstack.create_dataset('cell_lat', data=ftmp['HDFEOS/GRIDS/FileMainGroup/Data Fields/cell_lat'])
    lonarr = SMAPstack.create_dataset('cell_lon', data=ftmp['HDFEOS/GRIDS/FileMainGroup/Data Fields/cell_lon'])

    ftmp.close()

    #iterate through all found files
    for find in range(len(filelist)):
        ftmp = h5py.File(filelist[find], 'r')
        smcstack[find,:,:] = ftmp['HDFEOS/GRIDS/Analysis_Data/Data Fields/sm_surface_analysis']
        if len(filelist[find]) == 133:
            y = ftmp.filename[96:100]
            m = ftmp.filename[100:102]
            d = ftmp.filename[102:104]
        else:
            y = ftmp.filename[97:101]
            m = ftmp.filename[101:103]
            d = ftmp.filename[103:105]
        #time[find] = ftmp['time'][0]
        time[find] = dt.date(int(y),int(m),int(d)).toordinal()
        ftmp.close()

    SMAPstack.close()


def dem2equi7(out_path=None, dem_path=None):

    # this routine re-projects the DEM to the Equi7 grid and derives slope and aspect

    grid = Equi7Grid(10)
    grid.resample(dem_path, out_path, gdal_path="/usr/local/bin", sgrid_ids=['EU'], e7_folder=False,
                  outshortname="EDUDEM", withtilenameprefix=True, image_nodata=-32767, tile_nodata=-9999,
                  qlook_flag=False, resampling_type='bilinear')

    # get list of resampled DEM tiles
    filelist = [x for x in glob.glob(out_path + '*.tif')]

    # iterate through all files and derive slope and aspect
    for file in filelist:

        if (file.find('aspect') == -1) and (file.find('slope') == -1):
            # aspect
            aspect_path = out_path + os.path.basename(file)[:-4] + '_aspect.tif'
            if not os.path.exists(aspect_path):
                os.system('/usr/local/bin/gdaldem aspect ' + file + ' ' + aspect_path + ' -co "COMPRESS=LZW"')

            # slope
            slope_path = out_path + os.path.basename(file)[:-4] + '_slope.tif'
            if not os.path.exists(slope_path):
                os.system('/usr/local/bin/gdaldem slope ' + file + ' ' + slope_path + ' -co "COMPRESS=LZW"')


class Trainingset(object):


    def __init__(self, sgrt_root, sig0mpath, smcpath, dempath, outpath, uselc=True, subgrid='EU'):

        self.sgrt_root = sgrt_root
        self.sig0mpath = sig0mpath
        self.smcpath = smcpath
        self.dempath = dempath
        self.outpath = outpath
        self.uselc=uselc
        self.subgrid=subgrid

        #
        # Get processing extent
        #
        # get list of available parameter tiles
        # tiles = os.listdir(sig0mpath)
        tiles = ['E069N084T1']
        print(tiles)
        # tiles = ['E051N015T1', 'E048N014T1', 'E045N014T1', 'E032N014T1', 'E034N008T1', 'E034N007T1', 'E049N016T1']
        # Portugal
        # tiles = ['E034N007T1','E034N008T1']
        # South Tyrol
        # tiles = ['E048N014T1', 'E048N015T1', 'E049N014T1', 'E049N015T1']
        # Spain
        #tiles = ['E032N014T1']


        extents = []
        for tname in tiles:
            tmptile = Equi7Tile(subgrid + '010M_' + tname)
            extents.append(tmptile.extent)
            tmptile = None

        extents = np.array(extents)

        # randomly generate points for the training set
        self.points = self.create_random_points(aoi=extents)

        # extract parameters
        sig0lia = self.extr_sig0_lia(extents)

        #pickle.dump(sig0lia, open(self.outpath + 'sig0lia_dict.p', 'wb'))
        sig0lia = pickle.load(open(self.outpath + 'sig0lia_dict.p', 'rb'))

        #temp filter swath


        # filter samples with nan values
        samples = np.array([sig0lia.get(x) for x in sig0lia.keys()])
        samples = samples.transpose()
        valid = ~np.isnan(samples).any(axis=1)

        # define training and validation sets
        self.target = np.array(sig0lia['ssm'])[valid]
        self.features = np.vstack((#(np.array(sig0lia['sig0vv'])[valid]-np.array(sig0lia['vv_tmean'])[valid])/np.array(sig0lia['vv_tstd'])[valid],
                                   #(np.array(sig0lia['sig0vh'])[valid]-np.array(sig0lia['vh_tmean'])[valid])/np.array(sig0lia['vh_tstd'])[valid],
                                   #np.array(sig0lia['vv_sstd'])[valid],
                                   #np.array(sig0lia['vh_sstd'])[valid],
                                   #np.array(sig0lia['lia'])[valid],
                                   #np.array(sig0lia['lia_sstd'])[valid],
                                   #np.array(sig0lia['vv_tmean'])[valid],
                                   np.array(sig0lia['vv_tstd'])[valid],
                                   #np.array(sig0lia['vh_tmean'])[valid],
                                   np.array(sig0lia['vh_tstd'])[valid],
                                   np.array(sig0lia['sig0vv'])[valid],
                                   np.array(sig0lia['sig0vh'])[valid],
                                   np.array(sig0lia['vv_k1'])[valid],
                                   np.array(sig0lia['vh_k1'])[valid],
                                   #np.array(sig0lia['vv_k2'])[valid],
                                   #np.array(sig0lia['vh_k2'])[valid],
                                   np.array(sig0lia['vv_k3'])[valid],
                                   np.array(sig0lia['vh_k3'])[valid])).transpose()
                                   #np.array(sig0lia['vv_k4'])[valid],
                                   #np.array(sig0lia['vh_k4'])[valid])).transpose()
                                   #np.array(sig0lia['vv_slope'])[valid],
                                   #np.array(sig0lia['vh_slope'])[valid],#)).transpose()
                                   #np.array(sig0lia['height'])[valid],
                                   #np.array(sig0lia['aspect'])[valid],
                                   #np.array(sig0lia['slope'])[valid])).transpose()

        sig01sc = 1 - ((np.array(sig0lia['vv_sstd'])[valid] - np.array(sig0lia['vv_sstd'])[valid].min()) /
                       (np.array(sig0lia['vv_sstd'])[valid].max() - np.array(sig0lia['vv_sstd'])[valid].min()))
        sig02sc = 1 - ((np.array(sig0lia['vh_sstd'])[valid] - np.array(sig0lia['vh_sstd'])[valid].min()) /
                       (np.array(sig0lia['vh_sstd'])[valid].max() - np.array(sig0lia['vh_sstd'])[valid].min()))
        liasc = 1 - ((np.array(sig0lia['lia_sstd'])[valid] - np.array(sig0lia['lia_sstd'])[valid].min()) /
                     (np.array(sig0lia['lia_sstd'])[valid].max() - np.array(sig0lia['lia_sstd'])[valid].min()))

        #self.weights = (sig01sc + sig02sc + liasc) / 3
        self.weights = liasc

        print 'HELLO'


    def train_model(self):

        import scipy.stats
        from sklearn.ensemble import AdaBoostRegressor
        from sklearn.decomposition import PCA

        # filter bad ssm values
        valid = np.where(self.target > 0)
        self.target = self.target[valid[0]]
        self.features = self.features[valid[0],:]
        self.weights = self.weights[valid[0]]
        # filter nan values
        valid = ~np.any(np.isinf(self.features), axis=1)
        self.target = self.target[valid]
        self.features = self.features[valid,:]
        self.weights = self.weights[valid]

        # scaling
        scaler = sklearn.preprocessing.StandardScaler().fit(self.features)
        features = scaler.transform(self.features)
        #pca = PCA()
        #features = pca.transform(features)

        # split into independent training data and test data
        #x_train, x_test, y_train, y_test, w_train, w_test = train_test_split(features, self.target, self.weights, test_size=0.3,
        #                                                    train_size=0.7, random_state=0)

        x_train, x_test, y_train, y_test, w_train, w_test = train_test_split(features, self.target, self.weights,
                                                                             test_size=0.2,
                                                                             train_size=0.8, random_state=42)


        # ...----...----...----...----...----...----...----...----...----...
        # SVR -- SVR -- SVR -- SVR -- SVR -- SVR -- SVR -- SVR -- SVR -- SVR
        # ---....---....---....---....---....---....---....---....---....---

        # dictCV = dict(C =       np.logspace(-2,2,10),
        #               gamma =   np.logspace(-2,-0.5,10),
        #               epsilon = np.logspace(-2, -0.5,10))
        dictCV = dict(C=scipy.stats.expon(scale=100),
                      gamma=scipy.stats.expon(scale=.1),
                      epsilon=scipy.stats.expon(scale=.1),
                      kernel=['rbf'])

        # specify kernel
        # svr_rbf = SVR(kernel = 'rbf')
        svr_rbf = SVR()

        # parameter tuning -> grid search
        start = time()
        print('Model training startet at: ' + str(start))
        # gdCV = GridSearchCV(estimator=svr_rbf, param_grid=dictCV1, n_jobs=2, verbose=1, pre_dispatch='all')
        gdCV = RandomizedSearchCV(estimator=svr_rbf,
                                  param_distributions=dictCV,
                                  n_iter=200,
                                  n_jobs=8,
                                  pre_dispatch='all',
                                  cv=10,
                                  verbose=1)
                                  #fit_params={"sample_weight": w_train})


        gdCV.fit(x_train, y_train)
        y_CV_rbf = gdCV.predict(x_test)

        r = np.corrcoef(y_test, y_CV_rbf)
        error = np.sqrt(np.sum(np.square(y_test - y_CV_rbf)) / len(y_test))
        print(r)
        print(error)
        print('Elapse time for training: ' + str(time() - start))
        pickle.dump((gdCV.best_estimator_, scaler), open(self.outpath + 'mlmodel.p', 'wb'))

        # prediction on training set
        # y_CV_rbf = gdCV.predict(x_train)

        r = np.corrcoef(y_test, y_CV_rbf)
        error = np.sqrt(np.sum(np.square(y_test-y_CV_rbf)) / len(y_test))

        print('SVR performance based on test-set')
        print('R: ' + str(r[0,1]))
        print('RMSE. ' + str(error))

        # create plots
        plt.figure(figsize = (6,6))
        plt.scatter(y_test, y_CV_rbf, c='g', label='True vs Est')
        plt.xlim(0,0.5)
        plt.ylim(0,0.5)
        plt.xlabel("SMAP L4 SMC [m3m-3]")
        plt.ylabel("Estimated SMC [m3m-3]")
        plt.plot([0,0.5],[0,0.5], 'k--')
        plt.savefig(self.outpath + 'truevsest.png')
        plt.close()

        # prediction on trainingset
        y_CV_rbf = gdCV.predict(x_train)

        r = np.corrcoef(y_train, y_CV_rbf)
        error = np.sqrt(np.sum(np.square(y_train - y_CV_rbf)) / len(y_train))

        print('SVR performance based on train-set')
        print('R: ' + str(r[0, 1]))
        print('RMSE. ' + str(error))

        # create plots
        plt.figure(figsize=(6, 6))
        plt.scatter(y_train, y_CV_rbf, c='g', label='True vs Est')
        plt.xlim(0, 0.5)
        plt.ylim(0, 0.5)
        plt.xlabel("SMAP L4 SMC [m3m-3]")
        plt.ylabel("Estimated SMC [m3m-3]")
        plt.plot([0, 0.5], [0, 0.5], 'k--')
        plt.savefig(self.outpath + 'truevsest_training.png')
        plt.close()



        return (gdCV.best_estimator_, scaler)


    def get_terrain(self, x, y):
        # extract elevation, slope and aspect

        # set up grid to determine tile name
        Eq7 = Equi7Grid(10)

        # create ouput array
        topo = np.full((len(self.points), 3), -9999, dtype=np.float32)

        # get tile
        tilename = Eq7.identfy_tile(self.subgrid, (x, y))

        # elevation
        filename = glob.glob(self.dempath + tilename + '*T1.tif')
        elev = gdal.Open(filename[0], gdal.GA_ReadOnly)
        elevBand = elev.GetRasterBand(1)
        elevGeo = elev.GetGeoTransform()
        h = elevBand.ReadAsArray(int((x-elevGeo[0])/10), int((elevGeo[3]-y)/10), 1, 1)
        elev = None

        # aspect
        filename = glob.glob(self.dempath + tilename + '*_aspect.tif')
        asp = gdal.Open(filename[0], gdal.GA_ReadOnly)
        aspBand = asp.GetRasterBand(1)
        aspGeo = asp.GetGeoTransform()
        a = aspBand.ReadAsArray(int((x-aspGeo[0])/10), int((aspGeo[3]-y)/10), 1, 1)
        asp = None

        # slope
        filename = glob.glob(self.dempath + tilename + '*_slope.tif')
        slp = gdal.Open(filename[0], gdal.GA_ReadOnly)
        slpBand = slp.GetRasterBand(1)
        slpGeo = slp.GetGeoTransform()
        s = slpBand.ReadAsArray(int((x-slpGeo[0])/10), int((slpGeo[3]-y)/10), 1, 1)
        slp = None

        return (h[0,0], a[0,0], s[0,0])


    # def extr_sig0_lia(self, aoi, hour=None):
    #
    #     import sgrt_devels.extr_TS as exTS
    #     import random
    #     import datetime as dt
    #     import os
    #     from scipy.stats import moment
    #
    #     cntr = 0
    #     if hour == 5:
    #         path = "168"
    #     elif hour == 17:
    #         path = "117"
    #     else:
    #         path = None
    #
    #     valLClist = [10,11,12,13,18,19,20,21,26,27,28,29,32]
    #
    #     # cycle through all points
    #     for px in self.points:
    #
    #         # create 100 random points to sample the 9 km smap pixel
    #         sig0points = set()
    #         cntr2 = 0
    #         broken = 0
    #         while len(sig0points) <= 100:         # TODO: Increase number of subpoints
    #             if cntr2 >= 5000:
    #                 broken = 1
    #                 break
    #
    #             tmp_alfa = random.random()
    #             tmp_c = random.randint(0,4500)
    #             tmp_dx = math.sin(tmp_alfa) * tmp_c
    #             if tmp_alfa > .5:
    #                 tmp_dx = 0 - tmp_dx
    #             tmp_dy = math.cos(tmp_alfa) * tmp_c
    #
    #             tmpx = int(round(px[0] + tmp_dx))
    #             tmpy = int(round(px[1] + tmp_dy))
    #             #tmpx = random.randint(px[0]-4450,px[0]+4450)
    #             #tmpy = random.randint(px[1]-4450,px[1]+4450)
    #
    #             # check land cover
    #             LCpx = self.get_lc(tmpx, tmpy)
    #
    #             # get mean
    #             mean = self.get_sig0mean(tmpx, tmpy)
    #             # mean = np.float32(mean)
    #             # mean[mean != -9999] = mean[mean != -9999] / 100
    #
    #
    #             #if LCpx in valLClist and mean[0] != -9999 and mean[1] != -9999 and \
    #             #                tmpx >= aoi[px[2]][0]+100 and tmpx <= aoi[px[2]][2]-100 and \
    #             #                tmpy >= aoi[px[2]][1]+100 and tmpy <= aoi[px[2]][3]-100:
    #             if LCpx in valLClist and mean[0] != -9999 and mean[1] != -9999 and \
    #                             tmpx >= aoi[px[2]][0] and tmpx <= aoi[px[2]][2] and \
    #                             tmpy >= aoi[px[2]][1] and tmpy <= aoi[px[2]][3]:
    #                 sig0points.add((tmpx, tmpy))
    #
    #             cntr2 = cntr2 + 1
    #
    #         if broken == 1:
    #             continue
    #
    #         # cycle through the create points to retrieve a aerial mean value
    #         # dictionary to hold the time seres
    #         vvdict = {}
    #         vhdict = {}
    #         liadict = {}
    #         # slopelistVV = []
    #         # slopelistVH = []
    #         meanlistVV = []
    #         meanlistVH = []
    #         sdlistVV = []
    #         sdlistVH = []
    #         kdict = {"k1listVV":  [],
    #                  "k1listVH": [],
    #                  "k2listVV": [],
    #                  "k2listVH": [],
    #                  "k3listVV": [],
    #                  "k3listVH": [],
    #                  "k4listVV": [],
    #                  "k4listVH": []}
    #         hlist = []
    #         slist = []
    #         alist = []
    #
    #         # counter
    #         tsnum = 0
    #         for subpx in sig0points:
    #             # get slope
    #             # slope = self.get_slope(subpx[0],subpx[1])
    #             # slope = np.float32(slope)
    #             # slope[slope != -9999] = slope[slope != -9999] / 100
    #             # slopelistVV.append(slope[0])
    #             # slopelistVH.append(slope[1])
    #             # slopelistVV.append(0)
    #             # slopelistVH.append(0)
    #
    #             # get mean
    #             mean = self.get_sig0mean(subpx[0],subpx[1], path)
    #             mean = np.float32(mean)
    #             mean[mean != -9999] = np.power(10,(mean[mean != -9999] / 100)/10)
    #             meanlistVV.append(mean[0])
    #             meanlistVH.append(mean[1])
    #
    #             # get standard deviation
    #             sd = self.get_sig0sd(subpx[0],subpx[1], path)
    #             sd = np.float32(sd)
    #             sd[sd != -9999] = np.power(10,(sd[sd != -9999] / 100)/10)
    #             sdlistVV.append(sd[0])
    #             sdlistVH.append(sd[1])
    #
    #             # get k statistics
    #             #for kn in range(4):
    #             #vvname = "k" + str(kn+1) + "listVV"
    #             #vhname = "k" + str(kn+1) + "listVH"
    #             k = self.get_kN(subpx[0],subpx[1],1,path)
    #             kdict["k1listVV"].append(k[0]/1000.0)
    #             kdict["k1listVH"].append(k[1]/1000.0)
    #
    #             # get height, aspect, and slope
    #             terr = self.get_terrain(subpx[0],subpx[1])
    #             hlist.append(terr[0])
    #             alist.append(terr[1])
    #             slist.append(terr[2])
    #
    #             # get sig0 and lia timeseries
    #             #tmp_series = exTS.read_NORM_SIG0(self.sgrt_root, 'S1AIWGRDH', 'A0112', 'normalized', 10,
    #             #                                 subpx[0], subpx[1], 1, 1, pol_name=['VV','VH'], grid='Equi7')
    #             tmp_series = exTS.extr_SIG0_LIA_ts(self.sgrt_root, 'S1AIWGRDH', 'A0111', 'resampled', 10,
    #                                                subpx[0], subpx[1], 1, 1, pol_name=['VV','VH'], grid='Equi7', sat_pass='A')
    #             # lia_series = exTS.extr_SIG0_LIA_ts(self.sgrt_root, 'S1AIWGRDH', 'A0111', 'resampled', 10, subpx[0], subpx[1], 1, 1, pol_name=['VV','VH'], grid='Equi7')
    #             sig0vv = np.float32(tmp_series[1]['sig0'])
    #             sig0vh = np.float32(tmp_series[1]['sig02'])
    #             lia = np.float32(tmp_series[1]['lia'])
    #             sig0vv[sig0vv != -9999] = sig0vv[sig0vv != -9999]/100
    #             sig0vh[sig0vh != -9999] = sig0vh[sig0vh != -9999]/100
    #             lia[lia != -9999] = lia[lia != -9999]/100
    #
    #             # normalise backscatter
    #             # if slope[0] != 0 and slope[0] != -9999:
    #             #    sig0vv[sig0vv != -9999] = np.power(10,(sig0vv[sig0vv != -9999] - slope[0] * (lia[sig0vv != -9999] - 30))/10)
    #             # else:
    #             sig0vv[sig0vv != -9999] = np.power(10,sig0vv[sig0vv != -9999]/10)
    #             # if slope[1] != 0 and slope[1] != -9999:
    #             #     sig0vh[sig0vh != -9999] = np.power(10,(sig0vh[sig0vh != -9999] - slope[0] * (lia[sig0vh != -9999] - 30))/10)
    #             # else:
    #             sig0vh[sig0vh != -9999] = np.power(10,sig0vh[sig0vh != -9999]/10)
    #
    #             datelist = tmp_series[0]
    #             # datelist = [dt.datetime.fromordinal(x) for x in tmp_series[0]]
    #
    #             # create temporary dataframe
    #             tmp_df_vv = pd.DataFrame(sig0vv.squeeze(), index=datelist)
    #             tmp_df_vh = pd.DataFrame(sig0vh.squeeze(), index=datelist)
    #             tmp_df_lia = pd.DataFrame(lia.squeeze(), index=datelist)
    #
    #             # add to collection
    #             tmp_dict = {str(tsnum): tmp_df_vv[0]}
    #             vvdict.update(tmp_dict)
    #             tmp_dict = {str(tsnum): tmp_df_vh[0]}
    #             vhdict.update(tmp_dict)
    #             tmp_dict = {str(tsnum): tmp_df_lia[0]}
    #             liadict.update(tmp_dict)
    #
    #             tsnum = tsnum + 1
    #
    #             tmp_df_vv = None
    #             tmp_df_vh = None
    #             tmp_df_lia = None
    #             tmp_dict = None
    #
    #         # merge into panda data frame
    #         df_vv = pd.DataFrame(vvdict)
    #         df_vh = pd.DataFrame(vhdict)
    #         df_lia = pd.DataFrame(liadict)
    #
    #         # create mask
    #         arr_vv = np.array(df_vv)
    #         arr_vh = np.array(df_vh)
    #         arr_lia = np.array(df_lia)
    #
    #         mask = (arr_vv == -9999) | (arr_vv < -15.00) | (arr_vh == -9999) | (arr_vh < -15.00) | \
    #                (arr_lia < 10.00) | (arr_lia > 50.00)
    #
    #         df_vv.iloc[mask] = np.nan
    #         df_vh.iloc[mask] = np.nan
    #         df_lia.iloc[mask] = np.nan
    #
    #         # mask months
    #         monthmask = (df_vv.index.map(lambda x: x.month) > 1) & (df_vv.index.map(lambda x: x.month) < 12)
    #
    #         # calculate spatial mean and standard deviation
    #         df_vv_mean = 10*np.log10(df_vv.mean(1)[monthmask])
    #         df_vh_mean = 10*np.log10(df_vh.mean(1)[monthmask])
    #         df_lia_mean = df_lia.mean(1)[monthmask]
    #         df_vv_sstd = 10*np.log10(df_vv.std(1)[monthmask])
    #         df_vh_sstd = 10*np.log10(df_vh.std(1)[monthmask])
    #         df_lia_sstd = df_lia.std(1)[monthmask]
    #
    #         # merge to make sure all fits together
    #         tmp_dict = {'vv': df_vv_mean, 'vh': df_vh_mean, 'lia': df_lia_mean,
    #                     'vv_std': df_vv_sstd, 'vh_std': df_vh_sstd, 'lia_std': df_lia_sstd}
    #         df_bac = pd.DataFrame(tmp_dict)
    #
    #         # Cleanup
    #         tmp_dict = None
    #         df_vv_mean = None
    #         df_vh_mean = None
    #         df_lia_mean = None
    #         df_vv_sstd = None
    #         df_vh_sstd = None
    #         df_lia_sstd = None
    #
    #         # ------------------------------------------
    #         # get ssm
    #         tmp_ssm = self.get_ssm(px[0], px[1])
    #         ssm_series = pd.Series(index=df_bac.index)
    #         ssm_dates = np.array(tmp_ssm[0])
    #
    #         for i in range(len(df_bac.index)):
    #             current_day = df_bac.index[i].date()
    #             id = np.where(ssm_dates == current_day)
    #             if len(id[0]) > 0:
    #                 ssm_series.iloc[i] = tmp_ssm[1][id]
    #
    #         tmp_ssm = None
    #
    #         # convert lists to numpy arrays
    #         meanlistVV = np.array(meanlistVV)
    #         meanlistVH = np.array(meanlistVH)
    #         sdlistVV = np.array(sdlistVV)
    #         sdlistVH = np.array(sdlistVH)
    #         #klistVV = np.array(kdict['k1listVV'])
    #         #klistVH = np.array(kdict['k1listVH'])
    #         klistVV = np.log(meanlistVV)
    #         klistVH = np.log(meanlistVH)
    #
    #         # calculate mean temporal mean and standard deviation and slope
    #         meanMeanVV = 10*np.log10(np.mean(meanlistVV[meanlistVV != -9999]))
    #         meanMeanVH = 10*np.log10(np.mean(meanlistVH[meanlistVH != -9999]))
    #         meanSdVV = 10*np.log10(np.mean(sdlistVV[sdlistVV != -9999]))
    #         meanSdVH = 10*np.log10(np.mean(sdlistVH[sdlistVH != -9999]))
    #         # meanSlopeVV = np.mean(slopelistVV[slopelistVV != -9999])
    #         # meanSlopeVH = np.mean(slopelistVH[slopelistVH != -9999])
    #         # calculate mean of temporal k statistics (upscaling)
    #         meank1VV = np.mean(klistVV[klistVV != -9999])
    #         meank1VH = np.mean(klistVH[klistVH != -9999])
    #         meank2VV = moment(klistVV[klistVV != -9999], moment=2)
    #         meank2VH = moment(klistVH[klistVH != -9999], moment=2)
    #         meank3VV = moment(klistVV[klistVV != -9999], moment=3)
    #         meank3VH = moment(klistVH[klistVH != -9999], moment=3)
    #         meank4VV = moment(klistVV[klistVV != -9999], moment=4)
    #         meank4VH = moment(klistVH[klistVH != -9999], moment=4)
    #         # calculate mean terrain parameters
    #         meanH = np.mean(hlist[hlist != -9999])
    #         meanA = np.mean(alist[alist != -9999])
    #         meanS = np.mean(slist[slist != -9999])
    #
    #         if cntr == 0:
    #             ll = len(list(np.array(df_bac['vv']).squeeze()))
    #             sig0lia_samples = {'ssm': list(np.array(ssm_series).squeeze()),
    #                                'sig0vv': list(np.array(df_bac['vv']).squeeze()),
    #                                'sig0vh': list(np.array(df_bac['vh']).squeeze()),
    #                                'lia': list(np.array(df_bac['lia']).squeeze()),
    #                                'vv_sstd': list(np.array(df_bac['vv_std']).squeeze()),
    #                                'vh_sstd': list(np.array(df_bac['vh_std']).squeeze()),
    #                                'lia_sstd': list(np.array(df_bac['lia_std']).squeeze()),
    #                                'vv_tmean': [meanMeanVV]*ll,
    #                                'vh_tmean': [meanMeanVH]*ll,
    #                                'vv_tstd': [meanSdVV]*ll,
    #                                'vh_tstd': [meanSdVH]*ll,
    #                                # 'vv_slope': [meanSlopeVV]*ll,
    #                                # 'vh_slope': [meanSlopeVH]*ll,
    #                                'vv_k1': [meank1VV]*ll,
    #                                'vh_k1': [meank1VH]*ll,
    #                                'vv_k2': [meank2VV] * ll,
    #                                'vh_k2': [meank2VH] * ll,
    #                                'vv_k3': [meank3VV] * ll,
    #                                'vh_k3': [meank3VH] * ll,
    #                                'vv_k4': [meank4VV] * ll,
    #                                'vh_k4': [meank4VH] * ll,
    #                                'height': [meanH]*ll,
    #                                'aspect': [meanA]*ll,
    #                                'slope': [meanS]*ll}
    #         else:
    #             ll = len(list(np.array(df_bac['vv']).squeeze()))
    #             sig0lia_samples['ssm'].extend(list(np.array(ssm_series).squeeze()))
    #             sig0lia_samples['sig0vv'].extend(list(np.array(df_bac['vv']).squeeze()))
    #             sig0lia_samples['sig0vh'].extend(list(np.array(df_bac['vh']).squeeze()))
    #             sig0lia_samples['lia'].extend(list(np.array(df_bac['lia']).squeeze()))
    #             sig0lia_samples['vv_sstd'].extend(list(np.array(df_bac['vv_std']).squeeze()))
    #             sig0lia_samples['vh_sstd'].extend(list(np.array(df_bac['vh_std']).squeeze()))
    #             sig0lia_samples['lia_sstd'].extend(list(np.array(df_bac['lia_std']).squeeze()))
    #             sig0lia_samples['vv_tmean'].extend([meanMeanVV]*ll)
    #             sig0lia_samples['vh_tmean'].extend([meanMeanVH]*ll)
    #             sig0lia_samples['vv_tstd'].extend([meanSdVV]*ll)
    #             sig0lia_samples['vh_tstd'].extend([meanSdVH]*ll)
    #             # sig0lia_samples['vv_slope'].extend([meanSlopeVV]*ll)
    #             # sig0lia_samples['vh_slope'].extend([meanSlopeVH]*ll)
    #             sig0lia_samples['vv_k1'].extend([meank1VV]*ll)
    #             sig0lia_samples['vh_k1'].extend([meank1VH]*ll)
    #             sig0lia_samples['vv_k2'].extend([meank2VV] * ll)
    #             sig0lia_samples['vh_k2'].extend([meank2VH] * ll)
    #             sig0lia_samples['vv_k3'].extend([meank3VV] * ll)
    #             sig0lia_samples['vh_k3'].extend([meank3VH] * ll)
    #             sig0lia_samples['vv_k4'].extend([meank4VV] * ll)
    #             sig0lia_samples['vh_k4'].extend([meank4VH] * ll)
    #             sig0lia_samples['height'].extend([meanH]*ll)
    #             sig0lia_samples['aspect'].extend([meanA]*ll)
    #             sig0lia_samples['slope'].extend([meanS]*ll)
    #
    #         cntr = cntr + 1
    #         os.system('rm /tmp/*.vrt')
    #
    #     return sig0lia_samples


    def extr_sig0_lia(self, aoi, hour=None):

        import sgrt_devels.extr_TS as exTS
        import random
        import datetime as dt
        import os
        from scipy.stats import moment

        cntr = 0
        if hour == 5:
            path = "168"
        elif hour == 17:
            path = "117"
        else:
            path = None

        valLClist = [10, 11, 12, 13, 18, 19, 20, 21, 26, 27, 28, 29, 32]

        # cycle through all points
        for px in self.points:

            # create 100 random points to sample the 9 km smap pixel
            sig0points = set()
            cntr2 = 0
            broken = 0
            while len(sig0points) <= 100:  # TODO: Increase number of subpoints
                if cntr2 >= 5000:
                    broken = 1
                    break

                tmp_alfa = random.random()
                tmp_c = random.randint(0, 4500)
                tmp_dx = math.sin(tmp_alfa) * tmp_c
                if tmp_alfa > .5:
                    tmp_dx = 0 - tmp_dx
                tmp_dy = math.cos(tmp_alfa) * tmp_c

                tmpx = int(round(px[0] + tmp_dx))
                tmpy = int(round(px[1] + tmp_dy))

                # check land cover
                if self.uselc == True:
                    try:
                        LCpx = self.get_lc(tmpx, tmpy)
                    except:
                        LCpx = -1
                else:
                    LCpx = 10

                # get mean
                mean = self.get_sig0mean(tmpx, tmpy)

                if LCpx in valLClist and mean[0] != -9999 and mean[1] != -9999 and \
                                tmpx >= aoi[px[2]][0] and tmpx <= aoi[px[2]][2] and \
                                tmpy >= aoi[px[2]][1] and tmpy <= aoi[px[2]][3]:
                    sig0points.add((tmpx, tmpy))

                cntr2 = cntr2 + 1

            if broken == 1:
                continue

            # cycle through the create points to retrieve a aerial mean value
            # dictionary to hold the time seres
            vvdict = {}
            vhdict = {}
            liadict = {}
            # slopelistVV = []
            # slopelistVH = []
            meanlistVV = []
            meanlistVH = []
            sdlistVV = []
            sdlistVH = []
            kdict = {"k1listVV": [],
                     "k1listVH": [],
                     "k2listVV": [],
                     "k2listVH": [],
                     "k3listVV": [],
                     "k3listVH": [],
                     "k4listVV": [],
                     "k4listVH": []}
            hlist = []
            slist = []
            alist = []

            # counter
            tsnum = 0
            for subpx in sig0points:

                # get height, aspect, and slope
                if self.subgrid == 'EU':
                    terr = self.get_terrain(subpx[0], subpx[1])
                else:
                    terr = [0,0,0]
                hlist.append(terr[0])
                alist.append(terr[1])
                slist.append(terr[2])

                # get sig0 and lia timeseries
                tmp_series = exTS.extr_SIG0_LIA_ts(self.sgrt_root, 'S1AIWGRDH', 'A0111', 'resampled', 10,
                                                   subpx[0], subpx[1], 1, 1, pol_name=['VV', 'VH'], grid='Equi7', subgrid=self.subgrid,
                                                   sat_pass='A')

                sig0vv = np.float32(tmp_series[1]['sig0'])
                sig0vh = np.float32(tmp_series[1]['sig02'])
                lia = np.float32(tmp_series[1]['lia'])
                sig0vv[sig0vv != -9999] = sig0vv[sig0vv != -9999] / 100
                sig0vh[sig0vh != -9999] = sig0vh[sig0vh != -9999] / 100
                lia[lia != -9999] = lia[lia != -9999] / 100

                # linearise backscatter
                sig0vv[sig0vv != -9999] = np.power(10, sig0vv[sig0vv != -9999] / 10)
                sig0vh[sig0vh != -9999] = np.power(10, sig0vh[sig0vh != -9999] / 10)

                datelist = tmp_series[0]

                # create temporary dataframe
                tmp_df_vv = pd.DataFrame(sig0vv.squeeze(), index=datelist)
                tmp_df_vh = pd.DataFrame(sig0vh.squeeze(), index=datelist)
                tmp_df_lia = pd.DataFrame(lia.squeeze(), index=datelist)

                # add to collection
                tmp_dict = {str(tsnum): tmp_df_vv[0]}
                vvdict.update(tmp_dict)
                tmp_dict = {str(tsnum): tmp_df_vh[0]}
                vhdict.update(tmp_dict)
                tmp_dict = {str(tsnum): tmp_df_lia[0]}
                liadict.update(tmp_dict)

                tsnum = tsnum + 1

                tmp_df_vv = None
                tmp_df_vh = None
                tmp_df_lia = None
                tmp_dict = None

            # merge into panda data frame
            df_vv = pd.DataFrame(vvdict)
            df_vh = pd.DataFrame(vhdict)
            df_lia = pd.DataFrame(liadict)

            # create mask
            arr_vv = np.array(df_vv)
            arr_vh = np.array(df_vh)
            arr_lia = np.array(df_lia)

            mask = (arr_vv == -9999) | (arr_vv < -25.00) | (arr_vh == -9999) | (arr_vh < -25.00) | \
                   (arr_lia < 10.00) | (arr_lia > 50.00)

            df_vv.iloc[mask] = np.nan
            df_vh.iloc[mask] = np.nan
            df_lia.iloc[mask] = np.nan

            # mask months
            # monthmask = (df_vv.index.map(lambda x: x.month) > 4) & (df_vv.index.map(lambda x: x.month) < 10)
            monthmask = (df_vv.index.map(lambda x: x.month) >= 1) & (df_vv.index.map(lambda x: x.month) <= 12)

            # calculate spatial mean and standard deviation
            df_vv_mean = 10 * np.log10(df_vv.mean(1)[monthmask])
            df_vh_mean = 10 * np.log10(df_vh.mean(1)[monthmask])
            df_lia_mean = df_lia.mean(1)[monthmask]
            df_vv_sstd = 10 * np.log10(df_vv.std(1)[monthmask])
            df_vh_sstd = 10 * np.log10(df_vh.std(1)[monthmask])
            df_lia_sstd = df_lia.std(1)[monthmask]

            # merge to make sure all fits together
            tmp_dict = {'vv': df_vv_mean, 'vh': df_vh_mean, 'lia': df_lia_mean,
                        'vv_std': df_vv_sstd, 'vh_std': df_vh_sstd, 'lia_std': df_lia_sstd}
            df_bac = pd.DataFrame(tmp_dict)

            # Cleanup
            tmp_dict = None
            df_vv_mean = None
            df_vh_mean = None
            df_lia_mean = None
            df_vv_sstd = None
            df_vh_sstd = None
            df_lia_sstd = None

            # ------------------------------------------
            # get ssm
            tmp_ssm = self.get_ssm(px[0], px[1])
            ssm_series = pd.Series(index=df_bac.index)
            ssm_dates = np.array(tmp_ssm[0])

            for i in range(len(df_bac.index)):
                current_day = df_bac.index[i].date()
                id = np.where(ssm_dates == current_day)
                if len(id[0]) > 0:
                    ssm_series.iloc[i] = tmp_ssm[1][id]

            tmp_ssm = None

            # calculate paramters
            # convert lists to numpy arrays

            # calculate mean temporal mean and standard deviation and slope
            df_vv_mm = df_vv[(df_vv.index.map(lambda x: x.month)) > 4 & (df_vv.index.map(lambda x: x.month) < 10)]
            df_vh_mm = df_vh[(df_vh.index.map(lambda x: x.month)) > 4 & (df_vh.index.map(lambda x: x.month) < 10)]
            arr_vv = np.nanmean(np.array(df_vv_mm), axis=1)
            arr_vh = np.nanmean(np.array(df_vh_mm), axis=1)
            meanMeanVV = 10*np.log10(np.nanmean(arr_vv))
            meanMeanVH = 10*np.log10(np.nanmean(arr_vh))
            meanSdVV = 10*np.log10(np.nanstd(arr_vv))
            meanSdVH = 10*np.log10(np.nanstd(arr_vh))

            # calculate mean of temporal k statistics (upscaling)
            meank1VV = np.nanmean(10*np.log10(arr_vv))
            meank1VH = np.nanmean(10*np.log10(arr_vh))
            meank2VV = moment(10*np.log10(arr_vv.ravel()), moment=2, nan_policy='omit')
            meank2VH = moment(10*np.log10(arr_vh.ravel()), moment=2, nan_policy='omit')
            meank3VV = moment(10*np.log10(arr_vv.ravel()), moment=3, nan_policy='omit')
            meank3VH = moment(10*np.log10(arr_vh.ravel()), moment=3, nan_policy='omit')
            meank4VV = moment(10*np.log10(arr_vv.ravel()), moment=4, nan_policy='omit')
            meank4VH = moment(10*np.log10(arr_vh.ravel()), moment=4, nan_policy='omit')
            # calculate mean terrain parameters
            meanH = np.mean(hlist[hlist != -9999])
            meanA = np.mean(alist[alist != -9999])
            meanS = np.mean(slist[slist != -9999])

            if cntr == 0:
                ll = len(list(np.array(df_bac['vv']).squeeze()))
                sig0lia_samples = {'ssm': list(np.array(ssm_series).squeeze()),
                                   'sig0vv': list(np.array(df_bac['vv']).squeeze()),
                                   'sig0vh': list(np.array(df_bac['vh']).squeeze()),
                                   'lia': list(np.array(df_bac['lia']).squeeze()),
                                   'vv_sstd': list(np.array(df_bac['vv_std']).squeeze()),
                                   'vh_sstd': list(np.array(df_bac['vh_std']).squeeze()),
                                   'lia_sstd': list(np.array(df_bac['lia_std']).squeeze()),
                                   'vv_tmean': [meanMeanVV] * ll,
                                   'vh_tmean': [meanMeanVH] * ll,
                                   'vv_tstd': [meanSdVV] * ll,
                                   'vh_tstd': [meanSdVH] * ll,
                                   # 'vv_slope': [meanSlopeVV]*ll,
                                   # 'vh_slope': [meanSlopeVH]*ll,
                                   'vv_k1': [meank1VV] * ll,
                                   'vh_k1': [meank1VH] * ll,
                                   'vv_k2': [meank2VV] * ll,
                                   'vh_k2': [meank2VH] * ll,
                                   'vv_k3': [meank3VV] * ll,
                                   'vh_k3': [meank3VH] * ll,
                                   'vv_k4': [meank4VV] * ll,
                                   'vh_k4': [meank4VH] * ll,
                                   'height': [meanH] * ll,
                                   'aspect': [meanA] * ll,
                                   'slope': [meanS] * ll}
            else:
                ll = len(list(np.array(df_bac['vv']).squeeze()))
                sig0lia_samples['ssm'].extend(list(np.array(ssm_series).squeeze()))
                sig0lia_samples['sig0vv'].extend(list(np.array(df_bac['vv']).squeeze()))
                sig0lia_samples['sig0vh'].extend(list(np.array(df_bac['vh']).squeeze()))
                sig0lia_samples['lia'].extend(list(np.array(df_bac['lia']).squeeze()))
                sig0lia_samples['vv_sstd'].extend(list(np.array(df_bac['vv_std']).squeeze()))
                sig0lia_samples['vh_sstd'].extend(list(np.array(df_bac['vh_std']).squeeze()))
                sig0lia_samples['lia_sstd'].extend(list(np.array(df_bac['lia_std']).squeeze()))
                sig0lia_samples['vv_tmean'].extend([meanMeanVV] * ll)
                sig0lia_samples['vh_tmean'].extend([meanMeanVH] * ll)
                sig0lia_samples['vv_tstd'].extend([meanSdVV] * ll)
                sig0lia_samples['vh_tstd'].extend([meanSdVH] * ll)
                # sig0lia_samples['vv_slope'].extend([meanSlopeVV]*ll)
                # sig0lia_samples['vh_slope'].extend([meanSlopeVH]*ll)
                sig0lia_samples['vv_k1'].extend([meank1VV] * ll)
                sig0lia_samples['vh_k1'].extend([meank1VH] * ll)
                sig0lia_samples['vv_k2'].extend([meank2VV] * ll)
                sig0lia_samples['vh_k2'].extend([meank2VH] * ll)
                sig0lia_samples['vv_k3'].extend([meank3VV] * ll)
                sig0lia_samples['vh_k3'].extend([meank3VH] * ll)
                sig0lia_samples['vv_k4'].extend([meank4VV] * ll)
                sig0lia_samples['vh_k4'].extend([meank4VH] * ll)
                sig0lia_samples['height'].extend([meanH] * ll)
                sig0lia_samples['aspect'].extend([meanA] * ll)
                sig0lia_samples['slope'].extend([meanS] * ll)

            cntr = cntr + 1
            os.system('rm /tmp/*.vrt')

        return sig0lia_samples


    def create_random_points(self, aoi=None):

        # create random points within the area of interest, selected points are aligned with the
        # SMAP 9km EASE grid

        import random

        # get lat/lon of aoi
        Eq7SAR = Equi7Grid(10)
        aoi_latlon = list()
        for subaoi in aoi:
            ll = Eq7SAR.equi7xy2lonlat(self.subgrid, subaoi[0]+5000, subaoi[1]+5000)
            ur = Eq7SAR.equi7xy2lonlat(self.subgrid, subaoi[2]-5000, subaoi[3]-5000)
            # subtract a buffer to make shure the selected ease grid point are within the aoi
            aoi_latlon.append([ll[0],ll[1],ur[0],ur[1]])

        # load EASE grid definition
        EASE_lats = np.fromfile('/mnt/SAT/Workspaces/GrF/01_Data/EASE20/EASE2_M09km.lats.3856x1624x1.double', dtype=np.float64)
        EASE_lons = np.fromfile('/mnt/SAT/workspaces/GrF/01_Data/EASE20/EASE2_M09km.lons.3856x1624x1.double', dtype=np.float64)
        EASE_lats = EASE_lats.reshape(3856,1624)
        EASE_lons = EASE_lons.reshape(3856,1624)

        # find valid ease locations
        EASEpoint = list()
        for irow in range(3856):
            for icol in range(1624):

                subaoi_counter = 0
                for subaoi in aoi_latlon:
                    if (EASE_lons[irow,icol] > subaoi[0]) & \
                            (EASE_lons[irow,icol] < subaoi[2]) & \
                            (EASE_lats[irow,icol] > subaoi[1]) & \
                            (EASE_lats[irow,icol] < subaoi[3]):
                        EASEpoint.append((irow,icol, subaoi_counter))
                    subaoi_counter = subaoi_counter + 1

        # create list of 100 random points
        points = set()
        usedEASE = list()

        #while len(points) <= 50:
        for randInd in range(len(EASEpoint)):
            # tmpx = random.randint(aoi[0]+4500, aoi[2]-4500)
            # tmpy = random.randint(aoi[1]+4500, aoi[3]-4500)
            # draw a point from the list of EASE points
            #randInd = random.randint(0, len(EASEpoint)-1)
            # store the used points to avoid double usage
            if randInd not in usedEASE:
                usedEASE.append(randInd)
                tmplon = EASE_lons[EASEpoint[randInd][0], EASEpoint[randInd][1]]
                tmplat = EASE_lats[EASEpoint[randInd][0], EASEpoint[randInd][1]]
                tmpxy = Eq7SAR.lonlat2equi7xy(tmplon,tmplat)

                # get land cover
                if self.uselc == True:
                    try:
                        LCpx = self.get_lc(tmpxy[1]-2250, tmpxy[2]-2250, dx=60, dy=60)
                        ValidLCind = np.where((LCpx == 10) | (LCpx == 12) |
                                              (LCpx == 13) | (LCpx == 18) |
                                              (LCpx == 26) | (LCpx == 29) |
                                              (LCpx == 32) | (LCpx == 11) |
                                             #(LCpx == 16) | (LCpx == 17) |
                                              (LCpx == 19) | (LCpx == 20) |
                                              (LCpx == 21) | (LCpx == 27) |
                                              (LCpx == 28))
                        # check if at least 10 percent are usable pixels
                        ValidPrec = len(ValidLCind[0]) / (60.0*60.0)
                    except:
                        ValidPrec = 0
                else:
                    ValidPrec = 1


                if ValidPrec >= 0.1:
                    points.add((int(round(tmpxy[1])), int(round(tmpxy[2])), EASEpoint[randInd][2]))

                # check if open land
                # if LCpx in [10, 12, 13, 18, 26, 29, 32]:
                #     points.add((tmpxy[0],tmpxy[1]))
        print(len(points))
        return(points)


    def get_lc(self, x, y, dx=1, dy=1):

        # set up land cover grid
        Eq7LC = Equi7Grid(75)

        # get tile name of 75 Equi7 grid to check land-cover
        tilename = Eq7LC.identfy_tile(self.subgrid, (x, y))
        LCtile = SgrtTile(dir_root=self.sgrt_root,
                          product_id='S1AIWGRDH',
                          soft_id='E0110',
                          product_name='CORINE06',
                          ftile=tilename,
                          src_res=75)

        LCfilename = [xs for xs in LCtile._tile_files]
        LCfilename = LCtile.dir + '/' + LCfilename[0] + '.tif'
        LC = gdal.Open(LCfilename, gdal.GA_ReadOnly)
        LCband = LC.GetRasterBand(1)
        LCpx = LCband.ReadAsArray(int((x-LCtile.geotags['geotransform'][0])/75), int((LCtile.geotags['geotransform'][3]-y)/75), dx, dy)
        if dx==1 and dy==1:
            return LCpx[0][0]
        else:
            return LCpx


    def get_slope(self, x, y):

        # set up parameter grid
        Eq7Par = Equi7Grid(10)

        # get tile name of Equi7 10 grid
        tilename = Eq7Par.identfy_tile(self.subgrid, (x,y))
        Stile = SgrtTile(dir_root=self.sgrt_root,
                         product_id='S1AIWGRDH',
                         soft_id='B0212',
                         product_name='sig0m',
                         ftile=tilename,
                         src_res=10)

        SfilenameVV = [xs for xs in Stile._tile_files if 'SIGSL' in xs and 'VV' in xs and '_qlook' not in xs]
        SfilenameVH = [xs for xs in Stile._tile_files if 'SIGSL' in xs and 'VH' in xs and '_qlook' not in xs]
        SfilenameVV = Stile.dir + '/' + SfilenameVV[0] + '.tif'
        SfilenameVH = Stile.dir + '/' + SfilenameVH[0] + '.tif'

        SVV = gdal.Open(SfilenameVV, gdal.GA_ReadOnly)
        SVH = gdal.Open(SfilenameVH, gdal.GA_ReadOnly)
        SVVband = SVV.GetRasterBand(1)
        SVHband = SVH.GetRasterBand(1)
        slopeVV = SVVband.ReadAsArray(int((x-Stile.geotags['geotransform'][0])/10), int((Stile.geotags['geotransform'][3]-y)/10), 1, 1)[0][0]
        slopeVH = SVHband.ReadAsArray(int((x-Stile.geotags['geotransform'][0])/10), int((Stile.geotags['geotransform'][3]-y)/10), 1, 1)[0][0]

        return (slopeVV, slopeVH)


    def get_sig0mean(self, x, y, path=None):

        import math

        # set up parameter grid
        Eq7Par = Equi7Grid(10)

        # get tile name of Equi7 10 grid
        tilename = Eq7Par.identfy_tile(self.subgrid, (x,y))
        Stile = SgrtTile(dir_root=self.sgrt_root,
                         product_id='S1AIWGRDH',
                         soft_id='B0212',
                         product_name='sig0m',
                         ftile=tilename,
                         src_res=10)
        if path != None:
            SfilenameVV = [xs for xs in Stile._tile_files if 'SIG0M' in xs and 'VV' in xs and path in xs and '_qlook' not in xs]
            SfilenameVH = [xs for xs in Stile._tile_files if 'SIG0M' in xs and 'VH' in xs and path in xs and'_qlook' not in xs]
        else:
            SfilenameVV = [xs for xs in Stile._tile_files if 'SIG0M' in xs and 'VV' in xs and '_qlook' not in xs]
            SfilenameVH = [xs for xs in Stile._tile_files if 'SIG0M' in xs and 'VH' in xs and '_qlook' not in xs]
        if len(SfilenameVH) == 0 | len(SfilenameVV) == 0:
            return (-9999, -9999)
        SfilenameVV = Stile.dir + '/' + SfilenameVV[0] + '.tif'
        SfilenameVH = Stile.dir + '/' + SfilenameVH[0] + '.tif'

        SVV = gdal.Open(SfilenameVV, gdal.GA_ReadOnly)
        SVH = gdal.Open(SfilenameVH, gdal.GA_ReadOnly)
        SVVband = SVV.GetRasterBand(1)
        SVHband = SVH.GetRasterBand(1)

        img_x = int(math.floor((x-Stile.geotags['geotransform'][0])/10))
        img_y = int(math.floor((Stile.geotags['geotransform'][3]-y)/10))

        if img_x == 10000 or img_y == 10000:
            meanVH = -9999
            meanVV = -9999
        else:
            meanVV = SVVband.ReadAsArray(img_x, img_y, 1, 1)[0][0]
            meanVH = SVHband.ReadAsArray(img_x, img_y, 1, 1)[0][0]

        return (meanVV, meanVH)


    def get_sig0sd(self, x, y, path):

        # set up parameter grid
        Eq7Par = Equi7Grid(10)

        # get tile name of Equi7 10 grid
        tilename = Eq7Par.identfy_tile(self.subgrid, (x,y))
        Stile = SgrtTile(dir_root=self.sgrt_root,
                         product_id='S1AIWGRDH',
                         soft_id='B0212',
                         product_name='sig0m',
                         ftile=tilename,
                         src_res=10)

        if path != None:
            SfilenameVV = [xs for xs in Stile._tile_files if 'SIGSD' in xs and 'VV' in xs and path in xs and '_qlook' not in xs]
            SfilenameVH = [xs for xs in Stile._tile_files if 'SIGSD' in xs and 'VH' in xs and path in xs and '_qlook' not in xs]
        else:
            SfilenameVV = [xs for xs in Stile._tile_files if 'SIGSD' in xs and 'VV' in xs and '_qlook' not in xs]
            SfilenameVH = [xs for xs in Stile._tile_files if 'SIGSD' in xs and 'VH' in xs and '_qlook' not in xs]

        if len(SfilenameVH) == 0 | len(SfilenameVV) == 0:
            return (-9999, -9999)

        SfilenameVV = Stile.dir + '/' + SfilenameVV[0] + '.tif'
        SfilenameVH = Stile.dir + '/' + SfilenameVH[0] + '.tif'

        SVV = gdal.Open(SfilenameVV, gdal.GA_ReadOnly)
        SVH = gdal.Open(SfilenameVH, gdal.GA_ReadOnly)
        SVVband = SVV.GetRasterBand(1)
        SVHband = SVH.GetRasterBand(1)
        sdVV = SVVband.ReadAsArray(int((x-Stile.geotags['geotransform'][0])/10), int((Stile.geotags['geotransform'][3]-y)/10), 1, 1)[0][0]
        sdVH = SVHband.ReadAsArray(int((x-Stile.geotags['geotransform'][0])/10), int((Stile.geotags['geotransform'][3]-y)/10), 1, 1)[0][0]

        return (sdVV, sdVH)


    def get_kN(self, x, y, n, path=None):

        import math

        # set up parameter grid
        Eq7Par = Equi7Grid(10)

        # get tile name of Equi7 10 grid
        tilename = Eq7Par.identfy_tile(self.subgrid, (x, y))
        Stile = SgrtTile(dir_root=self.sgrt_root,
                         product_id='S1AIWGRDH',
                         soft_id='B0212',
                         product_name='sig0m',
                         ftile=tilename,
                         src_res=10)

        knr = 'K' + str(n)
        if path != None:
            SfilenameVV = [xs for xs in Stile._tile_files if
                           knr in xs and 'VV' in xs and path in xs and '_qlook' not in xs]
            SfilenameVH = [xs for xs in Stile._tile_files if
                           knr in xs and 'VH' in xs and path in xs and '_qlook' not in xs]
        else:
            SfilenameVV = [xs for xs in Stile._tile_files if knr in xs and 'VV' in xs and '_qlook' not in xs]
            SfilenameVH = [xs for xs in Stile._tile_files if knr in xs and 'VH' in xs and '_qlook' not in xs]
        if len(SfilenameVH) == 0 | len(SfilenameVV) == 0:
            return (-9999, -9999)
        SfilenameVV = Stile.dir + '/' + SfilenameVV[0] + '.tif'
        SfilenameVH = Stile.dir + '/' + SfilenameVH[0] + '.tif'

        SVV = gdal.Open(SfilenameVV, gdal.GA_ReadOnly)
        SVH = gdal.Open(SfilenameVH, gdal.GA_ReadOnly)
        SVVband = SVV.GetRasterBand(1)
        SVHband = SVH.GetRasterBand(1)

        img_x = int(math.floor((x - Stile.geotags['geotransform'][0]) / 10))
        img_y = int(math.floor((Stile.geotags['geotransform'][3] - y) / 10))

        if img_x == 10000 or img_y == 10000:
            kVH = -9999
            kVV = -9999
        else:
            kVV = SVVband.ReadAsArray(img_x, img_y, 1, 1)[0][0]
            kVH = SVHband.ReadAsArray(img_x, img_y, 1, 1)[0][0]

        return (kVV, kVH)


    def get_ssm(self, x, y):
        import math
        import datetime as dt

        grid = Equi7Grid(10)
        poi_lonlat = grid.equi7xy2lonlat(self.subgrid, x, y)

        #load file stack
        h5file = self.sgrt_root + '/Sentinel-1_CSAR/IWGRDH/ancillary/datasets/SMAPL4/SMAPL4SMC_2015.h5'
        ssm_stack = h5py.File(h5file, 'r')

        # find the 4 nearest gridpoints
        lat = ssm_stack['cell_lat']
        lon = ssm_stack['cell_lon']
        mindist = [10,10,10,10]
        mindist_lon = [0.0,0.0,0.0,0.0]
        mindist_lat = [0.0,0.0,0.0,0.0]

        for iy in range(lat.shape[0]):
            for ix in range(lat.shape[1]):

                dist = math.sqrt(math.pow(poi_lonlat[0]-lon[iy,ix],2) + math.pow(poi_lonlat[1]-lat[iy,ix],2))
                if dist < mindist[0]:
                    mindist[3] = mindist[2]
                    mindist[2] = mindist[1]
                    mindist[1] = mindist[0]
                    mindist[0] = dist

                    mindist_lon[3] = mindist_lon[2]
                    mindist_lon[2] = mindist_lon[1]
                    mindist_lon[1] = mindist_lon[0]
                    mindist_lon[0] = ix

                    mindist_lat[3] = mindist_lat[2]
                    mindist_lat[2] = mindist_lat[1]
                    mindist_lat[1] = mindist_lat[0]
                    mindist_lat[0] = iy

        # stack time series of the 4 nearest grid-points
        ssm = np.array([np.array(ssm_stack['SMCstack_2015'][:,mindist_lat[0], mindist_lon[0]]),
                       np.array(ssm_stack['SMCstack_2015'][:,mindist_lat[1], mindist_lon[1]]),
                       np.array(ssm_stack['SMCstack_2015'][:,mindist_lat[2], mindist_lon[2]]),
                       np.array(ssm_stack['SMCstack_2015'][:,mindist_lat[3], mindist_lon[3]])])

        # calculate the weighted average, using the distance as weights
        ssm = np.average(ssm, axis=0, weights=mindist)

        time = np.array(ssm_stack['time'])
        #timelist = [dt.date.fromtimestamp(x+946727936) for x in time]
        timelist = [dt.date.fromordinal(math.trunc(x)) for x in time]
        #dt.datetime.fromtimestamp()
        ssm_stack.close()

        return (timelist, ssm)


class Estimationset(object):


    def __init__(self, sgrt_root, tile, sig0mpath, dempath, outpath, mlmodel, subgrid="EU", uselc=True):

        self.sgrt_root = sgrt_root
        self.outpath = outpath
        self.sig0mpath = sig0mpath
        self.dempath = dempath
        self.subgrid = subgrid
        self.uselc = uselc

        # get list of available parameter tiles
        # tiles = os.listdir(sig0mpath)
        # self.tiles = ['E048N014T1']
        self.tiles = tile
        self.mlmodel = mlmodel


    def ssm_ts(self, x, y, fdim):

        from extr_TS import read_NORM_SIG0
        from extr_TS import extr_SIG0_LIA_ts
        from scipy.stats import moment

        # extract parameters
        siglia_ts = extr_SIG0_LIA_ts(self.sgrt_root, 'S1AIWGRDH', 'A0111', 'resampled', 10, x, y, fdim, fdim,
                                     pol_name=['VV', 'VH'], grid='Equi7', sat_pass='A', monthmask=[4,5,6,7,8,9,10])

        terr_arr = self.get_terrain(self.tiles[0])
        param_arr = self.get_params(self.tiles[0])
        bac_arr = self.get_sig0_lia(self.tiles[0], siglia_ts[0][0].strftime("D%Y%m%d_%H%M%S"))
        lc_arr = self.create_LC_mask(self.tiles[0], bac_arr)

        aoi_pxdim = [int((x-terr_arr['h'][1][0])/10),
                     int((terr_arr['h'][1][3]-y)/10),
                     int((x-terr_arr['h'][1][0])/10)+fdim,
                     int((terr_arr['h'][1][3]-y)/10)+fdim]
        a = terr_arr['a'][0][aoi_pxdim[1]:aoi_pxdim[3], aoi_pxdim[0]:aoi_pxdim[2]]
        s = terr_arr['s'][0][aoi_pxdim[1]:aoi_pxdim[3], aoi_pxdim[0]:aoi_pxdim[2]]
        h = terr_arr['h'][0][aoi_pxdim[1]:aoi_pxdim[3], aoi_pxdim[0]:aoi_pxdim[2]]
        sig0mVV = param_arr['sig0mVV'][0][aoi_pxdim[1]:aoi_pxdim[3], aoi_pxdim[0]:aoi_pxdim[2]]
        sig0mVH = param_arr['sig0mVH'][0][aoi_pxdim[1]:aoi_pxdim[3], aoi_pxdim[0]:aoi_pxdim[2]]
        sig0sdVV = param_arr['sig0sdVV'][0][aoi_pxdim[1]:aoi_pxdim[3], aoi_pxdim[0]:aoi_pxdim[2]]
        sig0sdVH = param_arr['sig0sdVH'][0][aoi_pxdim[1]:aoi_pxdim[3], aoi_pxdim[0]:aoi_pxdim[2]]
        #slpVV = param_arr['slpVV'][0][aoi_pxdim[1]:aoi_pxdim[3], aoi_pxdim[0]:aoi_pxdim[2]]
        #slpVH = param_arr['slpVH'][0][aoi_pxdim[1]:aoi_pxdim[3], aoi_pxdim[0]:aoi_pxdim[2]]
        # k1VV = param_arr['k1VV'][0][aoi_pxdim[1]:aoi_pxdim[3], aoi_pxdim[0]:aoi_pxdim[2]]
        # k1VH = param_arr['k1VH'][0][aoi_pxdim[1]:aoi_pxdim[3], aoi_pxdim[0]:aoi_pxdim[2]]
        # k2VV = param_arr['k2VV'][0][aoi_pxdim[1]:aoi_pxdim[3], aoi_pxdim[0]:aoi_pxdim[2]]
        # k2VH = param_arr['k2VH'][0][aoi_pxdim[1]:aoi_pxdim[3], aoi_pxdim[0]:aoi_pxdim[2]]
        # k3VV = param_arr['k3VV'][0][aoi_pxdim[1]:aoi_pxdim[3], aoi_pxdim[0]:aoi_pxdim[2]]
        # k3VH = param_arr['k3VH'][0][aoi_pxdim[1]:aoi_pxdim[3], aoi_pxdim[0]:aoi_pxdim[2]]
        # k4VV = param_arr['k4VV'][0][aoi_pxdim[1]:aoi_pxdim[3], aoi_pxdim[0]:aoi_pxdim[2]]
        # k4VH = param_arr['k4VH'][0][aoi_pxdim[1]:aoi_pxdim[3], aoi_pxdim[0]:aoi_pxdim[2]]

        # calculate k1,...,kN
        k1VV = np.full((fdim, fdim), -9999, dtype=np.float32)
        k2VV = np.full((fdim, fdim), -9999, dtype=np.float32)
        k3VV = np.full((fdim, fdim), -9999, dtype=np.float32)
        k4VV = np.full((fdim, fdim), -9999, dtype=np.float32)
        k1VH = np.full((fdim, fdim), -9999, dtype=np.float32)
        k2VH = np.full((fdim, fdim), -9999, dtype=np.float32)
        k3VH = np.full((fdim, fdim), -9999, dtype=np.float32)
        k4VH = np.full((fdim, fdim), -9999, dtype=np.float32)

        for ix in range(fdim):
            for iy in range(fdim):

                temp_ts1 = siglia_ts[1]['sig0'][:,iy,ix] / 100.
                temp_ts2 = siglia_ts[1]['sig02'][:,iy, ix] / 100.
                temp_mask1 = np.where(temp_ts1 != -99.99)
                temp_mask2 = np.where(temp_ts2 != -99.99)

                k1VV[iy, ix] = np.mean(temp_ts1[temp_mask1])
                k1VH[iy, ix] = np.mean(temp_ts2[temp_mask2])
                k2VV[iy, ix] = moment(temp_ts1[temp_mask1], moment=2, nan_policy='omit')
                k2VH[iy, ix] = moment(temp_ts2[temp_mask2], moment=2, nan_policy='omit')
                k3VV[iy, ix] = moment(temp_ts1[temp_mask1], moment=3, nan_policy='omit')
                k3VH[iy, ix] = moment(temp_ts2[temp_mask2], moment=3, nan_policy='omit')
                k4VV[iy, ix] = moment(temp_ts1[temp_mask1], moment=4, nan_policy='omit')
                k4VH[iy, ix] = moment(temp_ts2[temp_mask2], moment=4, nan_policy='omit')



        lc_mask = lc_arr[aoi_pxdim[1]:aoi_pxdim[3], aoi_pxdim[0]:aoi_pxdim[2]]
        terr_mask = (a != -9999) & \
                    (s != -9999) & \
                    (h != -9999)

        param_mask = (sig0mVH != -9999) & \
                     (sig0mVV != -9999) & \
                     (sig0sdVH != -9999) & \
                     (sig0sdVV != -9999)

        ssm_ts_out = (siglia_ts[0], np.full((len(siglia_ts[0])), -9999, dtype=np.float32))

        # average sig0 time-series based on selected pixel footprint
        for i in range(len(siglia_ts[0])):
            sig0_mask = (siglia_ts[1]['sig0'][i,:,:] != -9999) & \
                        (siglia_ts[1]['sig0'][i,:,:] >= -2000) & \
                        (siglia_ts[1]['sig02'][i,:,:] != -9999) & \
                        (siglia_ts[1]['sig02'][i,:,:] >= -2000) & \
                        (siglia_ts[1]['lia'][i,:,:] >= 1000) & \
                        (siglia_ts[1]['lia'][i,:,:] <= 5000)

            mask = lc_mask & terr_mask & sig0_mask

            tmp_smc_arr = np.full((fdim,fdim), -9999, dtype=np.float32)

            #estimate smc for each pixel in the 10x10 footprint
            # create a list of aeach feature
            sig0_l = list()
            sig02_l = list()
            #sigvvssd_l = list()
            #sigvhssd_l = list()
            lia_l = list()
            #liassd_l = list()
            sig0mvv_l = list()
            sig0sdvv_l = list()
            sig0mvh_l = list()
            sig0sdvh_l = list()
            #slpvv_l = list()
            #slpvh_l = list()
            h_l = list()
            a_l = list()
            s_l = list()
            k1vv_l = list()
            k1vh_l = list()
            k2vv_l = list()
            k2vh_l = list()
            k3vv_l = list()
            k3vh_l = list()
            k4vv_l = list()
            k4vh_l = list()


            for ix in range(fdim):
                for iy in range(fdim):

                    if mask[iy,ix] == True:

                        if param_mask[iy,ix] == True:
                            # fvect = [np.float32(siglia_ts[1]['sig0'][i,iy,ix])/100.,
                            #          np.float32(siglia_ts[1]['sig02'][i, iy, ix])/100.,
                            #          np.float32(siglia_ts[1]['lia'][i, iy, ix])/100.,
                            #          sig0mVV[iy, ix] / 100.,
                            #          sig0sdVV[iy,ix] / 100.,
                            #          sig0mVH[iy,ix] / 100.,
                            #          sig0sdVH[iy,ix] / 100.,
                            #          slpVV[iy,ix],
                            #          slpVH[iy,ix], #]
                            #          h[iy,ix],
                            #          a[iy,ix],
                            #          s[iy,ix]]
                            # fvect = self.mlmodel[1].transform(fvect)
                            # tmp_smc_arr[iy,ix] = self.mlmodel[0].predict(fvect)
                            sig0_l.append(np.float32(siglia_ts[1]['sig0'][i,iy,ix])/100.)
                            sig02_l.append(np.float32(siglia_ts[1]['sig02'][i,iy,ix])/100.)
                            lia_l.append(np.float32(siglia_ts[1]['lia'][i, iy, ix])/100.)
                            sig0mvv_l.append(sig0mVV[iy, ix] / 100.)
                            sig0sdvv_l.append(sig0sdVV[iy,ix] / 100.)
                            sig0mvh_l.append(sig0mVH[iy, ix] / 100.)
                            sig0sdvh_l.append(sig0sdVH[iy,ix] / 100.)
                            #slpvv_l.append(slpVV[iy,ix])
                            #slpvh_l.append(slpVH[iy,ix])
                            h_l.append(h[iy,ix])
                            a_l.append(a[iy,ix])
                            s_l.append(s[iy,ix])
                            k1vv_l.append(k1VV[iy,ix])
                            k1vh_l.append(k1VH[iy,ix])
                            k2vv_l.append(k2VV[iy, ix])
                            k2vh_l.append(k2VH[iy, ix])
                            k3vv_l.append(k3VV[iy, ix])
                            k3vh_l.append(k3VH[iy, ix])
                            k4vv_l.append(k4VV[iy, ix])
                            k4vh_l.append(k4VH[iy, ix])

            if len(sig0_l) > 0:
                # calculate average of features
                fvect = [#np.mean(np.array(lia_l)),
                         #10 * np.log10(np.mean(np.power(10, (np.array(sig0mvv_l) / 10.)))),
                         10 * np.log10(np.mean(np.power(10, (np.array(sig0sdvv_l) / 10.)))),
                         #10 * np.log10(np.mean(np.power(10, (np.array(sig0mvh_l) / 10.)))),
                         10 * np.log10(np.mean(np.power(10, (np.array(sig0sdvh_l) / 10.)))),
                         10 * np.log10(np.mean(np.power(10, (np.array(sig0_l) / 10.)))),
                         10 * np.log10(np.mean(np.power(10, (np.array(sig02_l) / 10.)))),
                         #10 * np.log10(np.std(np.power(10, (np.array(sig0_l) / 10.)))),
                         #10 * np.log10(np.std(np.power(10, (np.array(sig02_l) / 10.)))),
                         np.mean(k1vv_l),
                         np.mean(k1vh_l),
                         np.mean(k3vv_l),
                         np.mean(k3vh_l)]
                         #np.mean(np.array(lia_l)),
                         #np.std(np.array(lia_l)),
                         #10 * np.log10(np.mean(np.power(10, (np.array(sig0mvv_l) / 10.)))),
                         #10 * np.log10(np.mean(np.power(10, (np.array(sig0sdvv_l) / 10.)))),
                         #10 * np.log10(np.mean(np.power(10, (np.array(sig0mvh_l) / 10.)))),
                         #10 * np.log10(np.mean(np.power(10, (np.array(sig0sdvh_l) / 10.)))),
                         #np.mean(np.array(slpvv_l)),
                         #np.mean(np.array(slpvh_l))]#,
                         #np.mean(np.array(h_l)),
                         #np.mean(np.array(a_l)),
                         #np.mean(np.array(s_l))]

                ## val_ssm = tmp_smc_arr[tmp_smc_arr != -9999]
                # if len(val_ssm) > 0: ssm_ts_out[1][i] = np.mean(val_ssm)

                fvect = self.mlmodel[1].transform(fvect)
                ssm_ts_out[1][i] = self.mlmodel[0].predict(fvect)




        valid = np.where(ssm_ts_out[1] != -9999)
        xx = ssm_ts_out[0][valid]
        yy = ssm_ts_out[1][valid]

        plt.figure(figsize=(18, 6))
        plt.plot(xx,yy)
        plt.show()
        plt.savefig(self.outpath + 'ts' + str(x) + '_' + str(y) + '.png')
        plt.close()
        csvout = np.array([m.strftime("%B %d, %Y") for m in ssm_ts_out[0][valid]], dtype=np.str)
        csvout2 = np.array(ssm_ts_out[1][valid], dtype=np.str)
        # np.savetxt(self.outpath + 'ts.csv', np.hstack((csvout, csvout2)), fmt="%-10c", delimiter=",")
        with open(self.outpath + 'ts' + str(x) + '_' + str(y) + '.csv', 'w') as text_file:
            [text_file.write(csvout[i] + ', ' + csvout2[i] + '\n') for i in range(len(csvout))]

        print(ssm_ts_out[0][valid])

        print("Done")


    def ssm_map(self, date=None, path=None):

        from joblib import Parallel, delayed, load, dump
        import sys
        import tempfile
        import shutil

        for tname in self.tiles:

            for date in self.get_filenames(tname):

                # check if file already exists
                if os.path.exists(self.outpath + tname+'_SSM_' + date + '.tif'):
                    continue

                print("Retrieving soil moisture for " + tname + " / " + date)

                # get sig0 image to derive ssm
                bacArrs = self.get_sig0_lia(tname, date)
                terrainArrs = self.get_terrain(tname)
                paramArr = self.get_params(tname, path)

                # create masks
                if self.uselc == True:
                    lc_mask = self.create_LC_mask(tname, bacArrs)
                sig0_mask = (bacArrs['sig0vv'][0] != -9999) & \
                            (bacArrs['sig0vv'][0] >= -2000) & \
                            (bacArrs['sig0vh'][0] != -9999) & \
                            (bacArrs['sig0vh'][0] >= -2000) & \
                            (bacArrs['lia'][0] >= 1000) & \
                            (bacArrs['lia'][0] <= 5000)
                terr_mask = (terrainArrs['h'][0] != -9999) & \
                            (terrainArrs['a'][0] != -9999) & \
                            (terrainArrs['s'][0] != -9999)
                param_mask = (paramArr['sig0mVH'][0] != -9999) & \
                             (paramArr['sig0mVV'][0] != -9999) & \
                             (paramArr['sig0sdVH'][0] != -9999) & \
                             (paramArr['sig0sdVV'][0] != -9999)
                # combined mask
                if self.uselc == True:
                    mask = lc_mask & sig0_mask & terr_mask & param_mask
                else:
                    mask = sig0_mask & terr_mask & param_mask

                # resample
                bacArrs, paramArr, terrainArrs = self.resample(bacArrs,terrainArrs,paramArr, mask, 10)

                sig0_mask = (bacArrs['sig0vv'][0] != -9999) & \
                            (bacArrs['sig0vv'][0] >= -2000) & \
                            (bacArrs['sig0vh'][0] != -9999) & \
                            (bacArrs['sig0vh'][0] >= -2000) & \
                            (bacArrs['lia'][0] >= 1000) & \
                            (bacArrs['lia'][0] <= 5000)
                terr_mask = (terrainArrs['h'][0] != -9999) & \
                            (terrainArrs['a'][0] != -9999) & \
                            (terrainArrs['s'][0] != -9999)
                param_mask = (paramArr['sig0mVH'][0] != -9999) & \
                            (paramArr['sig0mVV'][0] != -9999) & \
                            (paramArr['sig0sdVH'][0] != -9999) & \
                            (paramArr['sig0sdVV'][0] != -9999)
                # combined mask
                if self.uselc == True:
                    mask = lc_mask & sig0_mask & terr_mask & param_mask
                else:
                    mask = sig0_mask & terr_mask & param_mask

                valid_ind = np.where(mask == True)
                valid_ind = np.ravel_multi_index(valid_ind, (10000,10000))

                #vv_sstd = _local_std(bacArrs['sig0vv'][0], -9999, valid_ind)
                #vh_sstd = _local_std(bacArrs['sig0vh'][0], -9999, valid_ind)
                #lia_sstd = _local_std(bacArrs['lia'][0], -9999, valid_ind, "lia")

                #bacStats = {"vv": vv_sstd, "vh": vh_sstd, 'lia': lia_sstd}
                bacStats = {'vv': bacArrs['sig0vv'][0], 'vh': bacArrs['sig0vh'][0], 'lia': bacArrs['lia'][0]}

                ssm_out = np.full((10000,10000), -9999, dtype=np.float32)

                # parallel prediction
                if not hasattr(sys.stdin, 'close'):
                    def dummy_close():
                        pass
                    sys.stdin.close = dummy_close

                ind_splits = np.array_split(valid_ind, 8)

                # prepare multi processing
                # dump arrays to temporary folder
                temp_folder = tempfile.mkdtemp()
                filename_in = os.path.join(temp_folder, 'joblib_dump1.mmap')
                if os.path.exists(filename_in): os.unlink(filename_in)
                _ = dump((bacArrs, terrainArrs, paramArr, bacStats), filename_in)
                large_memmap = load(filename_in, mmap_mode='r+')
                # output
                filename_out = os.path.join(temp_folder, 'joblib_dump2.mmap')
                ssm_out = np.memmap(filename_out, dtype=np.float32, mode='w+', shape=(10000,10000))
                ssm_out[:] = -9999

                # predict SSM
                Parallel(n_jobs=8, verbose=5, max_nbytes=None)(delayed(_estimate_ssm)(large_memmap[0],large_memmap[1],large_memmap[2], large_memmap[3],ssm_out,i,self.mlmodel) for i in ind_splits)

                # write ssm out
                self.write_ssm(tname, date, bacArrs, ssm_out)

                try:
                    shutil.rmtree(temp_folder)
                except:
                    print("Failed to delete: " + temp_folder)

                print 'HELLO'


    def get_sig0_lia(self, tname, date):

        # read sig0 vv/vh and lia in arrays
        tile = SgrtTile(dir_root=self.sgrt_root,
                        product_id='S1AIWGRDH',
                        soft_id='A0111',
                        product_name='resampled',
                        ftile=self.subgrid + '010M_'+tname,
                        src_res=10)
        tile_lia = SgrtTile(dir_root=self.sgrt_root,
                            product_id='S1AIWGRDH',
                            soft_id='A0111',
                            product_name='resampled',
                            ftile=self.subgrid + '010M_'+tname,
                            src_res=10)

        sig0vv = tile.read_tile(pattern=date+'.*VV.*')
        sig0vh = tile.read_tile(pattern=date+'.*VH.*')
        lia = tile_lia.read_tile(pattern=date+'.*_LIA.*')

        return {'sig0vv': sig0vv, 'sig0vh': sig0vh, 'lia': lia}


    def get_terrain(self, tname):

        if self.subgrid == 'AF':
            h = np.full([10000,10000], 0)
            elevGeo = 0
            a = np.full([10000, 10000], 0)
            aspGeo = 0
            s = np.full([10000, 10000], 0)
            slpGeo = 0
        else:

            # elevation
            filename = glob.glob(self.dempath + '*' + tname + '.tif')
            elev = gdal.Open(filename[0], gdal.GA_ReadOnly)
            elevBand = elev.GetRasterBand(1)
            elevGeo = elev.GetGeoTransform()
            h = elevBand.ReadAsArray()
            elev = None

            # aspect
            filename = glob.glob(self.dempath + '*' +  tname + '_aspect.tif')
            asp = gdal.Open(filename[0], gdal.GA_ReadOnly)
            aspBand = asp.GetRasterBand(1)
            aspGeo = asp.GetGeoTransform()
            a = aspBand.ReadAsArray()
            asp = None

            # slope
            filename = glob.glob(self.dempath + '*' +  tname + '_slope.tif')
            slp = gdal.Open(filename[0], gdal.GA_ReadOnly)
            slpBand = slp.GetRasterBand(1)
            slpGeo = slp.GetGeoTransform()
            s = slpBand.ReadAsArray()
            slp = None

        return {'h': (h, elevGeo), 'a': (a, aspGeo), 's': (s, slpGeo)}


    def get_params(self, tname, path=None):

        # get slope and sig0 mean and standard deviation
        Stile = SgrtTile(dir_root=self.sgrt_root,
                         product_id='S1AIWGRDH',
                         soft_id='B0212',
                         product_name='sig0m',
                         ftile=self.subgrid + '010M_' + tname,
                         src_res=10)

        if path == None:
            path = ''

        #slpVV = Stile.read_tile(pattern='.*SIGSL.*VV'+path+'.*T1$')
        #slpVH = Stile.read_tile(pattern='.*SIGSL.*VH'+path+'.*T1$')
        sig0mVV = Stile.read_tile(pattern='.*SIG0M.*VV'+path+'.*T1$')
        sig0mVH = Stile.read_tile(pattern='.*SIG0M.*VH'+path+'.*T1$')
        sig0sdVV = Stile.read_tile(pattern='.*SIGSD.*VV'+path+'.*T1$')
        sig0sdVH = Stile.read_tile(pattern='.*SIGSD.*VH'+path+'.*T1$')
        k1VV = Stile.read_tile(pattern='.*K1.*VV'+path+'.*T1$')
        k1VH = Stile.read_tile(pattern='.*K1.*VH' + path + '.*T1$')
        k2VV = Stile.read_tile(pattern='.*K2.*VV' + path + '.*T1$')
        k2VH = Stile.read_tile(pattern='.*K2.*VH' + path + '.*T1$')
        k3VV = Stile.read_tile(pattern='.*K3.*VV' + path + '.*T1$')
        k3VH = Stile.read_tile(pattern='.*K3.*VH' + path + '.*T1$')
        k4VV = Stile.read_tile(pattern='.*K4.*VV' + path + '.*T1$')
        k4VH = Stile.read_tile(pattern='.*K4.*VH' + path + '.*T1$')

        return {#'slpVV': slpVV,
                #'slpVH': slpVH,
                'sig0mVV': sig0mVV,
                'sig0mVH': sig0mVH,
                'sig0sdVV': sig0sdVV,
                'sig0sdVH': sig0sdVH,
                'k1VV': k1VV,
                'k1VH': k1VH,
                'k2VV': k2VV,
                'k2VH': k2VH,
                'k3VV': k3VV,
                'k3VH': k3VH,
                'k4VV': k4VV,
                'k4VH': k4VH}


    def create_LC_mask(self, tname, bacArrs):

        # get tile name in 75m lc grid
        eq7tile = Equi7Tile(self.subgrid + '010M_' + tname)
        tname75 = eq7tile.find_family_tiles(res=75)
        # load lc array, resampled to 10m
        lcArr = self.get_lc(tname75[0], bacArrs)

        #generate mask
        tmp = np.array(lcArr[0])
        mask = (tmp == 10) | (tmp == 12) | (tmp == 13) | (tmp == 18) | (tmp == 26) | (tmp == 29) | (tmp == 32) | \
               (tmp == 11) | (tmp == 19) | (tmp == 20) | (tmp == 21) | (tmp == 27) | (tmp == 28)

        return mask


    def get_lc(self, tname, bacArrs):

        # get tile name of 75 Equi7 grid to check land-cover
        LCtile = SgrtTile(dir_root=self.sgrt_root,
                          product_id='S1AIWGRDH',
                          soft_id='E0110',
                          product_name='CORINE06',
                          ftile=self.subgrid + '075M_'+tname,
                          src_res=75)

        LCfilename = [xs for xs in LCtile._tile_files]
        LCfilename = LCtile.dir + '/' + LCfilename[0] + '.tif'
        LC = gdal.Open(LCfilename, gdal.GA_ReadOnly)
        #LCband = LC.GetRasterBand(1)
        LCgeo = LC.GetGeoTransform()
        LCproj = LC.GetProjection()
        #LC = LCband.ReadAsArray()

        # resample to 10m grid
        dst_proj = bacArrs['sig0vv'][1]['spatialreference']
        dst_geotrans = bacArrs['sig0vv'][1]['geotransform']
        dst_width = 10000
        dst_height = 10000
        # define output
        LCres_filename = self.outpath + 'tmp_LCres.tif'
        LCres = gdal.GetDriverByName('GTiff').Create(LCres_filename, dst_width, dst_height, 1, gdalconst.GDT_Int16)
        LCres.SetGeoTransform(dst_geotrans)
        LCres.SetProjection(dst_proj)
        # resample
        gdal.ReprojectImage(LC, LCres, LCproj, dst_proj, gdalconst.GRA_NearestNeighbour)

        del LC, LCres

        LC = gdal.Open(LCres_filename, gdal.GA_ReadOnly)
        LCband = LC.GetRasterBand(1)
        LCgeo = LC.GetGeoTransform()
        LC = LCband.ReadAsArray()

        return (LC, LCgeo)


    def write_ssm(self, tname, date, bacArrs, outarr):

        # write ssm map
        dst_proj = bacArrs['sig0vv'][1]['spatialreference']
        dst_geotrans = bacArrs['sig0vv'][1]['geotransform']
        dst_width = 10000
        dst_height = 10000

        # set up output file
        ssm_path = self.outpath + tname+'_SSM_' + date + '.tif'
        ssm_map = gdal.GetDriverByName('GTiff').Create(ssm_path, dst_width, dst_height, 1, gdalconst.GDT_Int16)
        ssm_map.SetGeoTransform(dst_geotrans)
        ssm_map.SetProjection(dst_proj)

        # transform array to byte
        valid = np.where(outarr != -9999)
        novalid = np.where(outarr == -9999)
        outarr[valid] = np.around(outarr[valid] * 100)
        outarr[novalid] = -9999
        outarr = outarr.astype(dtype=np.int16)


        # write data
        ssm_outband = ssm_map.GetRasterBand(1)
        ssm_outband.WriteArray(outarr)
        ssm_outband.FlushCache()
        ssm_outband.SetNoDataValue(-9999)

        del ssm_map


    def moving_average(self, a, n=3):
        return median_filter(a, size=n, mode='constant', cval=-9999)


    def resample(self, bacArrs, terrainArrs, paramArrs, mask, resolution):

        # bacArrs
        for key in bacArrs.keys():
            tmparr = np.array(bacArrs[key][0], dtype=np.float32)
            tmparr[mask == False] = np.nan
            tmparr = self.moving_average(tmparr, n=10)
            bacArrs[key][0][:,:] = np.array(tmparr, dtype=np.int16)

        # paramArrs
        for key in paramArrs.keys():
            tmparr = np.array(paramArrs[key][0], dtype=np.float32)
            tmparr[mask == False] = np.nan
            tmparr = self.moving_average(tmparr, n=10)
            paramArrs[key][0][:,:] = tmparr

        # terrainArrs
        for key in terrainArrs.keys():
            tmparr = np.array(terrainArrs[key][0], dtype=np.float32)
            tmparr[mask == False] = np.nan
            tmparr = self.moving_average(tmparr, n=10)
            terrainArrs[key][0][:,:] = tmparr

        return (bacArrs, paramArrs, terrainArrs)


    def get_filenames(self, tname):

        tile = SgrtTile(dir_root=self.sgrt_root,
                        product_id='S1AIWGRDH',
                        soft_id='A0111',
                        product_name='resampled',
                        ftile=self.subgrid + '010M_' + tname,
                        src_res=10)

        filelist = tile._tile_files.keys()
        datelist = [x[0:16] for x in filelist]

        return(datelist)



def _estimate_ssm(bacArrs, terrainArrs, paramArr, bacStats, ssm_out, valid_ind, mlmodel):

    for i in valid_ind:
        ind = np.unravel_index(i, (10000,10000))
        try:
            # compile feature vector
            sig0vv = bacArrs['sig0vv'][0][ind] / 100.0
            sig0vh = bacArrs['sig0vh'][0][ind] / 100.0
            lia = bacArrs['lia'][0][ind] / 100.0
            h = terrainArrs['h'][0][ind]
            a = terrainArrs['a'][0][ind]
            s = terrainArrs['s'][0][ind]
            #slpvv = paramArr['slpVV'][0][ind]
            #slpvh = paramArr['slpVH'][0][ind]
            sig0mvv = paramArr['sig0mVV'][0][ind] / 100.0
            sig0mvh = paramArr['sig0mVH'][0][ind] / 100.0
            sig0sdvv = paramArr['sig0sdVV'][0][ind] / 100.0
            sig0sdvh = paramArr['sig0sdVH'][0][ind] / 100.0
            vvsstd = bacStats['vv'][ind]
            vhsstd = bacStats['vh'][ind]
            liasstd = bacStats['lia'][ind]
            k1VV = paramArr['k1VV'][0][ind]/100.
            k1VH = paramArr['k1VH'][0][ind]/100.
            k2VV = paramArr['k2VV'][0][ind]/100.
            k2VH = paramArr['k2VH'][0][ind]/100.
            k3VV = paramArr['k3VV'][0][ind]/100.
            k3VH = paramArr['k3VH'][0][ind]/100.
            k4VV = paramArr['k4VV'][0][ind]/100.
            k4VH = paramArr['k4VH'][0][ind]/100.


            # normalize sig0
            # if slpvv != 0: sig0vv = sig0vv - slpvv*(lia-30)
            # if slpvh != 0: sig0vh = sig0vh - slpvh*(lia-30)

            fvect = [#(sig0vv-sig0mvv)/sig0sdvv,
                     #(sig0vh-sig0sdvh)/sig0sdvh,
                     #sig0mvv,
                     sig0sdvv,
                     #sig0mvh,
                     sig0sdvh,
                     sig0vv,
                     sig0vh,
                     k1VV,
                     k1VH,
                     k3VV,
                     k3VH]
                     #vvsstd,
                     #vhsstd,
                     #lia,
                     #liasstd,
                     #sig0mvv,
                     #sig0sdvv,
                     #sig0mvh,
                     #sig0sdvh,
                     #slpvv,
                     #slpvh,
                     #h,
                     #a,
                     #s]

            fvect = mlmodel[1].transform(fvect)
            # predict ssm
            predssm = mlmodel[0].predict(fvect)
            ssm_out[ind] = predssm
        except:
            ssm_out[ind] = -9999


def _local_std(arr, nanval, valid_ind, parameter="sig0"):
    #calculate local variance of image

    from scipy import ndimage
    from joblib import Parallel, delayed, load, dump
    import sys
    import tempfile
    import shutil


    # conver to float, then from db to linear
    arr = np.float32(arr)
    valid = np.where(arr != nanval)
    arr[valid] = arr[valid] / 100.0
    if parameter == "sig0":
        arr[valid] = np.power(10,arr[valid]/10)
    arr[arr == nanval] = np.nan

    # prepare multi processing
    if not hasattr(sys.stdin, 'close'):
        def dummy_close():
            pass

        sys.stdin.close = dummy_close

    # prepare multi processing
    # dump arrays to temporary folder
    temp_folder = tempfile.mkdtemp()
    filename_in = os.path.join(temp_folder, 'joblib_dump1.mmap')
    if os.path.exists(filename_in): os.unlink(filename_in)
    _ = dump(arr, filename_in)
    inarr_map = load(filename_in, mmap_mode='r+')
    # output
    filename_out = os.path.join(temp_folder, 'joblib_dump2.mmap')
    outStd = np.memmap(filename_out, dtype=np.float32, mode='w+', shape=arr.shape)
    outStd[:] = np.nan

    #split arrays
    #inarr_splits = np.array_split(inarr_map, 8)
    #outarr_splits = np.array_split(outStd, 8)

    # get valid indices
    # valid_ind = np.where(np.isfinite(arr))
    # valid_ind = np.ravel_multi_index(valid_ind, arr.shape)
    valid_splits = np.array_split(valid_ind, 8)

    Parallel(n_jobs=8, verbose=5, max_nbytes=None)(
        delayed(_calc_std)(inarr_map, outStd, valid_splits[i], arr.shape) for i in range(8))

    #convert from linear to db
    if parameter == "sig0":
        valid = np.where(np.isfinite(outStd))
        outStd[valid] = 10*np.log10(outStd[valid])
    outStd[np.isnan(outStd)] = nanval

    try:
        shutil.rmtree(temp_folder)
    except:
        print("Failed to delete: " + temp_folder)

    return outStd


def _calc_std(inarr, outarr, valid_ind, shape):

    for i in valid_ind:
        ind = np.unravel_index(i, shape)

        if (ind[0] >= 5) and (ind[0] <= shape[0]-6) and (ind[1] >= 5) and (ind[1] <= shape[1]-6):
            outarr[ind] = np.nanstd(inarr[ind[0] - 5:ind[0] + 5, ind[1] - 5:ind[1] + 5])


def _local_mean(arr, nanval):
    # calculate local variance of image

    from scipy import ndimage

    # conver to float, then from db to linear
    arr = np.float32(arr)
    valid = np.where(arr != nanval)
    arr[valid] = arr[valid] / 100.0
    arr[valid] = np.power(10, arr[valid] / 10)
    arr[arr == nanval] = np.nan

    outStd = ndimage.generic_filter(arr, np.nanmean, size=3)

    # convert from linear to db
    valid = np.where(outStd != np.nan)
    outStd[valid] = 10 * np.log10(outStd[valid])
    outStd[outStd == np.nan] = nanval

    return outStd
