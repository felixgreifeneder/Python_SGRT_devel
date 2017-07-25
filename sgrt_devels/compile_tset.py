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
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from time import time
import math
from scipy.ndimage import median_filter
import datetime as dt
import ee
import scipy.stats
from sklearn import neighbors

def stackh5(root_path=None, out_path=None):


    #find h5 files to create time series
    filelist = rsearch.search_file(root_path, 'SMAP_L4*.h5')

    # load grid coodrinates
    ftmp = h5py.File(filelist[0])
    EASE_lats = np.array(ftmp['cell_lat'])
    EASE_lons = np.array(ftmp['cell_lon'])
    ftmp.close()

    # get number of files
    nfiles = len(filelist)

    # iterate through all files
    time_sec = np.full(nfiles, -9999, dtype=np.float64)
    sm_stack = np.full((nfiles, EASE_lats.shape[0], EASE_lats.shape[1]), -9999, dtype=np.float32)

    for find in range(nfiles):

        # load file
        ftmp = h5py.File(filelist[find], 'r')
        tmpsm = ftmp['Analysis_Data/sm_surface_analysis']
        # sm_subset = tmpsm[rowmin:rowmax, colmin:colmax]
        tmptime = ftmp['time'][0]

        time_sec[find] = tmptime
        sm_stack[find,:,:] = np.array(tmpsm)
        ftmp.close()

    # convert seconds since 2000-01-01 11:58:55.816 to datetime
    time_dt = [dt.datetime(2000,1,1,11,58,55,816) + dt.timedelta(seconds=x) for x in time_sec]

    # write to .h5 file
    f = h5py.File(out_path + 'SMAPL4_SMC_2015.h5', 'w')
    h5sm = f.create_dataset('SM_array', data=sm_stack)
    h5lats = f.create_dataset('LATS', data=EASE_lats)
    h5lons = f.create_dataset('LONS', data=EASE_lons)
    h5time = f.create_dataset('time', data=time_sec)
    f.close()


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


    def __init__(self, sgrt_root, sig0mpath, smcpath, dempath, outpath, uselc=True, subgrid='EU', tiles=None,
                 months=list([5,6,7,8,9]), ssm_target='SMAP', sig0_source='SGRT'):

        self.sgrt_root = sgrt_root
        self.sig0mpath = sig0mpath
        self.smcpath = smcpath
        self.dempath = dempath
        self.outpath = outpath
        self.uselc=uselc
        self.subgrid=subgrid
        self.months=months
        self.ssm_target=ssm_target
        self.sig0_source=sig0_source

        ee.Initialize()

        # create list of temporary files
        self.tempfiles = []

        if sig0_source == 'SGRT':
            extents = []
            for tname in tiles:
                print(tname)
                tmptile = Equi7Tile(subgrid + '010M_' + tname)
                extents.append(tmptile.extent)
                tmptile = None

            extents = np.array(extents)
        elif sig0_source == 'GEE':
            extents = tiles
            #extents = [36.39,-8.57,53.88,17.84]

        # Get the locations of training points
        if ssm_target == 'ASCAT':
            self.points = self.create_random_points(aoi=extents, sgrid='WARP', coords='latlon')
        elif ssm_target == 'SMAP':
            self.points = self.create_random_points(aoi=extents, sgrid='EASE20')

        # # extract parameters
        # if sig0_source == 'GEE':
        #     sig0lia = self.extr_sig0_lia_gee(extents)
        # elif sig0_source == 'SGRT':
        #     sig0lia = self.extr_sig0_lia(extents)

        #pickle.dump(sig0lia, open(self.outpath + 'sig0lia_dict.p', 'wb'))
        sig0lia = pickle.load(open(self.outpath + 'sig0lia_dict.p', 'rb'))
        self.trainingdata=sig0lia
        #np.savetxt(self.outpath + 'trainingdata.csv', sig0lia, delimiter=',')

        # filter samples with nan values
        samples = np.array([sig0lia.get(x) for x in sig0lia.keys()])
        samples = samples.transpose()
        valid = ~np.isnan(samples).any(axis=1) & ~np.isinf(samples).any(axis=1) & (np.array(sig0lia['ssm']) > 0)
        np.savetxt(self.outpath + 'trainingdata.csv', samples[valid,], delimiter=',')

        # define training and validation sets
        self.target = np.array(sig0lia['ssm'])[valid]
        self.features = np.vstack((#np.array(sig0lia['vv_tmean'])[valid],
                                   #np.array(sig0lia['vv_tstd'])[valid],
                                   #np.array(sig0lia['vh_tmean'])[valid],
                                   #np.array(sig0lia['vh_tstd'])[valid],
                                   #np.array(sig0lia['lon'])[valid],
                                   #np.array(sig0lia['lat'])[valid],
                                   np.array(sig0lia['sig0vv'])[valid],
                                   np.array(sig0lia['sig0vh'])[valid],
                                   np.array(sig0lia['vv_k1'])[valid],
                                   np.array(sig0lia['vh_k1'])[valid],
                                   np.array(sig0lia['vv_k2'])[valid],
                                   np.array(sig0lia['vh_k2'])[valid])).transpose()
                                   #np.array(sig0lia['vv_k1'])[valid],
                                   #np.array(sig0lia['vh_k1'])[valid],
                                   #np.array(sig0lia['vv_k2'])[valid],
                                   #np.array(sig0lia['vh_k2'])[valid])).transpose()
                                   #np.array(sig0lia['vv_k3'])[valid],
                                   #np.array(sig0lia['vh_k3'])[valid])).transpose()
                                   #np.array(sig0lia['vv_k4'])[valid],
                                   #np.array(sig0lia['vh_k4'])[valid])).transpose()

        for fi in self.tempfiles:
            os.remove(fi)

        print 'HELLO'


    def train_model(self):

        import scipy.stats
        from sklearn.ensemble import AdaBoostRegressor
        from sklearn.decomposition import PCA
        from sklearn.linear_model import TheilSenRegressor

        # filter bad ssm values
        valid = np.where(self.target > 0)
        self.target = self.target[valid[0]]
        self.features = self.features[valid[0],:]
        # filter nan values
        valid = ~np.any(np.isinf(self.features), axis=1)
        self.target = self.target[valid]
        self.features = self.features[valid,:]
        # filter nan
        valid = ~np.any(np.isnan(self.features), axis=1)
        self.target = self.target[valid]
        self.features = self.features[valid, :]

        # scaling
        scaler = sklearn.preprocessing.StandardScaler().fit(self.features)
        features = scaler.transform(self.features)

        # split into independent training data and test data
        #x_train, x_test, y_train, y_test, w_train, w_test = train_test_split(features, self.target, self.weights, test_size=0.3,
        #                                                    train_size=0.7, random_state=0)

        x_train, x_test, y_train, y_test = train_test_split(features, self.target,
                                                            test_size=0.2,
                                                            train_size=0.8)#, random_state=70)#, random_state=42)
        x_train = features
        y_train = self.target
        x_test = features
        y_test = self.target


        # ...----...----...----...----...----...----...----...----...----...
        # SVR -- SVR -- SVR -- SVR -- SVR -- SVR -- SVR -- SVR -- SVR -- SVR
        # ---....---....---....---....---....---....---....---....---....---

        dictCV = dict(C =       np.logspace(-2,2,10),
                      gamma =   np.logspace(-2,-0.5,10),
                      epsilon = np.logspace(-2, -0.5,10),
                      #nu = [0.01, 0.2, 0.5, 0.7, 1],
                      kernel = ['rbf'])
        # dictCV = dict(C=scipy.stats.expon(scale=100),
        #               gamma=scipy.stats.expon(scale=.1),
        #               epsilon=scipy.stats.expon(scale=.1),
        #               kernel=['rbf'])

        # specify kernel
        # svr_rbf = SVR(kernel = 'rbf')
        svr_rbf = SVR()

        # parameter tuning -> grid search
        start = time()
        print('Model training startet at: ' + str(start))
        #
        # SVR --- SVR --- SVR --- SVR --- SVR --- SVR --- SVR
        #
        gdCV = GridSearchCV(estimator=svr_rbf,
                            param_grid=dictCV,
                            n_jobs=8,
                            verbose=1,
                            pre_dispatch='all',
                            cv=KFold(n_splits=5, shuffle=True, random_state=42),
                            scoring='r2')
        # gdCV = RandomizedSearchCV(estimator=svr_rbf,
        #                           param_distributions=dictCV,
        #                           n_iter=100, # iterations used to be 200
        #                           n_jobs=8,
        #                           pre_dispatch='all',
        #                           cv=KFold(n_splits=3, shuffle=True, random_state=42), # cv used to be 10
        #                           verbose=1)
                                  #fit_params={"sample_weight": w_train})

        #
        # AdaBoost --- AdaBoost --- AdaBoost --- AdaBoost --- AdaBoost
        # gdCV = AdaBoostRegressor(DecisionTreeRegressor(max_depth=30), n_estimators=200, random_state=np.random.RandomState(1))

        #
        # GPR --- GPR --- GPR --- GPR --- GPR --- GPR --- GPR
        #
        # from sklearn.gaussian_process import GaussianProcessRegressor
        # from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
        #
        # kernel = 1.0 * RBF(length_scale=100.0, length_scale_bounds=(1e-2, 1e3)) \
        #          + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))
        # gdCV = GaussianProcessRegressor(kernel=kernel, alpha=0.0)

        # Linear - Linear - Linear

        #gdCV = TheilSenRegressor(random_state=42, n_jobs=8, verbose=True)

        gdCV.fit(x_train, y_train)
        #print(gdCV.best_params_)
        # prediction on test set
        y_CV_rbf = gdCV.predict(x_test)
        #print(gdCV.feature_importances_)

        r = np.corrcoef(y_test, y_CV_rbf)
        error = np.sqrt(np.sum(np.square(y_test - y_CV_rbf)) / len(y_test))

        print(r)
        print(error)
        #print(gdCV.best_params_)
        print('Elapse time for training: ' + str(time() - start))

        #pickle.dump((gdCV.best_estimator_, scaler, pca_transformer), open(self.outpath + 'mlmodel.p', 'wb'))
        pickle.dump((gdCV, scaler), open(self.outpath + 'mlmodel.p', 'wb'))


        print('SVR performance based on test-set')
        print('R: ' + str(r[0,1]))
        print('RMSE. ' + str(error))

        if self.ssm_target == 'SMAP':
            pltlims = 0.7
        else:
            pltlims = 100

        # create plots
        plt.figure(figsize = (6,6))
        plt.scatter(y_test, y_CV_rbf, c='g', label='True vs Est')
        plt.xlim(0,pltlims)
        plt.ylim(0,pltlims)
        plt.xlabel("SMAP L4 SMC [m3m-3]")
        plt.ylabel("Estimated SMC [m3m-3]")
        plt.plot([0,pltlims],[0,pltlims], 'k--')
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
        plt.xlim(0, pltlims)
        plt.ylim(0, pltlims)
        plt.xlabel("SMAP L4 SMC [m3m-3]")
        plt.ylabel("Estimated SMC [m3m-3]")
        plt.plot([0, pltlims], [0, pltlims], 'k--')
        plt.savefig(self.outpath + 'truevsest_training.png')
        plt.close()

        #self.SVRmodel = gdCV
        #self.scaler = scaler
        return (gdCV, scaler)

    # training the SVR per grid-point
    def train_model_alternative(self):


        from joblib import Parallel, delayed, load, dump
        import sys
        import tempfile
        import shutil
        import os

        # filter bad ssm values
        valid = np.where(self.target > 0)
        self.target = self.target[valid[0]]
        self.features = self.features[valid[0], :]
        # filter nan values
        valid = ~np.any(np.isinf(self.features), axis=1)
        self.target = self.target[valid]
        self.features = self.features[valid, :]
        # filter nan
        valid = ~np.any(np.isnan(self.features), axis=1)
        self.target = self.target[valid]
        self.features = self.features[valid, :]

        model_list = list()
        true = np.array([])
        estimated = np.array([])

        b = np.ascontiguousarray(self.features[:,0:2]).view(np.dtype((np.void, self.features[:,0:2].dtype.itemsize * self.features[:,0:2].shape[1])))
        _, idx = np.unique(b, return_index=True)

        # prepare multi processing
        # dump arrays to temporary folder
        temp_folder = tempfile.mkdtemp(dir='/tmp/')
        filename_in = os.path.join(temp_folder, 'joblib_dump1.mmap')
        if os.path.exists(filename_in): os.unlink(filename_in)
        _ = dump((self.features, self.target), filename_in)
        param_memmap = load(filename_in, mmap_mode='r+')

        if not hasattr(sys.stdin, 'close'):
            def dummy_close():
                pass

            sys.stdin.close = dummy_close

        #while (len(self.target) > 1):
        model_list = Parallel(n_jobs=8, verbose=5, max_nbytes=None)(delayed(_generate_model)(param_memmap[0], param_memmap[1],i) for i in idx)

        try:
            shutil.rmtree(temp_folder)
        except:
            print("Failed to delete: " + temp_folder)

        # filter model list
        model_list_fltrd = list()
        for tmp_i in range(len(model_list)):
            if model_list[tmp_i]['quality'] == 'good':
                model_list_fltrd.append(model_list[tmp_i])

        model_list = model_list_fltrd


        # generate nn model
        nn_target = np.array([str(x) for x in range(len(model_list))])
        nn_features = np.array([x['model_attr'] for x in model_list])

        clf = neighbors.KNeighborsClassifier()
        clf.fit(nn_features, nn_target)

        pickle.dump((model_list, clf), open(self.outpath + 'mlmodel.p', 'wb'))



        return (model_list, clf)


    def train_model_alternative_linear(self):


        from joblib import Parallel, delayed, load, dump
        import sys
        import tempfile
        import shutil
        import os

        # filter bad ssm values
        valid = np.where(self.target > 0)
        self.target = self.target[valid[0]]
        self.features = self.features[valid[0], :]
        # filter nan values
        valid = ~np.any(np.isinf(self.features), axis=1)
        self.target = self.target[valid]
        self.features = self.features[valid, :]
        # filter nan
        valid = ~np.any(np.isnan(self.features), axis=1)
        self.target = self.target[valid]
        self.features = self.features[valid, :]

        model_list = list()
        true = np.array([])
        estimated = np.array([])

        b = np.ascontiguousarray(self.features[:,0:2]).view(np.dtype((np.void, self.features[:,0:2].dtype.itemsize * self.features[:,0:2].shape[1])))
        _, idx = np.unique(b, return_index=True)

        # prepare multi processing
        # dump arrays to temporary folder
        temp_folder = tempfile.mkdtemp(dir='/tmp/')
        filename_in = os.path.join(temp_folder, 'joblib_dump1.mmap')
        if os.path.exists(filename_in): os.unlink(filename_in)
        _ = dump((self.features, self.target), filename_in)
        param_memmap = load(filename_in, mmap_mode='r+')

        if not hasattr(sys.stdin, 'close'):
            def dummy_close():
                pass

            sys.stdin.close = dummy_close

        #while (len(self.target) > 1):
        model_list = Parallel(n_jobs=8, verbose=5, max_nbytes=None)(delayed(_generate_model_linear)(param_memmap[0], param_memmap[1],i) for i in idx)
        #model_list = [_generate_model_linear(param_memmap[0], param_memmap[1], i) for i in idx]

        try:
            shutil.rmtree(temp_folder)
        except:
            print("Failed to delete: " + temp_folder)

        # filter model list
        model_list_fltrd = list()
        for tmp_i in range(len(model_list)):
            if model_list[tmp_i]['quality'] == 'good':
                model_list_fltrd.append(model_list[tmp_i])

        model_list = model_list_fltrd
        print(len(model_list))
        #model_list = model_list


        # generate nn model
        nn_target = np.array([str(x) for x in range(len(model_list))])
        nn_features = np.array([x['model_attr'] for x in model_list])

        clf = neighbors.KNeighborsClassifier()
        clf.fit(nn_features, nn_target)

        pickle.dump((model_list, clf), open(self.outpath + 'mlmodel.p', 'wb'))



        return (model_list, clf)


    def get_terrain(self, x, y, dx=1, dy=1):
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
        h = elevBand.ReadAsArray(int((x-elevGeo[0])/10), int((elevGeo[3]-y)/10), dx, dy)
        elev = None

        # aspect
        filename = glob.glob(self.dempath + tilename + '*_aspect.tif')
        asp = gdal.Open(filename[0], gdal.GA_ReadOnly)
        aspBand = asp.GetRasterBand(1)
        aspGeo = asp.GetGeoTransform()
        a = aspBand.ReadAsArray(int((x-aspGeo[0])/10), int((aspGeo[3]-y)/10), dx, dy)
        asp = None

        # slope
        filename = glob.glob(self.dempath + tilename + '*_slope.tif')
        slp = gdal.Open(filename[0], gdal.GA_ReadOnly)
        slpBand = slp.GetRasterBand(1)
        slpGeo = slp.GetGeoTransform()
        s = slpBand.ReadAsArray(int((x-slpGeo[0])/10), int((slpGeo[3]-y)/10), dx, dy)
        slp = None

        if dx == 1:
            return (h[0,0], a[0,0], s[0,0])
        else:
            return (h, a, s)


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

        valLClist = [10, 11, 12, 13, 18, 19, 20, 21, 26, 27, 28, 29]

        # cycle through all points
        for px in self.points:

            print('Grid-Point ' + str(cntr+1) + '/' + str(len(self.points)))

            # if cntr < 73:
            #     cntr = cntr + 1
            #     continue

            # define SMAP extent coordinates
            px_xmin = px[0] - 4500
            px_xmax = px[0] + 4500
            px_ymin = px[1] - 4500
            px_ymax = px[1] + 4500

            # read sig0 stack
            tmp_series = exTS.extr_SIG0_LIA_ts(self.sgrt_root, 'S1AIWGRDH', 'A0111', 'resampled', 10,
                                                   px_xmin, px_ymax, 900, 900, pol_name=['VV', 'VH'], grid='Equi7', subgrid=self.subgrid,
                                                   sat_pass='A', monthmask=self.months)

            vv_stack = np.array(tmp_series[1]['sig0'], dtype=np.float32)
            vh_stack = np.array(tmp_series[1]['sig02'], dtype=np.float32)
            lia_stack = np.array(tmp_series[1]['lia'], dtype=np.float32)

            # get lc
            if self.uselc == True:
                lc_data = self.get_lc(px_xmin, px_ymax, dx=900, dy=900)

            # get terrain
            #if self.subgrid == 'EU':
                #terr_data = self.get_terrain(px_xmin, px_ymax, dx=900, dy=900)

            if (self.uselc == True) and (self.subgrid == 'EU'):
                LCmask = np.reshape(np.in1d(lc_data, valLClist), (900,900))

            # mask backscatter_stack
            n_layers = vv_stack.shape[0]

            for li in range(n_layers):

                tmp_vv = vv_stack[li,:,:]
                tmp_vh = vh_stack[li,:,:]
                tmp_lia = lia_stack[li,:,:]

                if (self.uselc == True):
                    tmp_mask = np.where((tmp_vv < -2500) |
                                        (tmp_vv == -9999) |
                                        (tmp_vh < -2500) |
                                        (tmp_vh == -9999) |
                                        (tmp_lia < 1000) |
                                        (tmp_lia > 5000) |
                                        (LCmask == 0))
                else:
                    tmp_mask = np.where((tmp_vv < -2500) |
                                        (tmp_vv == -9999) |
                                        (tmp_vh < -2500) |
                                        (tmp_vh == -9999) |
                                        (tmp_lia < 1000) |
                                        (tmp_lia > 5000))

                tmp_vv[tmp_mask] = np.nan
                tmp_vh[tmp_mask] = np.nan
                tmp_lia[tmp_mask] = np.nan

            # db to lin
            vv_stack_lin = np.power(10, ((vv_stack/100.0) / 10))
            vh_stack_lin = np.power(10, ((vh_stack/100.0) / 10))
            lia_stack = lia_stack / 100.0

            # create spatial mean
            vv_smean = np.full(n_layers, -9999, dtype=np.float32)
            vh_smean = np.full(n_layers, -9999, dtype=np.float32)
            lia_smean = np.full(n_layers, -9999, dtype=np.float32)

            for li in range(n_layers):
                vv_smean[li] = 10 * np.log10(np.nanmean(vv_stack_lin[li,:,:]))
                vh_smean[li] = 10 * np.log10(np.nanmean(vh_stack_lin[li,:,:]))
                lia_smean[li] = np.nanmean(lia_stack[li,:,:])

            # create temporal mean and standard deviation
            vv_tmean = 10*np.log10(np.nanmean(np.nanmean(vv_stack_lin, axis=0)))
            vh_tmean = 10*np.log10(np.nanmean(np.nanmean(vh_stack_lin, axis=0)))
            vv_tstd = 10*np.log10(np.nanstd(np.nanstd(vv_stack_lin, axis=0)))
            vh_tstd = 10*np.log10(np.nanstd(np.nanstd(vh_stack_lin, axis=0)))

            # calculate k-statistics
            tmp_vv = np.nanmean(vv_stack/100.0, axis=0)
            tmp_vh = np.nanmean(vh_stack/100.0, axis=0)
            meank1VV = np.nanmean(tmp_vv)
            meank1VH = np.nanmean(tmp_vh)
            #meank2VV = moment(tmp_vv.ravel(), moment=2, nan_policy='omit')
            #meank2VH = moment(tmp_vh.ravel(), moment=2, nan_policy='omit')
            meank2VV = np.nanstd(tmp_vv)
            meank2VH = np.nanstd(tmp_vh)
            meank3VV = moment(tmp_vv.ravel(), moment=3, nan_policy='omit')
            meank3VH = moment(tmp_vh.ravel(), moment=3, nan_policy='omit')
            meank4VV = moment(tmp_vv.ravel(), moment=4, nan_policy='omit')
            meank4VH = moment(tmp_vh.ravel(), moment=4, nan_policy='omit')

            # calculate mean terrain parameters
            #H = terr_data[0]
            #A = terr_data[1]
            #S = terr_data[2]
            #meanH = np.mean(H[H != -9999])
            #meanA = np.mean(A[A != -9999])
            #meanS = np.mean(S[S != -9999])

            # ------------------------------------------
            # get ssm
            if self.ssm_target == 'SMAP':
                success, tmp_ssm = self.get_ssm(px[3], px[4])
            elif self.ssm_target == 'ASCAT':
                ssm_success, tmp_ssm = self.get_ssm(px[3], px[4], source='ASCAT')

            ssm_series = pd.Series(index=tmp_series[0])
            # ssm_dates = np.array(tmp_ssm[0])

            for i in range(len(ssm_series.index)):
                current_day = ssm_series.index[i]
                tmp_select = tmp_ssm.iloc[np.argmin(np.abs(tmp_ssm.index - current_day))]
                ssm_series.iloc[i] = tmp_select
                # id = np.where(ssm_dates == current_day)
                # if len(id[0]) > 0:
                #     ssm_series.iloc[i] = tmp_ssm[1][id]

            tmp_ssm = None


            if cntr == 0:
                ll = len(vv_smean)
                sig0lia_samples = {'ssm': list(np.array(ssm_series).squeeze()),
                                   'sig0vv': list(vv_smean),
                                   'sig0vh': list(vh_smean),
                                   'lia': list(lia_smean),
                                   'vv_tmean': [vv_tmean] * ll,
                                   'vh_tmean': [vh_tmean] * ll,
                                   'vv_tstd': [vv_tstd] * ll,
                                   'vh_tstd': [vh_tstd] * ll,
                                   'vv_k1': [meank1VV] * ll,
                                   'vh_k1': [meank1VH] * ll,
                                   'vv_k2': [meank2VV] * ll,
                                   'vh_k2': [meank2VH] * ll,
                                   'vv_k3': [meank3VV] * ll,
                                   'vh_k3': [meank3VH] * ll,
                                   'vv_k4': [meank4VV] * ll,
                                   'vh_k4': [meank4VH] * ll}#,
                                   #'height': [meanH] * ll,
                                   #'aspect': [meanA] * ll,
                                   #'slope': [meanS] * ll}
            else:
                ll = len(vv_smean)
                sig0lia_samples['ssm'].extend(list(np.array(ssm_series).squeeze()))
                sig0lia_samples['sig0vv'].extend(list(vv_smean))
                sig0lia_samples['sig0vh'].extend(list(vh_smean))
                sig0lia_samples['lia'].extend(list(lia_smean))
                sig0lia_samples['vv_tmean'].extend([vv_tmean] * ll)
                sig0lia_samples['vh_tmean'].extend([vh_tmean] * ll)
                sig0lia_samples['vv_tstd'].extend([vv_tstd] * ll)
                sig0lia_samples['vh_tstd'].extend([vh_tstd] * ll)
                sig0lia_samples['vv_k1'].extend([meank1VV] * ll)
                sig0lia_samples['vh_k1'].extend([meank1VH] * ll)
                sig0lia_samples['vv_k2'].extend([meank2VV] * ll)
                sig0lia_samples['vh_k2'].extend([meank2VH] * ll)
                sig0lia_samples['vv_k3'].extend([meank3VV] * ll)
                sig0lia_samples['vh_k3'].extend([meank3VH] * ll)
                sig0lia_samples['vv_k4'].extend([meank4VV] * ll)
                sig0lia_samples['vh_k4'].extend([meank4VH] * ll)
                #sig0lia_samples['height'].extend([meanH] * ll)
                #sig0lia_samples['aspect'].extend([meanA] * ll)
                #sig0lia_samples['slope'].extend([meanS] * ll)

            cntr = cntr + 1

        os.system('rm /tmp/*.vrt')

        return sig0lia_samples


    def extr_sig0_lia_gee(self, aoi, hour=None):

        import sgrt_devels.extr_TS as exTS
        import random
        import datetime as dt
        import os
        from scipy.stats import moment

        cntr = 0
        cntr2 = 0


        valLClist = [10, 11, 12, 13, 18, 19, 20, 21, 26, 27, 28, 29]
        sig0lia_samples = dict()

        # cycle through all points
        for px in self.points:

            print('Grid-Point ' + str(cntr2+1) + '/' + str(len(self.points)))

            # extract time series
            try:
                tmp_series = exTS.extr_SIG0_LIA_ts_GEE(float(px[3]), float(px[4]), bufferSize=6000, maskwinter=True)
            except:
                print('Failed to read S1 from GEE')
                cntr2 = cntr2 + 1
                continue

            for track_key in tmp_series.keys():

                if len(tmp_series[track_key][0]) < 10:
                    continue

                vv_series = np.array(tmp_series[track_key][1]['sig0'], dtype=np.float32)
                vh_series = np.array(tmp_series[track_key][1]['sig02'], dtype=np.float32)
                lia_series = np.array(tmp_series[track_key][1]['lia'], dtype=np.float32)

                # db to lin
                vv_series_lin = np.power(10, ((vv_series) / 10))
                vh_series_lin = np.power(10, ((vh_series) / 10))

                # create temporal mean and standard deviation
                vv_tmean = 10*np.log10(np.mean(vv_series_lin))
                vh_tmean = 10*np.log10(np.mean(vh_series_lin))
                vv_tstd = 10*np.log10(np.std(vv_series_lin))
                vh_tstd = 10*np.log10(np.std(vh_series_lin))

                # calculate k-statistics
                meank1VV = np.mean(np.log(vv_series_lin))
                meank1VH = np.mean(np.log(vh_series_lin))
                #meank2VV = moment(vv_series, moment=2)
                #meank2VH = moment(vh_series, moment=2)
                meank2VV = np.std(np.log(vv_series_lin))
                meank2VH = np.std(np.log(vh_series_lin))
                meank3VV = moment(vv_series, moment=3)
                meank3VH = moment(vh_series, moment=3)
                meank4VV = moment(vv_series, moment=4)
                meank4VH = moment(vh_series, moment=4)

                # ------------------------------------------
                # get ssm
                ssm_success, tmp_ssm = self.get_ssm(px[3], px[4], source=self.ssm_target)

                # check if the ssm time series is valid
                if len(tmp_ssm) < 1:
                    continue
                if ssm_success == 0:
                    continue
                ssm_series = pd.Series(index=tmp_series[track_key][0])
                # ssm_dates = np.array(tmp_ssm[0])

                for i in range(len(ssm_series.index)):
                    current_day = ssm_series.index[i]
                    timediff = np.min(np.abs(tmp_ssm.index - current_day))
                    if timediff > dt.timedelta(days=1):
                        continue

                    tmp_select = tmp_ssm.iloc[np.argmin(np.abs(tmp_ssm.index - current_day))]
                    ssm_series.iloc[i] = tmp_select

                tmp_ssm = None


                if cntr == 0:
                    ll = len(vv_series)
                    sig0lia_samples = {'ssm': list(np.array(ssm_series).squeeze()),
                                       'sig0vv': list(vv_series),
                                       'sig0vh': list(vh_series),
                                       'lia': list(lia_series),
                                       'vv_tmean': [vv_tmean] * ll,
                                       'vh_tmean': [vh_tmean] * ll,
                                       'vv_tstd': [vv_tstd] * ll,
                                       'vh_tstd': [vh_tstd] * ll,
                                       'vv_k1': [meank1VV] * ll,
                                       'vh_k1': [meank1VH] * ll,
                                       'vv_k2': [meank2VV] * ll,
                                       'vh_k2': [meank2VH] * ll,
                                       'vv_k3': [meank3VV] * ll,
                                       'vh_k3': [meank3VH] * ll,
                                       'vv_k4': [meank4VV] * ll,
                                       'vh_k4': [meank4VH] * ll,
                                       'lon': [px[3]] * ll,
                                       'lat': [px[4]] * ll}
                else:
                    ll = len(vv_series)
                    sig0lia_samples['ssm'].extend(list(np.array(ssm_series).squeeze()))
                    sig0lia_samples['sig0vv'].extend(list(vv_series))
                    sig0lia_samples['sig0vh'].extend(list(vh_series))
                    sig0lia_samples['lia'].extend(list(lia_series))
                    sig0lia_samples['vv_tmean'].extend([vv_tmean] * ll)
                    sig0lia_samples['vh_tmean'].extend([vh_tmean] * ll)
                    sig0lia_samples['vv_tstd'].extend([vv_tstd] * ll)
                    sig0lia_samples['vh_tstd'].extend([vh_tstd] * ll)
                    sig0lia_samples['vv_k1'].extend([meank1VV] * ll)
                    sig0lia_samples['vh_k1'].extend([meank1VH] * ll)
                    sig0lia_samples['vv_k2'].extend([meank2VV] * ll)
                    sig0lia_samples['vh_k2'].extend([meank2VH] * ll)
                    sig0lia_samples['vv_k3'].extend([meank3VV] * ll)
                    sig0lia_samples['vh_k3'].extend([meank3VH] * ll)
                    sig0lia_samples['vv_k4'].extend([meank4VV] * ll)
                    sig0lia_samples['vh_k4'].extend([meank4VH] * ll)
                    sig0lia_samples['lon'].extend([px[3]] * ll)
                    sig0lia_samples['lat'].extend([px[4]] * ll)

                cntr = cntr + 1

            cntr2 = cntr2 + 1

        #os.system('rm /tmp/*.vrt')
        return(sig0lia_samples)


    def create_random_points(self, aoi=None, sgrid='EASE20', coords='equi7'):

        import pygeogrids.netcdf as nc

        if coords == 'latlon':

            tmpgrid = Equi7Grid(10)
            tmpsgrid, eqlatmin, eqlonmin = tmpgrid.lonlat2equi7xy(aoi[1],aoi[0])
            tmpsgrid, eqlatmax, eqlonmax = tmpgrid.lonlat2equi7xy(aoi[3], aoi[2])
            aoi = [[int(eqlatmin), int(eqlonmin), int(eqlatmax), int(eqlonmax)]]

        if sgrid == 'EASE20':
            # create a list of EASE grid point within the aois
            # SMAP 9km EASE grid

            # get lat/lon of aoi
            Eq7SAR = Equi7Grid(10)

            # load EASE grid definition
            EASE_lats = np.fromfile('/mnt/SAT/Workspaces/GrF/01_Data/EASE20/EASE2_M09km.lats.3856x1624x1.double', dtype=np.float64)
            EASE_lons = np.fromfile('/mnt/SAT/workspaces/GrF/01_Data/EASE20/EASE2_M09km.lons.3856x1624x1.double', dtype=np.float64)
            EASE_lats = EASE_lats.reshape(1624, 3856)
            EASE_lons = EASE_lons.reshape(1624, 3856)

            # find valid ease locations
            points = set()

            # check if EASE grid point is within one of the aois
            for irow in range(1624):
            #for irow in range(200,210):
                for icol in range(3856):
                #for icol in range(2000,2100):

                    # calculate Eq7 coordinates
                    EASE_lon = EASE_lons[irow, icol]
                    EASE_lat = EASE_lats[irow, icol]
                    # this is temporary to speed up the process - only for central europe
                    #if (EASE_lat < 10) | (EASE_lat > 50) | (EASE_lon < -14) | (EASE_lon > 25):
                    if (EASE_lat < 29) | (EASE_lat > 32) | (EASE_lon < 34) | (EASE_lon > 36):
                        continue
                    EASE_xy = Eq7SAR.lonlat2equi7xy(EASE_lon, EASE_lat)

                    # iterate through all aois to determine wether one of them includes the curent EASE grid point
                    subaoi_counter = 0
                    for subaoi in aoi:
                        if EASE_xy[0] == self.subgrid:
                            if (EASE_xy[1] > (subaoi[0]+5000)) & (EASE_xy[1] < (subaoi[2]-5000)) & \
                                    (EASE_xy[2] > (subaoi[1]+5000)) & (EASE_xy[2] < (subaoi[3]-5000)):
                                # check land cover
                                # get land cover
                                if self.uselc == True:
                                    try:
                                        LCpx = self.get_lc(EASE_xy[1] - 4500, EASE_xy[2] + 4500, dx=900, dy=900)
                                        ValidLCind = np.where((LCpx == 10) | (LCpx == 12) |
                                                              (LCpx == 13) | (LCpx == 18) |
                                                              (LCpx == 26) | (LCpx == 29) |
                                                              (LCpx == 32) | (LCpx == 11) |
                                                              # (LCpx == 16) | (LCpx == 17) |
                                                              (LCpx == 19) | (LCpx == 20) |
                                                              (LCpx == 21) | (LCpx == 27) |
                                                              (LCpx == 28))
                                        # check if at least 10 percent are usable pixels
                                        ValidPrec = len(ValidLCind[0]) / (900.0 * 900.0)
                                    except:
                                        ValidPrec = 0
                                else:
                                    ValidPrec = 1

                                if ValidPrec >= 0.3:
                                    # add point to list
                                    points.add((int(round(EASE_xy[1])), int(round(EASE_xy[2])), subaoi_counter, EASE_lon, EASE_lat))

                        subaoi_counter = subaoi_counter + 1

            print(len(points))

        elif sgrid == 'WARP':

            # get lat/lon of aoi
            Eq7SAR = Equi7Grid(10)

            # find valid ease locations
            points = set()

            # load WARP grid configs
            warpgrid = nc.load_grid('/mnt/SAT/Workspaces/GrF/01_Data/ASCAT/h109_auxiliary/grid/TUW_WARP5_grid_info_2_1.nc')

            # iterate through all aois
            subaoi_counter = 0
            for subaoi in aoi:
                latmin, lonmin = Eq7SAR.equi7xy2lonlat(self.subgrid, subaoi[0], subaoi[1])
                latmax, lonmax = Eq7SAR.equi7xy2lonlat(self.subgrid, subaoi[2], subaoi[3])
                warpsubset = warpgrid.get_bbox_grid_points(latmin, latmax, lonmin, lonmax)

                # iterate through all gpis within the current aois
                for gpi in warpsubset:

                    warp_lat, warp_lon = warpgrid.gpi2lonlat(gpi)
                    EASE_xy = Eq7SAR.lonlat2equi7xy(float(warp_lon), float(warp_lat))

                    points.add((int(round(EASE_xy[1])), int(round(EASE_xy[2])), subaoi_counter, float(warp_lon), float(warp_lat)))

                subaoi_counter = subaoi_counter + 1


        return(points)


    def get_lc(self, x, y, dx=1, dy=1):

        # set up land cover grid
        Eq7LC = Equi7Grid(75)
        Eq7Sig0 = Equi7Grid(10)

        # get tile name of 75 Equi7 grid to check land-cover
        tilename = Eq7LC.identfy_tile(self.subgrid, (x, y))
        tilename_res = Eq7Sig0.identfy_tile(self.subgrid, (x,y))

        resLCpath = self.outpath + 'LC_' + tilename_res + '.tif'

        # check if resampled tile alread exisits
        if os.path.exists(resLCpath) == False:

            self.tempfiles.append(resLCpath)
            # resample lc to sig0
            # source
            LCtile = SgrtTile(dir_root=self.sgrt_root,
                              product_id='S1AIWGRDH',
                              soft_id='E0110',
                              product_name='CORINE06',
                              ftile=tilename,
                              src_res=75)

            LCfilename = [xs for xs in LCtile._tile_files]
            LCfilename = LCtile.dir + '/' + LCfilename[0] + '.tif'

            LC = gdal.Open(LCfilename, gdal.GA_ReadOnly)
            LC_proj = LC.GetProjection()
            LC_geotrans = LC.GetGeoTransform()

            #target
            Sig0tname = Eq7Sig0.identfy_tile(self.subgrid, (x,y))
            S0tile = SgrtTile(dir_root=self.sgrt_root,
                              product_id='S1AIWGRDH',
                              soft_id='A0111',
                              product_name='resampled',
                              ftile=Sig0tname,
                              src_res=10)
            Sig0fname = [xs for xs in S0tile._tile_files]
            Sig0fname = S0tile.dir + '/' + Sig0fname[0] + '.tif'
            s0ds = gdal.Open(Sig0fname, gdal.GA_ReadOnly)
            s0_proj = s0ds.GetProjection()
            s0_geotrans = s0ds.GetGeoTransform()
            wide = s0ds.RasterXSize
            high = s0ds.RasterYSize

            # resample
            resLCds = gdal.GetDriverByName('GTiff').Create(resLCpath, wide, high, 1, gdalconst.GDT_Byte)
            resLCds.SetGeoTransform(s0_geotrans)
            resLCds.SetProjection(s0_proj)

            gdal.ReprojectImage(LC, resLCds, LC_proj, s0_proj, gdalconst.GRA_NearestNeighbour)

            del resLCds

        LC = gdal.Open(resLCpath, gdal.GA_ReadOnly)
        LC_geotrans = LC.GetGeoTransform()
        LCband = LC.GetRasterBand(1)
        LCpx = LCband.ReadAsArray(xoff=int((x-LC_geotrans[0])/10.0), yoff=int((LC_geotrans[3]-y)/10.0), win_xsize=dx, win_ysize=dy)

        if dx==1 and dy==1:
            return LCpx[0][0]
        else:
            return LCpx


    def get_slope(self, x, y, dx=1, dy=1):

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
        slopeVV = SVVband.ReadAsArray(int((x-Stile.geotags['geotransform'][0])/10), int((Stile.geotags['geotransform'][3]-y)/10), dx, dy)[0][0]
        slopeVH = SVHband.ReadAsArray(int((x-Stile.geotags['geotransform'][0])/10), int((Stile.geotags['geotransform'][3]-y)/10), dx, dy)[0][0]

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


    def get_ssm(self, x, y, source='SMAP'):
        import math
        import datetime as dt

        success = 1

        grid = Equi7Grid(10)
        poi_lonlat = grid.equi7xy2lonlat(self.subgrid, x, y)

        if source == 'SMAP':
            #load file stack
            h5file = self.sgrt_root + '/Sentinel-1_CSAR/IWGRDH/ancillary/datasets/SMAPL4/SMAPL4_SMC_2015.h5'
            ssm_stack = h5py.File(h5file, 'r')

            # find the nearest gridpoint
            lat = ssm_stack['LATS']
            lon = ssm_stack['LONS']
            mindist = 10

            dist = np.sqrt(np.power(x - lon, 2) + np.power(y - lat, 2))
            mindist_loc = np.unravel_index(dist.argmin(), dist.shape)

            # stack time series of the nearest grid-point
            ssm = np.array(ssm_stack['SM_array'][:,mindist_loc[0], mindist_loc[1]])
            # create the time vector
            time_sec = np.array(ssm_stack['time'])
            time_dt = [dt.datetime(2000,1,1,11,58,55,816) + dt.timedelta(seconds=x) for x in time_sec]
            ssm_series = pd.Series(data=ssm, index=time_dt)

            ssm_stack.close()

        elif source == 'ASCAT':

            import ascat

            ascat_db = ascat.AscatH109_SSM('/mnt/SAT/Workspaces/GrF/01_Data/ASCAT/h109/SM_ASCAT_TS12.5_DR2016/',
                                           '/mnt/SAT/Workspaces/GrF/01_Data/ASCAT/h109_auxiliary/grid/',
                                           grid_info_filename = 'TUW_WARP5_grid_info_2_1.nc',
                                           static_path = '/mnt/SAT/Workspaces/GrF/01_Data/ASCAT/h109_auxiliary/static_layers/')

            ascat_series = ascat_db.read_ssm(x, y)
            if ascat_series.wetland_frac > 20:
                success = 0
            valid = np.where((ascat_series.data['proc_flag'] == 0) & (ascat_series.data['ssf'] == 1) & (ascat_series.data['snow_prob'] < 20))
            ssm_series = pd.Series(data=ascat_series.data['sm'][valid[0]], index=ascat_series.data.index[valid[0]])



        return (success, ssm_series)


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


    def ssm_ts(self, x, y, fdim, name=None):

        from extr_TS import read_NORM_SIG0
        from extr_TS import extr_SIG0_LIA_ts
        from scipy.stats import moment

        # extract parameters
        siglia_ts = extr_SIG0_LIA_ts(self.sgrt_root, 'S1AIWGRDH', 'A0111', 'resampled', 10, x, y, fdim, fdim,
                                     pol_name=['VV', 'VH'], grid='Equi7', sat_pass='A', monthmask=[1,2,3,4,5,6,7,8,9,10,11,12])

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
                fvect = [  # np.mean(np.array(lia_l)),
                            10 * np.log10(np.mean(np.power(10, (np.array(sig0mvv_l) / 10.)))),
                            10 * np.log10(np.mean(np.power(10, (np.array(sig0sdvv_l) / 10.)))),
                            10 * np.log10(np.mean(np.power(10, (np.array(sig0mvh_l) / 10.)))),
                            10 * np.log10(np.mean(np.power(10, (np.array(sig0sdvh_l) / 10.)))),
                            10 * np.log10(np.mean(np.power(10, (np.array(sig0_l) / 10.)))),
                            10 * np.log10(np.mean(np.power(10, (np.array(sig02_l) / 10.)))),
                            np.mean(k1vv_l),
                            np.mean(k1vh_l),
                            np.mean(k2vv_l),
                            np.mean(k2vh_l)]
                # fvect = [#np.mean(np.array(lia_l)),
                #     (10 * np.log10(np.mean(np.power(10, (np.array(sig0_l) / 10.))))) - (10 * np.log10(np.mean(np.power(10, (np.array(sig0mvv_l) / 10.))))),
                #     (10 * np.log10(np.mean(np.power(10, (np.array(sig0_l) / 10.))))) - (10 * np.log10(np.mean(np.power(10, (np.array(sig0sdvv_l) / 10.))))),
                #     (10 * np.log10(np.mean(np.power(10, (np.array(sig02_l) / 10.))))) - (10 * np.log10(np.mean(np.power(10, (np.array(sig0mvh_l) / 10.))))),
                #     (10 * np.log10(np.mean(np.power(10, (np.array(sig02_l) / 10.))))) - (10 * np.log10(np.mean(np.power(10, (np.array(sig0sdvh_l) / 10.))))),
                #          10 * np.log10(np.mean(np.power(10, (np.array(sig0_l) / 10.)))),
                #          10 * np.log10(np.mean(np.power(10, (np.array(sig02_l) / 10.)))),
                #          #10 * np.log10(np.std(np.power(10, (np.array(sig0_l) / 10.)))),
                #          #10 * np.log10(np.std(np.power(10, (np.array(sig02_l) / 10.)))),
                #     (10 * np.log10(np.mean(np.power(10, (np.array(sig0_l) / 10.))))) - np.mean(k1vv_l),
                #     (10 * np.log10(np.mean(np.power(10, (np.array(sig02_l) / 10.))))) - np.mean(k1vh_l),
                #     (10 * np.log10(np.mean(np.power(10, (np.array(sig0_l) / 10.))))) - np.mean(k2vv_l),
                #     (10 * np.log10(np.mean(np.power(10, (np.array(sig02_l) / 10.))))) - np.mean(k2vh_l),
                #     (10 * np.log10(np.mean(np.power(10, (np.array(sig0_l) / 10.))))) - np.mean(k3vv_l),
                #     (10 * np.log10(np.mean(np.power(10, (np.array(sig02_l) / 10.))))) - np.mean(k3vh_l),
                #     (10 * np.log10(np.mean(np.power(10, (np.array(sig0_l) / 10.))))) - np.mean(k4vv_l),
                #     (10 * np.log10(np.mean(np.power(10, (np.array(sig02_l) / 10.))))) - np.mean(k4vh_l)]
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
        if name == None:
            outfile = self.outpath + 'ts' + str(x) + '_' + str(y)
        else:
            outfile = self.outpath + 'ts_' + name
        plt.savefig(outfile + '.png')
        plt.close()
        csvout = np.array([m.strftime("%d/%m/%Y") for m in ssm_ts_out[0][valid]], dtype=np.str)
        csvout2 = np.array(ssm_ts_out[1][valid], dtype=np.str)
        # np.savetxt(self.outpath + 'ts.csv', np.hstack((csvout, csvout2)), fmt="%-10c", delimiter=",")
        with open(outfile + '.csv', 'w') as text_file:
            [text_file.write(csvout[i] + ', ' + csvout2[i] + '\n') for i in range(len(csvout))]

        print(ssm_ts_out[0][valid])

        print("Done")


    def ssm_ts_alternative(self, x, y, fdim, name=None):

        from extr_TS import read_NORM_SIG0
        from extr_TS import extr_SIG0_LIA_ts
        from scipy.stats import moment

        # extract model attributes
        # model_attrs = np.append(self.mlmodel[0]['model_attr'], self.mlmodel[0]['r'][0,1])
        class_model = self.mlmodel[1]
        reg_model = self.mlmodel[0]
        #model_rs = self.mlmodel[0]['r'][0,1]

        # for mit in range(1,len(self.mlmodel)):
        #    model_attrs = np.vstack((model_attrs, np.append(self.mlmodel[mit]['model_attr'], self.mlmodel[mit]['r'][0,1])))
        #    #model_rs = np.vstack((model_rs, self.mlmodel[mit]['r'][0,1]))



        # extract parameters
        siglia_ts = extr_SIG0_LIA_ts(self.sgrt_root, 'S1AIWGRDH', 'A0111', 'resampled', 10, x, y, fdim, fdim,
                                     pol_name=['VV', 'VH'], grid='Equi7', sat_pass='A', monthmask=[1,2,3,4,5,6,7,8,9,10,11,12])

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
            lia_l = list()
            sig0mvv_l = list()
            sig0sdvv_l = list()
            sig0mvh_l = list()
            sig0sdvh_l = list()
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
                            sig0_l.append(np.float32(siglia_ts[1]['sig0'][i,iy,ix])/100.)
                            sig02_l.append(np.float32(siglia_ts[1]['sig02'][i,iy,ix])/100.)
                            lia_l.append(np.float32(siglia_ts[1]['lia'][i, iy, ix])/100.)
                            sig0mvv_l.append(sig0mVV[iy, ix] / 100.)
                            sig0sdvv_l.append(sig0sdVV[iy,ix] / 100.)
                            sig0mvh_l.append(sig0mVH[iy, ix] / 100.)
                            sig0sdvh_l.append(sig0sdVH[iy,ix] / 100.)

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
                model_attr = [10 * np.log10(np.mean(np.power(10, (np.array(sig0mvv_l) / 10.)))),
                              10 * np.log10(np.mean(np.power(10, (np.array(sig0sdvv_l) / 10.)))),
                              10 * np.log10(np.mean(np.power(10, (np.array(sig0mvh_l) / 10.)))),
                              10 * np.log10(np.mean(np.power(10, (np.array(sig0sdvh_l) / 10.))))]
                fvect = [10 * np.log10(np.mean(np.power(10, (np.array(sig0_l) / 10.)))),
                         10 * np.log10(np.mean(np.power(10, (np.array(sig02_l) / 10.))))]

                # def calc_nn(a, poi=None):
                #     if a[4] > 0.5:
                #         dist = np.sqrt(np.square(poi[0]-a[0]) +
                #                        np.square(poi[1]-a[1]) +
                #                        np.square(poi[2]-a[2]) +
                #                        np.square(poi[3]-a[3]))
                #     else:
                #         dist = 9999
                #
                #     return(dist)

                # find the best model
                #nn = np.argmin(np.apply_along_axis(calc_nn, 1, model_attrs, poi=np.array(model_attr)))
                nn = class_model.predict(model_attr)
                nn = int(nn[0])
                nn_model = reg_model[nn]['model']
                nn_scaler = reg_model[nn]['scaler']

                fvect = nn_scaler.transform(fvect)
                ssm_ts_out[1][i] = nn_model.predict(fvect)




        valid = np.where(ssm_ts_out[1] != -9999)
        xx = ssm_ts_out[0][valid]
        yy = ssm_ts_out[1][valid]

        plt.figure(figsize=(18, 6))
        plt.plot(xx,yy)
        plt.show()
        if name == None:
            outfile = self.outpath + 'ts' + str(x) + '_' + str(y)
        else:
            outfile = self.outpath + 'ts_' + name
        plt.savefig(outfile + '.png')
        plt.close()
        csvout = np.array([m.strftime("%d/%m/%Y") for m in ssm_ts_out[0][valid]], dtype=np.str)
        csvout2 = np.array(ssm_ts_out[1][valid], dtype=np.str)
        # np.savetxt(self.outpath + 'ts.csv', np.hstack((csvout, csvout2)), fmt="%-10c", delimiter=",")
        with open(outfile + '.csv', 'w') as text_file:
            [text_file.write(csvout[i] + ', ' + csvout2[i] + '\n') for i in range(len(csvout))]

        print(ssm_ts_out[0][valid])

        print("Done")


    def ssm_ts_gee(self, lon, lat, x, y, fdim, name=None, plotts=False):

        from extr_TS import read_NORM_SIG0
        from extr_TS import extr_SIG0_LIA_ts_GEE
        from scipy.stats import moment

        # ee.Initialize()

        # extract model attributes
        reg_model = self.mlmodel

        # extract parameters
        siglia_ts = extr_SIG0_LIA_ts_GEE(lon, lat, bufferSize=fdim)
        #siglia_ts_alldays = extr_SIG0_LIA_ts_GEE(lon, lat, bufferSize=fdim, maskwinter=False)

        # create ts stack
        cntr=0
        for track_id in siglia_ts.keys():

            # calculate k1,...,kN and sig0m
            temp_ts1 = siglia_ts[track_id][1]['sig0'].astype(np.float)
            temp_ts2 = siglia_ts[track_id][1]['sig02'].astype(np.float)
            valid = np.where(np.isfinite(temp_ts1) & np.isfinite(temp_ts2))
            temp_ts1 = temp_ts1[valid]
            temp_ts2 = temp_ts2[valid]

            ts_length = len(temp_ts1)

            if ts_length < 10:
                continue

            temp_ts1_lin = np.power(10, temp_ts1/10.)
            temp_ts2_lin = np.power(10, temp_ts2/10.)
            k1VV = np.mean(np.log(temp_ts1_lin))
            k1VH = np.mean(np.log(temp_ts2_lin))
            k2VV = np.std(np.log(temp_ts1_lin))
            k2VH = np.std(np.log(temp_ts1_lin))

            ts_length = len(temp_ts1)

            fmat_tmp = np.hstack((np.repeat(k1VV, ts_length).reshape(ts_length, 1),
                                  np.repeat(k1VH, ts_length).reshape(ts_length, 1),
                                  np.repeat(k2VV, ts_length).reshape(ts_length, 1),
                                  np.repeat(k2VH, ts_length).reshape(ts_length, 1),
                                  temp_ts1.reshape(ts_length, 1),
                                  temp_ts2.reshape(ts_length, 1)))
                                  #np.repeat(k2VV, ts_length).reshape(ts_length, 1),
                                  #np.repeat(k2VH, ts_length).reshape(ts_length, 1)))

            #dates_tmp = siglia_ts_alldays[track_id][0][valid]
            dates_tmp = siglia_ts[track_id][0][valid]

            if cntr == 0:
                fmat = fmat_tmp
                dates = dates_tmp
            else:
                fmat = np.vstack((fmat, fmat_tmp))
                dates = np.concatenate((dates, dates_tmp))

            cntr = cntr + 1



        ssm_estimated = np.full(len(dates), -9999, dtype=np.float)
        for i in range(len(dates)):

            nn_model = reg_model[0]
            nn_scaler = reg_model[1]
            fvect = nn_scaler.transform(fmat[i,:].reshape(1,-1))
            ssm_estimated[i] = nn_model.predict(fvect)


        valid = np.where(ssm_estimated != -9999)
        ssm_ts = pd.Series(ssm_estimated[valid], index=dates[valid])
        ssm_ts.sort_index(inplace=True)

        #valid = np.where(ssm_ts_out[1] != -9999)
        #xx = ssm_ts_out[0][valid]
        #yy = ssm_ts_out[1][valid]
        if plotts == True:
            plt.figure(figsize=(18, 6))
            # plt.plot(xx,yy)
            # plt.show()
            ssm_ts.plot()
            plt.show()

            if name == None:
                outfile = self.outpath + 's1ts' + str(x) + '_' + str(y)
            else:
                outfile = self.outpath + 's1ts_' + name

            plt.savefig(outfile + '.png')
            plt.close()
            csvout = np.array([m.strftime("%d/%m/%Y") for m in ssm_ts.index], dtype=np.str)
            csvout2 = np.array(ssm_ts, dtype=np.str)
            # np.savetxt(self.outpath + 'ts.csv', np.hstack((csvout, csvout2)), fmt="%-10c", delimiter=",")
            with open(outfile + '.csv', 'w') as text_file:
                [text_file.write(csvout[i] + ', ' + csvout2[i] + '\n') for i in range(len(csvout))]

        print(ssm_ts)

        print("Done")
        return(ssm_ts)


    def ssm_ts_gee_alternative(self, lon, lat, x, y, fdim, name=None, plotts=False):

        from extr_TS import read_NORM_SIG0
        from extr_TS import extr_SIG0_LIA_ts_GEE
        from scipy.stats import moment

        # ee.Initialize()

        # extract model attributes
        class_model = self.mlmodel[1]
        reg_model = self.mlmodel[0]

        # extract parameters
        siglia_ts = extr_SIG0_LIA_ts_GEE(lon, lat, bufferSize=fdim)
        #siglia_ts_alldays = extr_SIG0_LIA_ts_GEE(lon, lat, bufferSize=fdim, maskwinter=False)

        # create ts stack
        cntr=0
        for track_id in siglia_ts.keys():

            # calculate k1,...,kN and sig0m
            temp_ts1 = siglia_ts[track_id][1]['sig0'].astype(np.float)
            temp_ts2 = siglia_ts[track_id][1]['sig02'].astype(np.float)
            valid = np.where(np.isfinite(temp_ts1) & np.isfinite(temp_ts2))
            temp_ts1 = temp_ts1[valid]
            temp_ts2 = temp_ts2[valid]

            ts_length = len(temp_ts1)

            if ts_length < 10:
                continue

            temp_ts1_lin = np.power(10, temp_ts1/10.)
            temp_ts2_lin = np.power(10, temp_ts2/10.)
            sig0mVV = 10*np.log10(np.mean(temp_ts1_lin))
            sig0mVH = 10*np.log10(np.mean(temp_ts2_lin))
            sig0sdVV = 10*np.log10(np.std(temp_ts1_lin))
            sig0sdVH = 10*np.log10(np.std(temp_ts2_lin))
            k1VV = np.mean(np.log(temp_ts1_lin))
            k1VH = np.mean(np.log(temp_ts2_lin))
            k2VV = np.std(np.log(temp_ts1_lin))
            k2VH = np.std(np.log(temp_ts1_lin))
            k3VV = moment(temp_ts1, moment=3, nan_policy='omit')
            k3VH = moment(temp_ts2, moment=3, nan_policy='omit')
            k4VV = moment(temp_ts1, moment=4, nan_policy='omit')
            k4VH = moment(temp_ts2, moment=4, nan_policy='omit')

            #siglia_ts = None
            #temp_ts1 = siglia_ts_alldays[track_id][1]['sig0'].astype(np.float)
            #temp_ts2 = siglia_ts_alldays[track_id][1]['sig02'].astype(np.float)
            #valid = np.where(np.isfinite(temp_ts1) & np.isfinite(temp_ts2))
            #temp_ts1 = temp_ts1[valid]
            #temp_ts2 = temp_ts2[valid]

            #ts_length = len(temp_ts1)

            # fmat_tmp = np.hstack((np.repeat(sig0mVV, ts_length).reshape(ts_length, 1),
            #                       np.repeat(sig0sdVV, ts_length).reshape(ts_length, 1),
            #                       np.repeat(sig0mVH, ts_length).reshape(ts_length, 1),
            #                       np.repeat(sig0sdVH, ts_length).reshape(ts_length, 1),
            #                       temp_ts1.reshape(ts_length, 1),
            #                       temp_ts2.reshape(ts_length, 1),
            #                       np.repeat(k1VV, ts_length).reshape(ts_length, 1),
            #                       np.repeat(k1VH, ts_length).reshape(ts_length, 1)))
            #                       #np.repeat(k2VV, ts_length).reshape(ts_length, 1),
            #                       #np.repeat(k2VH, ts_length).reshape(ts_length, 1)))

            model_attr_tmp = np.hstack((np.repeat(k1VV, ts_length).reshape(ts_length, 1),
                                        np.repeat(k1VH, ts_length).reshape(ts_length, 1),
                                        np.repeat(k2VV, ts_length).reshape(ts_length, 1),
                                        np.repeat(k2VH, ts_length).reshape(ts_length, 1)))
            fmat_tmp = np.hstack((temp_ts1.reshape(ts_length, 1),
                                  temp_ts2.reshape(ts_length, 1)))

            #dates_tmp = siglia_ts_alldays[track_id][0][valid]
            dates_tmp = siglia_ts[track_id][0][valid]

            if cntr == 0:
                model_attr = model_attr_tmp
                fmat = fmat_tmp
                dates = dates_tmp
            else:
                model_attr = np.vstack((model_attr, model_attr_tmp))
                fmat = np.vstack((fmat, fmat_tmp))
                dates = np.concatenate((dates, dates_tmp))

            cntr = cntr + 1



        ssm_estimated = np.full(len(dates), -9999, dtype=np.float)
        ssm_error = np.full(len(dates), -9999, dtype=np.float)
        for i in range(len(dates)):

            nn = class_model.predict(model_attr[i,:].reshape(1,-1))
            nn = int(nn[0])
            if reg_model[nn]['quality'] == 'good':
                nn_model = reg_model[nn]['model']
                nn_scaler = reg_model[nn]['scaler']
                fvect = nn_scaler.transform(fmat[i,:].reshape(1,-1))
                ssm_estimated[i] = nn_model.predict(fvect)
                ssm_error[i] = reg_model[nn]['rmse']
            else:
                nn_model = reg_model[nn]['model']
                nn_scaler = reg_model[nn]['scaler']
                fvect = nn_scaler.transform(fmat[i, :].reshape(1, -1))
                ssm_estimated[i] = nn_model.predict(fvect)
                ssm_error[i] = reg_model[nn]['rmse']
                #ssm_estimated[i] = -50

        # nn_model = self.mlmodel[0]
        # nn_scaler = self.mlmodel[1]
        # fmat = nn_scaler.transform(fmat)
        # ssm_estimated = nn_model.predict(fmat)

        valid = np.where(ssm_estimated != -9999)
        ssm_ts = pd.Series(ssm_estimated[valid], index=dates[valid])
        error_ts = pd.Series(ssm_error[valid], index=dates[valid])
        ssm_ts.sort_index(inplace=True)
        error_ts.sort_index(inplace=True)

        #valid = np.where(ssm_ts_out[1] != -9999)
        #xx = ssm_ts_out[0][valid]
        #yy = ssm_ts_out[1][valid]
        if plotts == True:
            plt.figure(figsize=(18, 6))
            # plt.plot(xx,yy)
            # plt.show()
            ssm_ts.plot()
            plt.show()

            if name == None:
                outfile = self.outpath + 's1ts' + str(x) + '_' + str(y)
            else:
                outfile = self.outpath + 's1ts_' + name

            plt.savefig(outfile + '.png')
            plt.close()
            csvout = np.array([m.strftime("%d/%m/%Y") for m in ssm_ts.index], dtype=np.str)
            csvout2 = np.array(ssm_ts, dtype=np.str)
            # np.savetxt(self.outpath + 'ts.csv', np.hstack((csvout, csvout2)), fmt="%-10c", delimiter=",")
            with open(outfile + '.csv', 'w') as text_file:
                [text_file.write(csvout[i] + ', ' + csvout2[i] + '\n') for i in range(len(csvout))]

        print(ssm_ts)
        print(error_ts)

        print("Done")
        return(ssm_ts, error_ts)


    def ssm_map(self, date=None, path=None):

        from joblib import Parallel, delayed, load, dump
        import sys
        import tempfile
        import shutil

        for tname in self.tiles:

            #for date in ['D20160628_170640']:

            for date in self.get_filenames(tname):

                # check if file already exists
                if os.path.exists(self.outpath + tname+'_SSM_' + date + '.tif'):
                    print(tname + ' / ' + date + ' allready processed')
                    continue

                print("Retrieving soil moisture for " + tname + " / " + date)

                # get sig0 image to derive ssm
                bacArrs = self.get_sig0_lia(tname, date)
                #terrainArrs = self.get_terrain(tname)
                paramArr = self.get_params(tname, path)

                # create masks
                if self.uselc == True:
                    lc_mask = self.create_LC_mask(tname, bacArrs)
                sig0_mask = (bacArrs['sig0vv'][0] != -9999) & \
                            (bacArrs['sig0vv'][0] >= -2500) & \
                            (bacArrs['sig0vh'][0] != -9999) & \
                            (bacArrs['sig0vh'][0] >= -2500) & \
                            (bacArrs['lia'][0] >= 1000) & \
                            (bacArrs['lia'][0] <= 5000)
                # terr_mask = (terrainArrs['h'][0] != -9999) & \
                #             (terrainArrs['a'][0] != -9999) & \
                #             (terrainArrs['s'][0] != -9999)
                param_mask = (paramArr['k1VH'][0] != -9999) & \
                             (paramArr['k1VV'][0] != -9999) & \
                             (paramArr['k2VH'][0] != -9999) & \
                             (paramArr['k2VV'][0] != -9999)

                # extrapolation mask
                # sig0vv_extr = ((((bacArrs['sig0vv'][0]/100.0) - self.mlmodel[1].mean_[4]) / self.mlmodel[1].std_[4]) > self.mlmodel[0].best_estimator_.support_vectors_[:,4].min()) & \
                #             ((((bacArrs['sig0vv'][0] / 100.0) - self.mlmodel[1].mean_[4]) / self.mlmodel[1].std_[4]) < self.mlmodel[0].best_estimator_.support_vectors_[:, 4].max())
                # sig0vh_extr = ((((bacArrs['sig0vh'][0] / 100.0) - self.mlmodel[1].mean_[5]) / self.mlmodel[1].std_[5]) >
                #                self.mlmodel[0].best_estimator_.support_vectors_[:, 5].min()) & \
                #               ((((bacArrs['sig0vh'][0] / 100.0) - self.mlmodel[1].mean_[5]) / self.mlmodel[1].std_[5]) <
                #                self.mlmodel[0].best_estimator_.support_vectors_[:, 5].max())
                # k1vv_extr = ((((paramArr['k1VV'][0] / 100.0) - self.mlmodel[1].mean_[0]) / self.mlmodel[1].std_[0]) >
                #                self.mlmodel[0].best_estimator_.support_vectors_[:, 0].min()) & \
                #               ((((paramArr['k1VV'][0] / 100.0) - self.mlmodel[1].mean_[0]) / self.mlmodel[1].std_[0]) <
                #                self.mlmodel[0].best_estimator_.support_vectors_[:, 0].max())
                # k1vh_extr = ((((paramArr['k1VH'][0] / 100.0) - self.mlmodel[1].mean_[1]) / self.mlmodel[1].std_[1]) >
                #              self.mlmodel[0].best_estimator_.support_vectors_[:, 1].min()) & \
                #             ((((paramArr['k1VH'][0] / 100.0) - self.mlmodel[1].mean_[1]) / self.mlmodel[1].std_[1]) <
                #              self.mlmodel[0].best_estimator_.support_vectors_[:, 1].max())
                # k2vv_extr = ((((paramArr['k2VV'][0] / 100.0) - self.mlmodel[1].mean_[2]) / self.mlmodel[1].std_[2]) >
                #              self.mlmodel[0].best_estimator_.support_vectors_[:, 2].min()) & \
                #             ((((paramArr['k2VV'][0] / 100.0) - self.mlmodel[1].mean_[2]) / self.mlmodel[1].std_[2]) <
                #              self.mlmodel[0].best_estimator_.support_vectors_[:, 2].max())
                # k2vh_extr = ((((paramArr['k2VV'][0] / 100.0) - self.mlmodel[1].mean_[3]) / self.mlmodel[1].std_[3]) >
                #              self.mlmodel[0].best_estimator_.support_vectors_[:, 3].min()) & \
                #             ((((paramArr['k2VV'][0] / 100.0) - self.mlmodel[1].mean_[3]) / self.mlmodel[1].std_[3]) <
                #              self.mlmodel[0].best_estimator_.support_vectors_[:, 3].max())
                # extr_mask = sig0vv_extr & sig0vh_extr & k1vv_extr & k1vh_extr & k2vv_extr & k2vh_extr



                # combined mask
                if self.uselc == True:
                    #mask = lc_mask & sig0_mask & terr_mask & param_mask
                    mask = lc_mask & sig0_mask & param_mask
                else:
                    # mask = sig0_mask & terr_mask & param_mask
                    mask = sig0_mask & param_mask

                # resample
                # bacArrs, paramArr, terrainArrs = self.resample(bacArrs,terrainArrs,paramArr, mask, 5)
                bacArrs, paramArr = self.resample(bacArrs, paramArr, mask, 5)
                print('Resampled: check')

                sig0_mask = (bacArrs['sig0vv'][0] != -9999) & \
                            (bacArrs['sig0vv'][0] >= -2500) & \
                            (bacArrs['sig0vh'][0] != -9999) & \
                            (bacArrs['sig0vh'][0] >= -2500) & \
                            (bacArrs['lia'][0] >= 1000) & \
                            (bacArrs['lia'][0] <= 5000)
                # terr_mask = (terrainArrs['h'][0] != -9999) & \
                #             (terrainArrs['a'][0] != -9999) & \
                #             (terrainArrs['s'][0] != -9999)
                param_mask = (paramArr['k1VH'][0] != -9999) & \
                            (paramArr['k1VV'][0] != -9999) & \
                            (paramArr['k2VH'][0] != -9999) & \
                            (paramArr['k2VV'][0] != -9999)

                # extrapolation mask
                sig0vv_extr = ((((bacArrs['sig0vv'][0] / 100.0) - self.mlmodel[1].mean_[4]) / self.mlmodel[1].std_[4]) >
                               self.mlmodel[0].best_estimator_.support_vectors_[:, 4].min()) & \
                              ((((bacArrs['sig0vv'][0] / 100.0) - self.mlmodel[1].mean_[4]) / self.mlmodel[1].std_[4]) <
                               self.mlmodel[0].best_estimator_.support_vectors_[:, 4].max())
                sig0vh_extr = ((((bacArrs['sig0vh'][0] / 100.0) - self.mlmodel[1].mean_[5]) / self.mlmodel[1].std_[5]) >
                               self.mlmodel[0].best_estimator_.support_vectors_[:, 5].min()) & \
                              ((((bacArrs['sig0vh'][0] / 100.0) - self.mlmodel[1].mean_[5]) / self.mlmodel[1].std_[5]) <
                               self.mlmodel[0].best_estimator_.support_vectors_[:, 5].max())
                k1vv_extr = ((((paramArr['k1VV'][0] / 100.0) - self.mlmodel[1].mean_[0]) / self.mlmodel[1].std_[0]) >
                             self.mlmodel[0].best_estimator_.support_vectors_[:, 0].min()) & \
                            ((((paramArr['k1VV'][0] / 100.0) - self.mlmodel[1].mean_[0]) / self.mlmodel[1].std_[0]) <
                             self.mlmodel[0].best_estimator_.support_vectors_[:, 0].max())
                k1vh_extr = ((((paramArr['k1VH'][0] / 100.0) - self.mlmodel[1].mean_[1]) / self.mlmodel[1].std_[1]) >
                             self.mlmodel[0].best_estimator_.support_vectors_[:, 1].min()) & \
                            ((((paramArr['k1VH'][0] / 100.0) - self.mlmodel[1].mean_[1]) / self.mlmodel[1].std_[1]) <
                             self.mlmodel[0].best_estimator_.support_vectors_[:, 1].max())
                k2vv_extr = ((((paramArr['k2VV'][0] / 100.0) - self.mlmodel[1].mean_[2]) / self.mlmodel[1].std_[2]) >
                             self.mlmodel[0].best_estimator_.support_vectors_[:, 2].min()) & \
                            ((((paramArr['k2VV'][0] / 100.0) - self.mlmodel[1].mean_[2]) / self.mlmodel[1].std_[2]) <
                             self.mlmodel[0].best_estimator_.support_vectors_[:, 2].max())
                k2vh_extr = ((((paramArr['k2VV'][0] / 100.0) - self.mlmodel[1].mean_[3]) / self.mlmodel[1].std_[3]) >
                             self.mlmodel[0].best_estimator_.support_vectors_[:, 3].min()) & \
                            ((((paramArr['k2VV'][0] / 100.0) - self.mlmodel[1].mean_[3]) / self.mlmodel[1].std_[3]) <
                             self.mlmodel[0].best_estimator_.support_vectors_[:, 3].max())
                extr_mask = sig0vv_extr & sig0vh_extr & k1vv_extr & k1vh_extr & k2vv_extr & k2vh_extr

                # create masks
                if self.uselc == True:
                    lc_mask = self.create_LC_mask(tname, bacArrs, res=1000)

                # combined mask
                if self.uselc == True:
                    #mask = lc_mask & sig0_mask & terr_mask & param_mask
                    mask = lc_mask & sig0_mask & param_mask & extr_mask
                else:
                    #mask = sig0_mask & terr_mask & param_mask
                    mask = sig0_mask & param_mask & extr_mask

                valid_ind = np.where(mask == True)
                valid_ind = np.ravel_multi_index(valid_ind, (1000,1000))

                #vv_sstd = _local_std(bacArrs['sig0vv'][0], -9999, valid_ind)
                #vh_sstd = _local_std(bacArrs['sig0vh'][0], -9999, valid_ind)
                #lia_sstd = _local_std(bacArrs['lia'][0], -9999, valid_ind, "lia")

                #bacStats = {"vv": vv_sstd, "vh": vh_sstd, 'lia': lia_sstd}
                bacStats = {'vv': bacArrs['sig0vv'][0], 'vh': bacArrs['sig0vh'][0], 'lia': bacArrs['lia'][0]}

                ssm_out = np.full((1000,1000), -9999, dtype=np.float32)

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
                #_ = dump((bacArrs, terrainArrs, paramArr, bacStats), filename_in)
                _ = dump((bacArrs, paramArr, bacStats), filename_in)
                large_memmap = load(filename_in, mmap_mode='r+')
                # output
                filename_out = os.path.join(temp_folder, 'joblib_dump2.mmap')
                ssm_out = np.memmap(filename_out, dtype=np.float32, mode='w+', shape=(1000,1000))
                ssm_out[:] = -9999

                # extract model attributes
                #model_attrs = self.mlmodel[0][0]['model_attr']

                #for mit in range(1, len(self.mlmodel[0])):
                #    model_attrs = np.vstack((model_attrs, self.mlmodel[0][mit]['model_attr']))
                #reg_model = self.mlmodel[0]
                #class_model = self.mlmodel[1]

                # predict SSM
                #Parallel(n_jobs=8, verbose=5, max_nbytes=None)(delayed(_estimate_ssm_alternative)(large_memmap[0],large_memmap[1],large_memmap[2], large_memmap[3],ssm_out,i,reg_model,class_model) for i in ind_splits)
                Parallel(n_jobs=8, verbose=5, max_nbytes=None)(delayed(_estimate_ssm)(large_memmap[0],large_memmap[1],large_memmap[2], ssm_out,i,self.mlmodel) for i in ind_splits)

                # _estimate_ssm(bacArrs, terrainArrs, paramArr, bacStats, ssm_out, valid_ind, self.mlmodel)

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
        #sig0mVV = Stile.read_tile(pattern='.*SIG0M.*VV'+path+'.*T1$')
        #sig0mVH = Stile.read_tile(pattern='.*SIG0M.*VH'+path+'.*T1$')
        #sig0sdVV = Stile.read_tile(pattern='.*SIGSD.*VV'+path+'.*T1$')
        #sig0sdVH = Stile.read_tile(pattern='.*SIGSD.*VH'+path+'.*T1$')
        k1VV = Stile.read_tile(pattern='.*K1.*VV'+path+'.*T1$')
        k1VH = Stile.read_tile(pattern='.*K1.*VH' + path + '.*T1$')
        k2VV = Stile.read_tile(pattern='.*K2.*VV' + path + '.*T1$')
        k2VH = Stile.read_tile(pattern='.*K2.*VH' + path + '.*T1$')
        #k3VV = Stile.read_tile(pattern='.*K3.*VV' + path + '.*T1$')
        #k3VH = Stile.read_tile(pattern='.*K3.*VH' + path + '.*T1$')
        #k4VV = Stile.read_tile(pattern='.*K4.*VV' + path + '.*T1$')
        #k4VH = Stile.read_tile(pattern='.*K4.*VH' + path + '.*T1$')

        return {#'slpVV': slpVV,
                #'slpVH': slpVH,
                #'sig0mVV': sig0mVV,
                #'sig0mVH': sig0mVH,
                #'sig0sdVV': sig0sdVV,
                #'sig0sdVH': sig0sdVH,
                'k1VV': k1VV,
                'k1VH': k1VH,
                'k2VV': k2VV,
                'k2VH': k2VH}
                #'k3VV': k3VV,
                #'k3VH': k3VH,
                #'k4VV': k4VV,
                #'k4VH': k4VH}


    def create_LC_mask(self, tname, bacArrs, res=10000):

        # get tile name in 75m lc grid
        eq7tile = Equi7Tile(self.subgrid + '010M_' + tname)
        tname75 = eq7tile.find_family_tiles(res=75)
        # load lc array, resampled to 10m
        lcArr = self.get_lc(tname75[0], bacArrs, res)

        #generate mask
        tmp = np.array(lcArr[0])
        mask = (tmp == 10) | (tmp == 12) | (tmp == 13) | (tmp == 18) | (tmp == 26) | (tmp == 29) | (tmp == 32) | \
               (tmp == 11) | (tmp == 19) | (tmp == 20) | (tmp == 21) | (tmp == 27) | (tmp == 28)

        return mask


    def get_lc(self, tname, bacArrs, res=10000):



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
        dst_width = res
        dst_height = res
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
        dst_width = 1000
        dst_height = 1000

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


    def resample(self, bacArrs, paramArrs, mask, resolution=10):

        # bacArrs
        #bacArrsRes = dict()
        for key in bacArrs.keys():
            tmparr = np.array(bacArrs[key][0], dtype=np.float32)
            tmpmeta = bacArrs[key][1]
            tmpmeta['geotransform'] = (tmpmeta['geotransform'][0], 100.0, tmpmeta['geotransform'][2], tmpmeta['geotransform'][3], tmpmeta['geotransform'][4], -100.0)
            tmparr[mask == False] = np.nan
            #tmparr = self.moving_average(tmparr, resolution)
            if key == 'sig0vv' or key == 'sig0vh':
                tmparr[mask] = np.power(10, tmparr[mask]/1000.0)
                #tmparr[mask == False] = np.nan
                tmparr_res = np.full((1000,1000), fill_value=np.nan)
                for ix in range(1000):
                    for iy in range(1000):
                        tmparr_res[iy,ix] = np.nanmean(tmparr[(iy*10):(iy*10+10),(ix*10):(ix*10+10)])

                tmparr_res = 1000. * np.log10(tmparr_res)
            else:
                #tmparr_res = (tmparr[::10, ::10] + tmparr[::10, 1::10] + tmparr[1::10, ::10] + tmparr[1::10,
                #                                                                               1::10]) / 100.
                tmparr_res = np.full((1000, 1000), fill_value=np.nan)
                for ix in range(1000):
                    for iy in range(1000):
                        tmparr_res[iy, ix] = np.nanmean(tmparr[(iy*10):(iy*10+10),(ix*10):(ix*10+10)])

            #bacArrs[key][0][:,:] = np.array(tmparr, dtype=np.int16)
            tmparr_res[np.isnan(tmparr_res)] = -9999
            bacArrs[key] = (np.array(tmparr_res, dtype=np.int16), tmpmeta)

        # paramArrs
        for key in paramArrs.keys():
            tmparr = np.array(paramArrs[key][0], dtype=np.float32)
            tmpmeta = paramArrs[key][1]
            tmpmeta['geotransform'] = (
            tmpmeta['geotransform'][0], 100.0, tmpmeta['geotransform'][2], tmpmeta['geotransform'][3],
            tmpmeta['geotransform'][4], -100.0)
            tmparr[mask == False] = np.nan
            #tmparr = self.moving_average(tmparr, resolution)
            #tmparr_res = (tmparr[::10, ::10] + tmparr[::10, 1::10] + tmparr[1::10, ::10] + tmparr[1::10, 1::10]) / 100.
            tmparr_res = np.full((1000, 1000), fill_value=np.nan)
            for ix in range(1000):
                for iy in range(1000):
                    tmparr_res[iy, ix] = np.nanmean(tmparr[(iy*10):(iy*10+10),(ix*10):(ix*10+10)])

            tmparr_res[np.isnan(tmparr_res)] = -9999
            paramArrs[key] = (tmparr_res, tmpmeta)

        # terrainArrs
        # for key in terrainArrs.keys():
        #     tmparr = np.array(terrainArrs[key][0], dtype=np.float32)
        #     tmparr[mask == False] = np.nan
        #     tmparr = self.moving_average(tmparr, resolution)
        #     terrainArrs[key][0][:,:] = tmparr

        return (bacArrs, paramArrs)


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


def _calc_nn(a, poi=None):
    dist = np.sqrt(np.square(poi[0] - a[0]) +
                   np.square(poi[1] - a[1]) +
                   np.square(poi[2] - a[2]) +
                   np.square(poi[3] - a[3]))
    return (dist)


def _estimate_ssm(bacArrs, paramArr, bacStats, ssm_out, valid_ind, mlmodel):

    for i in valid_ind:
        ind = np.unravel_index(i, (1000,1000))
        try:
            # compile feature vector
            sig0vv = bacArrs['sig0vv'][0][ind] / 100.0
            sig0vh = bacArrs['sig0vh'][0][ind] / 100.0
            #lia = bacArrs['lia'][0][ind] / 100.0
            #h = terrainArrs['h'][0][ind]
            #a = terrainArrs['a'][0][ind]
            #s = terrainArrs['s'][0][ind]
            #slpvv = paramArr['slpVV'][0][ind]
            #slpvh = paramArr['slpVH'][0][ind]
            #sig0mvv = paramArr['sig0mVV'][0][ind] / 100.0
            #sig0mvh = paramArr['sig0mVH'][0][ind] / 100.0
            #sig0sdvv = paramArr['sig0sdVV'][0][ind] / 100.0
            #sig0sdvh = paramArr['sig0sdVH'][0][ind] / 100.0
            #vvsstd = bacStats['vv'][ind]
            #vhsstd = bacStats['vh'][ind]
            #liasstd = bacStats['lia'][ind]
            k1VV = paramArr['k1VV'][0][ind]/100.
            k1VH = paramArr['k1VH'][0][ind]/100.
            k2VV = paramArr['k2VV'][0][ind]/100.
            k2VH = paramArr['k2VH'][0][ind]/100.
            #k3VV = paramArr['k3VV'][0][ind]/100.
            #k3VH = paramArr['k3VH'][0][ind]/100.
            #k4VV = paramArr['k4VV'][0][ind]/100.
            #k4VH = paramArr['k4VH'][0][ind]/100.

            fvect = [#sig0mvv,
                     #sig0sdvv,
                     #sig0mvh,
                     #sig0sdvh,
                     k1VV,
                     k1VH,
                     k2VV,
                     k2VH,
                     sig0vv,
                     sig0vh]

            fvect = mlmodel[1].transform(np.reshape(fvect, (1,-1)))
            # predict ssm
            predssm = mlmodel[0].predict(fvect)
            if predssm < 0:
                ssm_out[ind] = 0
            else:
                ssm_out[ind] = predssm
        except:
            ssm_out[ind] = -9999


def _estimate_ssm_alternative(bacArrs, terrainArrs, paramArr, bacStats, ssm_out, valid_ind, mlmodel, model_attrs):

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

            attr = [sig0mvv,
                     sig0sdvv,
                     sig0mvh,
                     sig0sdvh]

            fvect =  [sig0vv,
                      sig0vh]

            # find the best model
            #nn = np.argmin(np.apply_along_axis(_calc_nn, 1, model_attrs, poi=np.array(attr)))
            nn = model_attrs.predict(np.reshape(attr, (1,-1)))
            nn = int(nn[0])

            # load the best model
            nn_model = mlmodel[nn]['model']
            nn_scaler = mlmodel[nn]['scaler']


            fvect = nn_scaler.transform(np.reshape(fvect, (1,-1)))
            # predict ssm
            predssm = nn_model.predict(fvect)
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


def _generate_model(features, target, urowidx):

    rows_select = np.where((features[:, 0] == features[urowidx, 0]) &
                           (features[:, 1] == features[urowidx, 1]))# &
                           #(features[:, 2] == features[urowidx, 2]) &
                           #(features[:, 3] == features[urowidx, 3]))  # &
    # (self.features[:, 4] == self.features[0, 4]) &
    # (self.features[:, 5] == self.features[0, 5]))

    #if (len(rows_select) <= 9):
    #    # delete selected features from array
    #    # self.target = np.delete(self.target, rows_select)
    #    # self.features = np.delete(self.features, rows_select, axis=0)
    #    return

    utarg = np.copy(target[rows_select].squeeze())
    ufeat = np.copy(features[rows_select, 6::].squeeze())
    point_model = {'model_attr': features[0, 2:6]}

    # scaling
    scaler = sklearn.preprocessing.StandardScaler().fit(ufeat)
    ufeat = scaler.transform(ufeat)

    point_model['scaler'] = scaler

    # split into independent training data and test data
    x_train = ufeat
    y_train = utarg
    x_test = ufeat
    y_test = utarg

    # ...----...----...----...----...----...----...----...----...----...
    # SVR -- SVR -- SVR -- SVR -- SVR -- SVR -- SVR -- SVR -- SVR -- SVR
    # ---....---....---....---....---....---....---....---....---....---

    dictCV = dict(C=scipy.stats.expon(scale=100),
                  gamma=scipy.stats.expon(scale=.1),
                  epsilon=scipy.stats.expon(scale=.1),
                  kernel=['rbf'])

    # specify kernel
    svr_rbf = SVR()

    # parameter tuning -> grid search

    gdCV = RandomizedSearchCV(estimator=svr_rbf,
                              param_distributions=dictCV,
                              n_iter=200,  # iterations used to be 200
                              n_jobs=1,
                              # pre_dispatch='all',
                              cv=KFold(n_splits=2, shuffle=True),  # cv used to be 10
                              # cv=sklearn.model_selection.TimeSeriesSplit(n_splits=3),
                              verbose=0,
                              scoring='r2')

    gdCV.fit(x_train, y_train)
    # prediction on test set
    tmp_est = gdCV.predict(x_test)
    # estimate point accuracies
    tmp_r = np.corrcoef(y_test, tmp_est)
    error = np.sqrt(np.sum(np.square(y_test - tmp_est)) / len(y_test))

    point_model['model'] = gdCV
    point_model['rmse'] = error
    point_model['r'] = tmp_r

    if (tmp_r[0, 1] > 0.5) & (error < 20):

        # set quality flag
        point_model['quality'] = 'good'
        # add to overall list
        #estimated = np.append(estimated, tmp_est)
        #true = np.append(true, y_test)
    else:
        #return
        point_model['quality'] = 'bad'

    # add model to model list
    return(point_model)


def _generate_model_linear(features, target, urowidx):

    from sklearn.linear_model import TheilSenRegressor

    rows_select = np.where((features[:, 0] == features[urowidx, 0]) &
                           (features[:, 1] == features[urowidx, 1]))# &
                           #(features[:, 2] == features[urowidx, 2]) &
                           #(features[:, 3] == features[urowidx, 3]))  # &
    # (self.features[:, 4] == self.features[0, 4]) &
    # (self.features[:, 5] == self.features[0, 5]))

    #if (len(rows_select) <= 9):
    #    # delete selected features from array
    #    # self.target = np.delete(self.target, rows_select)
    #    # self.features = np.delete(self.features, rows_select, axis=0)
    #    return

    utarg = np.copy(target[rows_select].squeeze())
    ufeat = np.copy(features[rows_select, 6::].squeeze())
    point_model = {'model_attr': features[rows_select[0][0], 2:6]}

    # scaling
    scaler = sklearn.preprocessing.StandardScaler().fit(ufeat)
    ufeat = scaler.transform(ufeat)

    point_model['scaler'] = scaler

    # split into independent training data and test data
    x_train = ufeat
    y_train = utarg
    x_test = ufeat
    y_test = utarg

    # ...----...----...----...----...----...----...----...----...----...
    # Linear regression
    # ---....---....---....---....---....---....---....---....---....---

    # specify estimator
    estimator = TheilSenRegressor(random_state=42)

    # fit the model

    estimator.fit(x_train, y_train)
    # prediction on test set
    tmp_est = estimator.predict(x_test)
    # estimate point accuracies
    tmp_r = np.corrcoef(y_test, tmp_est)
    error = np.sqrt(np.sum(np.square(y_test - tmp_est)) / len(y_test))

    point_model['model'] = estimator
    point_model['rmse'] = error
    point_model['r'] = tmp_r

    if (tmp_r[0, 1] > 0.75) & (error < 5):

        # set quality flag
        point_model['quality'] = 'good'
        # add to overall list
        #estimated = np.append(estimated, tmp_est)
        #true = np.append(true, y_test)
    else:
        #return
        point_model['quality'] = 'bad'

    # add model to model list
    return(point_model)