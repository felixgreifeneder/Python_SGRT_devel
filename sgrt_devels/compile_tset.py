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
from sklearn.cross_validation import train_test_split
from time import time

def stackh5(root_path=None, out_path=None):


    #find h5 files to create time series
    filelist = rsearch.search_file(root_path, '*.h5')

    #get file size
    ftmp = h5py.File(filelist[0], 'r')
    ftmp_smc = ftmp['Analysis_Data/sm_surface_analysis']

    z_dim = len(filelist)
    col_dim = ftmp_smc.shape[1]
    row_dim = ftmp_smc.shape[0]

    #initiate output stack
    SMAPstack = h5py.File(root_path + '/SMAPL4SMC_2015.h5','w')

    smcstack = SMAPstack.create_dataset("SMCstack_2015", (z_dim, row_dim, col_dim), dtype="<f4")
    time = SMAPstack.create_dataset("time", (z_dim,), dtype="<f8")
    latarr = SMAPstack.create_dataset('cell_lat', data=ftmp['cell_lat'])
    lonarr = SMAPstack.create_dataset('cell_lon', data=ftmp['cell_lon'])

    ftmp.close()

    #iterate through all found files
    for find in range(len(filelist)):
        ftmp = h5py.File(filelist[find], 'r')
        smcstack[find,:,:] = ftmp['Analysis_Data/sm_surface_analysis']
        time[find] = ftmp['time'][0]
        ftmp.close()

    SMAPstack.close()


def dem2equi7(out_path=None, dem_path=None):

    # this routine reproject the DEM to the Equi7 grid and derives slope and aspect

    grid = Equi7Grid(10)
    grid.resample(dem_path,out_path, gdal_path="/usr/local/bin", sgrid_ids=['EU'], e7_folder=False,
                  outshortname="EDUDEM", withtilenameprefix=True, image_nodata=0, tile_nodata=-9999,
                  qlook_flag=False, resampling_type='bilinear')

    # get list of resampled DEM tiles
    filelist = [x for x in glob.glob(out_path + '*.tif')]

    # iterate through all files and derive slope and aspect
    for file in filelist:

        # aspect
        aspect_path = out_path + os.path.basename(file)[:-4] + '_aspect.tif'
        # slope
        slope_path = out_path + os.path.basename(file)[:-4] + '_slope.tif'

        os.system('/usr/local/bin/gdaldem slope ' + file + ' ' + slope_path + ' -co "COMPRESS=LZW"')
        os.system('/usr/local/bin/gdaldem aspect ' + file + ' ' + aspect_path + ' -co "COMPRESS=LZW"')


class Trainingset(object):


    def __init__(self, sgrt_root, sig0mpath, smcpath, dempath, outpath):

        self.sgrt_root = sgrt_root
        self.sig0mpath = sig0mpath
        self.smcpath = smcpath
        self.dempath = dempath
        self.outpath = outpath

        #
        # Get processing extent
        #
        # get list of available parameter tiles
        tiles = os.listdir(sig0mpath)
        extents = []
        for tname in tiles:
            tmptile = Equi7Tile('EU010M_' + tname)
            extents.append(tmptile.extent)
            tmptile = None

        extents = np.array(extents)
        if len(tiles) > 1:
            aoi = [np.min(extents[:,0]), np.min(extents[:,1]), np.max(extents[:,2]), np.max(extents[:,3])]
        else:
            aoi = extents[0]

        self.points = self.create_random_points(aoi=aoi)

        sig0lia = self.extr_sig0_lia()
        pickle.dump(sig0lia, open(self.outpath + 'sig0lia_dict.p', 'wb'))
        # sig0lia = pickle.load(open(self.outpath + 'sig0lia_dict.p', 'rb'))

        # filter samples with nan values
        samples = np.array([sig0lia.get(x) for x in sig0lia.keys()])
        samples = samples.transpose()
        samples = samples[~np.isnan(samples).any(axis=1)]

        # get training and validation sets
        self.target = samples[:, 12]
        self.features = np.delete(samples, [6,7,12,13], axis=1)

        print 'HELLO'


    def train_model(self):

        # scaling
        scaler = sklearn.preprocessing.StandardScaler().fit(self.features)
        features = scaler.transform(self.features)

        # split into independent training data and test data
        x_train, x_test, y_train, y_test = train_test_split(features, self.target, test_size=0.3, train_size=0.7)

        # ...----...----...----...----...----...----...----...----...----...
        # SVR -- SVR -- SVR -- SVR -- SVR -- SVR -- SVR -- SVR -- SVR -- SVR
        # ---....---....---....---....---....---....---....---....---....---

        dictCV = dict(C =       np.logspace(-2,2,10),
                      gamma =   np.logspace(-2,1,10),
                      epsilon = np.logspace(-2, -0.5, 10))

        # specify kernel
        svr_rbf = SVR(kernel = 'rbf')

        # parameter tuning -> grid search
        start = time()
        print('Model training startet at: ' + str(start))
        gdCV = GridSearchCV(estimator=svr_rbf, param_grid=dictCV, n_jobs=2, verbose=1, pre_dispatch='all')
        gdCV.fit(x_train, y_train)
        print('Elapse time for training: ' + str(time() - start))
        pickle.dump(gdCV, open(self.outpath + 'mlmodel.p', 'wb'))

        # prediction
        y_CV_rbf = gdCV.predict(x_test)

        r = np.corrcoef(y_test, y_CV_rbf)
        error = np.sqrt(np.sum(np.square(y_test-y_CV_rbf)) / len(y_test))

        print('SVR performance based on test-set')
        print('R: ' + str(r[0,1]))
        print('RMSE. ' + str(error))

        # create plots
        plt.figure(figsize = (6,6))
        plt.scatter(y_test, y_CV_rbf, c='g', label='True vs Est')
        plt.xlim(0.1,0.5)
        plt.ylim(0.1,0.5)
        plt.plot([0.1,0.5],[0.1,0.5], 'k--')
        plt.savefig(self.outpath + 'truevsest.png')
        plt.close()

        return (gdCV, scaler)


    def get_terrain(self, x, y):
        # extract elevation, slope and aspect

        # set up grid to determine tile name
        Eq7 = Equi7Grid(10)

        # create ouput array
        topo = np.full((len(self.points), 3), -9999, dtype=np.float32)

        # get tile
        tilename = Eq7.identfy_tile('EU', (x, y))

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


    def extr_sig0_lia(self):

        import sgrt_devels.extr_TS as exTS
        import random
        import datetime as dt
        import os

        cntr = 0

        # cycle through all points
        for px in self.points:

            # create 100 random points to sample the 9 km smap pixel
            sig0points = set()
            while len(sig0points) <= 5:         # TODO: Increase number of subpoints
                tmpx = random.randint(px[0]-4500,px[0]+4500)
                tmpy = random.randint(px[1]-4500,px[1]+4500)

                # check land cover
                LCpx = self.get_lc(tmpx, tmpy)

                if LCpx in [10, 12, 13, 18, 26]:
                    sig0points.add((tmpx, tmpy))

            # cycle through the create points to retrieve a aerial mean value
            # dictionary to hold the time seres
            vvdict = {}
            vhdict = {}
            liadict = {}
            slopelistVV = []
            slopelistVH = []
            meanlistVV = []
            meanlistVH = []
            sdlistVV = []
            sdlistVH = []
            hlist = []
            slist = []
            alist = []

            # counter
            tsnum = 0
            for subpx in sig0points:
                # get slope
                slope = self.get_slope(subpx[0],subpx[1])
                slope = np.float32(slope)
                slope[slope != -9999] = slope[slope != -9999] / 100
                slopelistVV.append(slope[0])
                slopelistVH.append(slope[1])

                # get mean
                mean = self.get_sig0mean(subpx[0],subpx[1])
                mean = np.float32(mean)
                mean[mean != -9999] = mean[mean != -9999] / 100
                meanlistVV.append(mean[0])
                meanlistVH.append(mean[1])

                # get standard deviation
                sd = self.get_sig0sd(subpx[0],subpx[1])
                sd = np.float32(sd)
                sd[sd != -9999] = sd[sd != -9999] / 100
                sdlistVV.append(sd[0])
                sdlistVH.append(sd[1])

                # get height, aspect, and slope
                terr = self.get_terrain(subpx[0],subpx[1])
                hlist.append(terr[0])
                alist.append(terr[1])
                slist.append(terr[2])

                # get sig0 and lia timeseries
                tmp_series = exTS.extr_SIG0_LIA_ts(self.sgrt_root, 'S1AIWGRDH', 'A0110', 'resampled', 10, subpx[0], subpx[1], 1, 1, pol_name=['VV','VH'], grid='Equi7')
                sig0vv = np.float32(tmp_series[1]['sig0'])
                sig0vh = np.float32(tmp_series[1]['sig02'])
                lia = np.float32(tmp_series[1]['lia'])
                sig0vv[sig0vv != -9999] = sig0vv[sig0vv != -9999]/100
                sig0vh[sig0vh != -9999] = sig0vh[sig0vh != -9999]/100
                lia[lia != -9999] = lia[lia != -9999]/100

                # normalise backscatter
                if slope[0] != 0 and slope[0] != -9999:
                    sig0vv[sig0vv != -9999] = np.power(10,(sig0vv[sig0vv != -9999] - slope[0] * (lia[sig0vv != -9999] - 30))/10)
                else:
                    sig0vv[sig0vv != -9999] = np.power(10,sig0vv[sig0vv != -9999]/10)
                if slope[1] != 0 and slope[1] != -9999:
                    sig0vh[sig0vh != -9999] = np.power(10,(sig0vh[sig0vh != -9999] - slope[0] * (lia[sig0vh != -9999] - 30))/10)
                else:
                    sig0vh[sig0vh != -9999] = np.power(10,sig0vh[sig0vh != -9999]/10)

                datelist = tmp_series[0]
                # datelist = [dt.datetime.fromordinal(x) for x in tmp_series[0]]

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

            mask = (arr_vv == -9999) | (arr_vv < -20.00) | (arr_vh == -9999) | (arr_vh < -20.00) | \
                   (arr_lia < 10.00) | (arr_lia > 50.00)

            df_vv.iloc[mask] = np.nan
            df_vh.iloc[mask] = np.nan
            df_lia.iloc[mask] = np.nan

            # mask months
            monthmask = (df_vv.index.map(lambda x: x.month) > 4) & (df_vv.index.map(lambda x: x.month) < 11)

            # calculate spatial mean and standard deviation
            df_vv_mean = 10*np.log10(df_vv.mean(1)[monthmask])
            df_vh_mean = 10*np.log10(df_vh.mean(1)[monthmask])
            df_lia_mean = df_lia.mean(1)[monthmask]
            df_vv_sstd = 10*np.log10(df_vv.std(1)[monthmask])
            df_vh_sstd = 10*np.log10(df_vh.std(1)[monthmask])
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



            # tmp_vv = pd.DataFrame(df_vv_mean)
            # tmp_df_ssm = pd.DataFrame(tmp_ssm[1], index=tmp_ssm[0])
            # tmp_df_ssm.index = tmp_df_ssm.index.to_datetime().date
            # tmp_vv.index = tmp_vv.index.date
            # ssm_dict = {'ssm': df_ssm[0], 'vv': tmp_vv[0]}
            # df_ssm = pd.DataFrame(ssm_dict)
            # df_ssm = tmp_vv.join(tmp_df_ssm, how='left', lsuffix='_vv', rsuffix='_ssm')
            # df_ssm = df_ssm['0_ssm']
            tmp_ssm = None
            # tmp_vv = None
            # tmp_df_ssm = None

            # calculate mean temporal mean and standard deviation and slope
            meanMeanVV = np.mean(meanlistVV[meanlistVV != -9999])
            meanMeanVH = np.mean(meanlistVH[meanlistVH != -9999])
            meanSdVV = np.mean(sdlistVV[sdlistVV != -9999])
            meanSdVH = np.mean(sdlistVH[sdlistVH != -9999])
            meanSlopeVV = np.mean(slopelistVV[slopelistVV != -9999])
            meanSlopeVH = np.mean(slopelistVH[slopelistVH != -9999])
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
                                   'vv_tmean': [meanMeanVV]*ll,
                                   'vh_tmean': [meanMeanVH]*ll,
                                   'vv_tstd': [meanSdVV]*ll,
                                   'vh_tstd': [meanSdVH]*ll,
                                   'vv_slope': [meanSlopeVV]*ll,
                                   'vh_slope': [meanSlopeVH]*ll,
                                   'height': [meanH]*ll,
                                   'aspect': [meanA]*ll,
                                   'slope': [meanS]*ll}
            else:
                ll = len(list(np.array(df_bac['vv']).squeeze()))
                sig0lia_samples['ssm'].extend(list(np.array(ssm_series).squeeze()))
                sig0lia_samples['sig0vv'].extend(list(np.array(df_bac['vv']).squeeze()))
                sig0lia_samples['sig0vh'].extend(list(np.array(df_bac['vh']).squeeze()))
                sig0lia_samples['lia'].extend(list(np.array(df_bac['lia']).squeeze()))
                sig0lia_samples['vv_sstd'].extend(list(np.array(df_bac['vv_std']).squeeze()))
                sig0lia_samples['vh_sstd'].extend(list(np.array(df_bac['vh_std']).squeeze()))
                sig0lia_samples['lia_sstd'].extend(list(np.array(df_bac['lia_std']).squeeze()))
                sig0lia_samples['vv_tmean'].extend([meanMeanVV]*ll)
                sig0lia_samples['vh_tmean'].extend([meanMeanVH]*ll)
                sig0lia_samples['vv_tstd'].extend([meanSdVV]*ll)
                sig0lia_samples['vh_tstd'].extend([meanSdVH]*ll)
                sig0lia_samples['vv_slope'].extend([meanSlopeVV]*ll)
                sig0lia_samples['vh_slope'].extend([meanSlopeVH]*ll)
                sig0lia_samples['height'].extend([meanH]*ll)
                sig0lia_samples['aspect'].extend([meanA]*ll)
                sig0lia_samples['slope'].extend([meanS]*ll)

            cntr = cntr + 1
            os.system('rm /tmp/*.vrt')

        return sig0lia_samples


    def create_random_points(self, aoi=None):

        import random

        # set up land cover grid
        Eq7LC = Equi7Grid(75)

        # create list of 1000 random points
        points = set()

        while len(points) <= 500:
            tmpx = random.randint(aoi[0]+4500, aoi[2]-4500)
            tmpy = random.randint(aoi[1]+4500, aoi[3]-4500)

            # get land cover
            LCpx = self.get_lc(tmpx, tmpy)

            # check if open land
            if LCpx in [10, 12, 13, 18, 26]:
                points.add((tmpx,tmpy))

        return(points)


    def get_lc(self, x, y):

        # set up land cover grid
        Eq7LC = Equi7Grid(75)

        # get tile name of 75 Equi7 grid to check land-cover
        tilename = Eq7LC.identfy_tile('EU', (x, y))
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
        LCpx = LCband.ReadAsArray(int((x-LCtile.geotags['geotransform'][0])/75), int((LCtile.geotags['geotransform'][3]-y)/75), 1, 1)

        return LCpx[0][0]


    def get_slope(self, x, y):

        # set up parameter grid
        Eq7Par = Equi7Grid(10)

        # get tile name of Equi7 10 grid
        tilename = Eq7Par.identfy_tile('EU', (x,y))
        Stile = SgrtTile(dir_root=self.sgrt_root,
                         product_id='S1AIWGRDH',
                         soft_id='B0210',
                         product_name='sig0m',
                         ftile=tilename,
                         src_res=10)

        SfilenameVV = [xs for xs in Stile._tile_files if 'SIGSL' in xs and 'VV' in xs]
        SfilenameVH = [xs for xs in Stile._tile_files if 'SIGSL' in xs and 'VH' in xs]
        SfilenameVV = Stile.dir + '/' + SfilenameVV[0] + '.tif'
        SfilenameVH = Stile.dir + '/' + SfilenameVH[0] + '.tif'

        SVV = gdal.Open(SfilenameVV, gdal.GA_ReadOnly)
        SVH = gdal.Open(SfilenameVH, gdal.GA_ReadOnly)
        SVVband = SVV.GetRasterBand(1)
        SVHband = SVH.GetRasterBand(1)
        slopeVV = SVVband.ReadAsArray(int((x-Stile.geotags['geotransform'][0])/10), int((Stile.geotags['geotransform'][3]-y)/10), 1, 1)[0][0]
        slopeVH = SVHband.ReadAsArray(int((x-Stile.geotags['geotransform'][0])/10), int((Stile.geotags['geotransform'][3]-y)/10), 1, 1)[0][0]

        return (slopeVV, slopeVH)


    def get_sig0mean(self, x, y):

        # set up parameter grid
        Eq7Par = Equi7Grid(10)

        # get tile name of Equi7 10 grid
        tilename = Eq7Par.identfy_tile('EU', (x,y))
        Stile = SgrtTile(dir_root=self.sgrt_root,
                         product_id='S1AIWGRDH',
                         soft_id='B0210',
                         product_name='sig0m',
                         ftile=tilename,
                         src_res=10)

        SfilenameVV = [xs for xs in Stile._tile_files if 'SIG0M' in xs and 'VV' in xs]
        SfilenameVH = [xs for xs in Stile._tile_files if 'SIG0M' in xs and 'VH' in xs]
        SfilenameVV = Stile.dir + '/' + SfilenameVV[0] + '.tif'
        SfilenameVH = Stile.dir + '/' + SfilenameVH[0] + '.tif'

        SVV = gdal.Open(SfilenameVV, gdal.GA_ReadOnly)
        SVH = gdal.Open(SfilenameVH, gdal.GA_ReadOnly)
        SVVband = SVV.GetRasterBand(1)
        SVHband = SVH.GetRasterBand(1)
        meanVV = SVVband.ReadAsArray(int((x-Stile.geotags['geotransform'][0])/10), int((Stile.geotags['geotransform'][3]-y)/10), 1, 1)[0][0]
        meanVH = SVHband.ReadAsArray(int((x-Stile.geotags['geotransform'][0])/10), int((Stile.geotags['geotransform'][3]-y)/10), 1, 1)[0][0]

        return (meanVV, meanVH)


    def get_sig0sd(self, x, y):

        # set up parameter grid
        Eq7Par = Equi7Grid(10)

        # get tile name of Equi7 10 grid
        tilename = Eq7Par.identfy_tile('EU', (x,y))
        Stile = SgrtTile(dir_root=self.sgrt_root,
                         product_id='S1AIWGRDH',
                         soft_id='B0210',
                         product_name='sig0m',
                         ftile=tilename,
                         src_res=10)

        SfilenameVV = [xs for xs in Stile._tile_files if 'SIGSD' in xs and 'VV' in xs]
        SfilenameVH = [xs for xs in Stile._tile_files if 'SIGSD' in xs and 'VH' in xs]
        SfilenameVV = Stile.dir + '/' + SfilenameVV[0] + '.tif'
        SfilenameVH = Stile.dir + '/' + SfilenameVH[0] + '.tif'

        SVV = gdal.Open(SfilenameVV, gdal.GA_ReadOnly)
        SVH = gdal.Open(SfilenameVH, gdal.GA_ReadOnly)
        SVVband = SVV.GetRasterBand(1)
        SVHband = SVH.GetRasterBand(1)
        sdVV = SVVband.ReadAsArray(int((x-Stile.geotags['geotransform'][0])/10), int((Stile.geotags['geotransform'][3]-y)/10), 1, 1)[0][0]
        sdVH = SVHband.ReadAsArray(int((x-Stile.geotags['geotransform'][0])/10), int((Stile.geotags['geotransform'][3]-y)/10), 1, 1)[0][0]

        return (sdVV, sdVH)


    def get_ssm(self, x, y):
        import math
        import datetime as dt

        grid = Equi7Grid(10)
        poi_lonlat = grid.equi7xy2lonlat('EU', x, y)

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
        timelist = [dt.date.fromtimestamp(x+946727936) for x in time]
        #dt.datetime.fromtimestamp()
        ssm_stack.close()

        return (timelist, ssm)


class Estimationset(object):


    def __init__(self, sgrt_root, sig0mpath, dempath, outpath, mlmodel, date):

        self.sgrt_root = sgrt_root
        self.outpath = outpath
        self.sig0mpath = sig0mpath
        self.dempath = dempath

        # get list of available parameter tiles
        # tiles = os.listdir(sig0mpath)
        tiles = 'E048N015T1'

        for tname in tiles:

            # get sig0 image to derive ssm
            bacArrs = self.get_sig0_lia(tname, date)
            terrainArrs = self.get_terrain(tname)
            paramArr = self.get_params(tname)

            # create masks
            lc_mask = self.create_LC_mask(tname, bacArrs)
            sig0_mask = (bacArrs['sig0vv'][0] != -9999) & \
                        (bacArrs['sig0vv'][0] >= -2000) & \
                        (bacArrs['sig0vh'][0] != -9999) & \
                        (bacArrs['sig0vh'][0] >= -2000) & \
                        (bacArrs['lia'][0] >= 1000) & \
                        (bacArrs['lia'][0] <= 5000)
            # combined mask
            mask = lc_mask & sig0_mask

            valid_ind = np.where(mask == True)
            valid_ind = np.ravel_multi_index(valid_ind, (10000,10000))
            ssm_out = np.full((10000,10000), -9999, dtype=np.float32)

            for i in valid_ind:
                ind = np.unravel_index(i, (10000,10000))
                # compile feature vector
                sig0vv = bacArrs['sig0vv'][0][ind] / 100.0
                sig0vh = bacArrs['sig0vh'][0][ind] / 100.0
                lia = bacArrs['lia'][0][ind] / 100.0
                h = terrainArrs['h'][0][ind]
                a = terrainArrs['a'][0][ind]
                s = terrainArrs['s'][0][ind]
                slpvv = paramArr['slpVV'][0][ind] / 100.0
                slpvh = paramArr['slpVH'][0][ind] / 100.0
                sig0mvv = paramArr['sig0mVV'][0][ind] / 100.0
                sig0mvh = paramArr['sig0mVH'][0][ind] / 100.0
                sig0sdvv = paramArr['sig0sdVV'][0][ind] / 100.0
                sig0sdvh = paramArr['sig0sdVH'][0][ind] / 100.0

                # normalize sig0
                if slpvv != 0: sig0vv = sig0vv - slpvv*(lia-30)
                if slpvh != 0: sig0vh = sig0vh - slpvh*(lia-30)

                fvect = [s, sig0sdvh, sig0sdvv, sig0mvh, lia, sig0mvv, sig0vv, slpvv, slpvh, h, sig0vh, a]
                fvect = mlmodel[1].transform(fvect)
                # predict ssm
                predssm = mlmodel[0].predict(fvect)
                ssm_out[ind] = predssm

            # write ssm out
            self.write_ssm(tname, bacArrs, ssm_out)



            print 'HELLO'


    def get_sig0_lia(self, tname, date):

        # read sig0 vv/vh and lia in arrays
        tile = SgrtTile(dir_root=self.sgrt_root,
                        product_id='S1AIWGRDH',
                        soft_id='A0110',
                        product_name='resampled',
                        ftile='EU010M_'+tname,
                        src_res=10)

        sig0vv = tile.read_tile(pattern=date+'.*VV.*')
        sig0vh = tile.read_tile(pattern=date+'.*VH.*')
        lia = tile.read_tile(pattern=date+'.*PLIA.*')

        return {'sig0vv': sig0vv, 'sig0vh': sig0vh, 'lia': lia}


    def get_terrain(self, tname):

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


    def get_params(self, tname):

        # get slope and sig0 mean and standard deviation
        Stile = SgrtTile(dir_root=self.sgrt_root,
                         product_id='S1AIWGRDH',
                         soft_id='B0210',
                         product_name='sig0m',
                         ftile='EU010M_' + tname,
                         src_res=10)

        slpVV = Stile.read_tile(pattern='.*SIGSL.*VV.*')
        slpVH = Stile.read_tile(pattern='.*SIGSL.*VH.*')
        sig0mVV = Stile.read_tile(pattern='.*SIG0M.*VV.*')
        sig0mVH = Stile.read_tile(pattern='.*SIG0M.*VH.*')
        sig0sdVV = Stile.read_tile(pattern='.*SIGSD.*VV.*')
        sig0sdVH = Stile.read_tile(pattern='.*SIGSD.*VH.*')

        return {'slpVV': slpVV,
                'slpVH': slpVH,
                'sig0mVV': sig0mVV,
                'sig0mVH': sig0mVH,
                'sig0sdVV': sig0sdVV,
                'sig0sdVH': sig0sdVH}


    def create_LC_mask(self, tname, bacArrs):

        # get tile name in 75m lc grid
        eq7tile = Equi7Tile('EU010M_' + tname)
        tname75 = eq7tile.find_family_tiles(res=75)
        # load lc array, resampled to 10m
        lcArr = self.get_lc(tname75[0], bacArrs)

        #generate mask
        tmp = np.array(lcArr[0])
        mask = (tmp == 10) | (tmp == 12) | (tmp == 13) | (tmp == 18) | (tmp == 26)

        return mask


    def get_lc(self, tname, bacArrs):

        # get tile name of 75 Equi7 grid to check land-cover
        LCtile = SgrtTile(dir_root=self.sgrt_root,
                          product_id='S1AIWGRDH',
                          soft_id='E0110',
                          product_name='CORINE06',
                          ftile='EU075M_'+tname,
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


    def write_ssm(self, tname, bacArrs, outarr):

        # write ssm map
        dst_proj = bacArrs['sig0vv'][1]['spatialreference']
        dst_geotrans = bacArrs['sig0vv'][1]['geotransform']
        dst_width = 10000
        dst_height = 10000

        # set up output file
        ssm_path = self.outpath + tname+'_SSM.tif'
        ssm_map = gdal.GetDriverByName('GTiff').Create(ssm_path, dst_width, dst_height, 1, gdalconst.GDT_Float32)
        ssm_map.SetGeoTransform(dst_geotrans)
        ssm_map.SetProjection(dst_proj)

        # write data
        ssm_outband = ssm_map.GetRasterBand(1)
        ssm_outband.WriteArray(outarr)
        ssm_outband.FlushCache()
        ssm_outband.SetNoDataValue(-9999)

        del ssm_map


