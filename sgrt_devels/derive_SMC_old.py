__author__ = 'usergre'

import devel_fgreifen2.extr_TS
import numpy as np
from osgeo import gdal, gdalconst
import os
from progressbar import *
from sklearn import ensemble
import cPickle
#import os.path


def create_trafo_string(minX, minY, maxX, maxY, pxsize, no_data):
    trafo_string = ('/usr/local/bin/gdalwarp -te ' +
              str(minX) + ' ' + str(minY) + ' ' + str(maxX) + ' ' +
              str(maxY) + ' -tr ' + str(pxsize) + ' ' +
              str(pxsize) + ' -tap -r near -srcnodata ' + str(no_data) + ' -dstnodata ' + str(no_data) +
              ' -of Gtiff ')

    return(trafo_string)


def smc_map(sig0file, liafile, demfile, ndvifile, lcfile, modelfile, workpath, outpath):

    slopefile = demfile[:-4] + '_slope.tif'
    aspectfile = demfile[:-4] + '_aspect.tif'

    # copy data to work-directory

    os.system('cp ' + sig0file + ' ' + workpath)
    os.system('cp ' + liafile + ' ' + workpath)

    sig0file = workpath + '/' + os.path.basename(sig0file)
    liafile = workpath + '/' + os.path.basename(liafile)

    #parameters = np.array([1,2,3,4,5,6], dtype=np.float32)

    # load sig0 and lia-----------------------------------------------------------------
    ds_sig0 = gdal.Open(sig0file)
    ds_lia = gdal.Open(liafile)
    geotrans = ds_sig0.GetGeoTransform()

    originX = geotrans[0]
    originY = geotrans[3]
    pixelSize = geotrans[1]
    sig0proj = ds_sig0.GetProjection()

    sig0arr = np.array(ds_sig0.GetRasterBand(1).ReadAsArray())
    liaarr = np.array(ds_lia.GetRasterBand(1).ReadAsArray())

    # ---------------------------------------------------------------------------------
    # resample dem and lc -------------------------------------------------------------
    # resample height, slope, aspect, and land-cover to allign with sig0
    # elevation
    no_data = 32767
    ds_h_orig = gdal.Open(demfile)
    h_proj = ds_h_orig.GetProjection()
    ds_h = gdal.GetDriverByName('GTiff').Create(workpath + '/' + os.path.basename(demfile),
                                                sig0arr.shape[1],
                                                sig0arr.shape[0],
                                                1,
                                                gdalconst.GDT_Float32)
    ds_h.SetGeoTransform(geotrans)
    ds_h.SetProjection(sig0proj)
    gdal.ReprojectImage(ds_h_orig, ds_h, h_proj, sig0proj, gdalconst.GRA_NearestNeighbour)
    demfile = workpath + '/' + os.path.basename(demfile)
    del ds_h_orig

    # slope
    no_data = -9999
    ds_s_orig = gdal.Open(slopefile)
    s_proj = ds_s_orig.GetProjection()
    ds_s = gdal.GetDriverByName('GTiff').Create(workpath + '/' + os.path.basename(slopefile),
                                                sig0arr.shape[1],
                                                sig0arr.shape[0],
                                                1,
                                                gdalconst.GDT_Float32)
    ds_s.SetGeoTransform(geotrans)
    ds_s.SetProjection(sig0proj)
    gdal.ReprojectImage(ds_s_orig, ds_s, s_proj, sig0proj, gdalconst.GRA_NearestNeighbour)
    slopefile = workpath + '/' + os.path.basename(slopefile)
    del ds_s_orig

    # aspect
    ds_a_orig = gdal.Open(aspectfile)
    a_proj = ds_a_orig.GetProjection()
    ds_a = gdal.GetDriverByName('GTiff').Create(workpath + '/' + os.path.basename(aspectfile),
                                                sig0arr.shape[1],
                                                sig0arr.shape[0],
                                                1,
                                                gdalconst.GDT_Float32)
    ds_a.SetGeoTransform(geotrans)
    ds_a.SetProjection(sig0proj)
    gdal.ReprojectImage(ds_a_orig, ds_a, a_proj, sig0proj, gdalconst.GRA_NearestNeighbour)
    aspectfile = workpath + '/' + os.path.basename(aspectfile)
    del ds_a_orig

    # land cover
    no_data = 0
    ds_lc_orig = gdal.Open(lcfile)
    lc_proj = ds_lc_orig.GetProjection()
    ds_lc = gdal.GetDriverByName('GTiff').Create(workpath + '/' + os.path.basename(lcfile),
                                                sig0arr.shape[1],
                                                sig0arr.shape[0],
                                                1,
                                                gdalconst.GDT_Byte)
    ds_lc.SetGeoTransform(geotrans)
    ds_lc.SetProjection(sig0proj)
    gdal.ReprojectImage(ds_lc_orig, ds_lc, lc_proj, sig0proj, gdalconst.GRA_NearestNeighbour)
    lcfile = workpath + '/' + os.path.basename(lcfile)
    del ds_lc_orig

    # ndvi
    no_data = -1
    ds_ndvi_orig = gdal.Open(ndvifile)
    ndvi_proj = ds_ndvi_orig.GetProjection()
    ds_ndvi_orig.GetRasterBand(1).SetNoDataValue(no_data)
    ds_ndvi = gdal.GetDriverByName('GTiff').Create(workpath + '/' + os.path.basename(ndvifile),
                                                   sig0arr.shape[1],
                                                   sig0arr.shape[0],
                                                   1,
                                                   gdalconst.GDT_Float32)
    ds_ndvi.SetGeoTransform(geotrans)
    ds_ndvi.SetProjection(sig0proj)
    gdal.ReprojectImage(ds_ndvi_orig, ds_ndvi, ndvi_proj, sig0proj, gdalconst.GRA_Bilinear)
    ndvifile = workpath + '/' + os.path.basename(ndvifile)
    del ds_ndvi_orig

    #mean sig
#    no_data = 0
#    ds_msig_orig = gdal.Open(meansigfile)
#    msig_proj = ds_msig_orig.GetProjection()
#    ds_msig = gdal.GetDriverByName('GTiff').Create(workpath + '/' + os.path.basename(meansigfile),
#                                                sig0arr.shape[1],
#                                                sig0arr.shape[0],
#                                                1,
#                                                gdalconst.GDT_Byte)
#    ds_msig.SetGeoTransform(geotrans)
#    ds_msig.SetProjection(sig0proj)
#    gdal.ReprojectImage(ds_msig_orig, ds_msig, msig_proj, sig0proj, gdalconst.GRA_NearestNeighbour)
#    meansigfile = workpath + '/' + os.path.basename(meansigfile)
#    del ds_msig_orig

    # -----------------------------------------------------------------------------------
    # load all data to numpy arrays

    harr = np.array(ds_h.GetRasterBand(1).ReadAsArray())
    sarr = np.array(ds_s.GetRasterBand(1).ReadAsArray())
    aarr = np.array(ds_a.GetRasterBand(1).ReadAsArray())
    lcarr = np.array(ds_lc.GetRasterBand(1).ReadAsArray())
    ndviarr = np.array(ds_ndvi.GetRasterBand(1).ReadAsArray())
#    msigarr = np.array(ds_msig.GetRasterBand(1).ReadAsArrqay())

    # ------------------------------------------------------------------------------------
    # derive SMC

    # create mask
    goodlc = np.array([10,12,13,15,16,17,18,19,20,21,26,27,28,29])
    valpx = np.where((sig0arr != -9999) & (sig0arr < 0) &# (msigarr != -9999) &
                     (liaarr != -9999) & (liaarr > 2000) & (liaarr < 6000) &
                     (ndviarr != -1) & (ndviarr != 0) &
                     (harr != 0) & (harr != 32767) &
                     (sarr != 0) & (sarr != -9999) &
                     (aarr != 0) & (aarr != -9999) &
                     (lcarr != 0) & (np.reshape(np.in1d(lcarr, goodlc), lcarr.shape)))
    pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=len(valpx[0])).start()

    #load machine learning model
    with open(modelfile, 'rb') as fid:
        model = cPickle.load(fid)

    # cycle through valid values
    cntr = 0
    smcarr = np.array(np.copy(sig0arr), dtype=np.float32)
    smcarr[:,:] = -9999

    for ind in range(len(valpx[0])):
        #xy = np.unravel_index([ind], sig0arr.shape)
        #iy = xy[0]
        #ix = xy[1]
        iy = valpx[0][ind]
        ix = valpx[1][ind]

        sig0 = sig0arr[iy,ix]
        lia = liaarr[iy,ix]
        ndvi = ndviarr[iy,ix]
        h = harr[iy,ix]
        s = sarr[iy,ix]
        a = aarr[iy,ix]
        lc = lcarr[iy,ix]
#        msig = msigarr[iy, ix]

        fline = np.array([sig0, lia, h, s, a])
        smc = model.predict(fline)
        smcarr[iy, ix] = smc

        #check data
        # use_sample = True
        # if sig0 > 0: use_sample = False
        # if lia < 2000 or lia > 6000: use_sample = False
        # if h == -9999 or h == 32767: use_sample = False
        # if s == -9999 or h == 32767: use_sample = False
        # if a == -9999 or a == 32767: use_sample = False
        # if np.any(goodlc == lc) == False: use_sample = False

        # if use_sample == True:
        #     param_line = np.array([sig0, lia, h, s, a, lc], dtype=np.float32)
        #     if cntr == 0:
        #         parameters = param_line
        #     else:
        #         parameters = np.vstack((parameters, param_line))
        #     cntr = cntr + 1
        # print(ind)
        pbar.update(ind+1)

    pbar.finish()

    x_pixel = smcarr.shape[1]
    y_pixel = smcarr.shape[0]

    driver = gdal.GetDriverByName('GTiff')

    smc_ds = driver.Create(outpath + '/SMCmap.tif',
                           x_pixel,
                           y_pixel,
                           1,
                           gdal.GDT_Float32)

    smc_ds.SetGeoTransform(geotrans)
    smc_ds.SetProjection(sig0proj)
    smc_ds.GetRasterBand(1).WriteArray(smcarr)
    smc_ds.FlushCache()

    os.system('rm ' + workpath + '/*.tif')

    # np.savetxt(outpath + '/MapParams.csv', parameters, delimiter=',')
