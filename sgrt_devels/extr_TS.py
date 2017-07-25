__author__ = 'felix'

import numpy as np
import sgrt.common.grids.Equi7Grid as Equi7
import sgrt.common.utils.SgrtTile as SgrtTile
from osgeo import gdal
from osgeo.gdalconst import *
from netCDF4 import Dataset, num2date
from datetime import datetime
import ee
import datetime as dt
import time
from sgrt_tools.access_google_drive import gdrive

# extract ERA-Land/Interim soil moisture at position lat lon
def extr_ERA_SMC(path, lon, lat):
    # load dataset
    eraFile = Dataset(path, 'r')

    eraLon = np.array(eraFile.variables['longitude'])
    eraLat = np.array(eraFile.variables['latitude'])
    eratmp = eraFile.variables['time']
    eratmp =  num2date(eratmp[:], units=eratmp.units, calendar=eratmp.calendar)
    eraTime = np.zeros(len(eratmp))
    for i in range(len(eratmp)): eraTime[i] = eratmp[i].toordinal()
    #eraTime = np.array(eraFile.variables['time'])
    #eraSMC = np.array(eraFile.variables['swvl1'])

    dist = np.ones([len(eraLat), len(eraLon)])
    dist[:,:] = 9999

    # calculate distance between each grid point and lon, lat
    for i in range(len(eraLon)):
        for j in range(len(eraLat)):
            dist[j,i] = np.sqrt(np.square(eraLon[i]-lon) + np.square(eraLat[j]-lat))

    # # get the nearest pixel
    # nearest = np.unravel_index(dist.argmin(), dist.shape)
    #
    # SMCts = np.zeros([len(eraTime), 2])
    # SMCts[:,1] = np.array(eraFile.variables['swvl1'][:, nearest[0], nearest[1]])
    # for i in range(len(eraTime)): SMCts[i,0] = eraTime[i]


    # get the four nearest pixels
    fourNearest = []
    weights = []
    for i in range(4):
        tmp = np.unravel_index(dist.argmin(), dist.shape)
        fourNearest.append([tmp[0], tmp[1]])
        weights.append(dist[tmp[0], tmp[1]])
        dist[tmp[0], tmp[1]] = 9999

    weights = np.array(weights)
    weights = weights.max() - weights

    # retrieve SMC
    SMCtsw = np.zeros([len(eraTime), 2])
    SMCtsw[1,:] = -9999
    SMCtsp = np.zeros([len(eraTime), 5])
    SMCtsp[1::,:] = -9999
    #load smc data
    for i in range(4):
        SMCtsp[:,i+1] = np.array(eraFile.variables['swvl1'][:,fourNearest[i][0], fourNearest[i][1]])

    # compute weighted average
    for i in range(len(eraTime)):
        SMCtsw[i,1] = np.sum(SMCtsp[i,1::]*weights) / weights.sum()
        SMCtsw[i,0] = eraTime[i]
        #SMCts[1,i] = SMCw

    eraFile.close()
    return(SMCtsw)


def multitemporalDespeckle(images, radius, units, opt_timeWindow=None):

    def mapMeanSpace(i):
        reducer = ee.Reducer.mean()
        kernel = ee.Kernel.square(radius, units)
        mean = i.reduceNeighborhood(reducer, kernel).rename(bandNamesMean)
        ratio = i.divide(mean).rename(bandNamesRatio)
        return(i.addBands(mean).addBands(ratio))

    if opt_timeWindow == None:
        timeWindow = dict(before=-3, after=3, units='month')
    else:
        timeWindow = opt_timeWindow

    bandNames = ee.Image(images.first()).bandNames()
    bandNamesMean = bandNames.map(lambda b: ee.String(b).cat('_mean'))
    bandNamesRatio = bandNames.map(lambda b: ee.String(b).cat('_ratio'))

    # compute spatial average for all images
    meanSpace = images.map(mapMeanSpace)

    # computes a multi-temporal despeckle function for a single image

    def multitemporalDespeckleSingle(image):
        t = image.date()
        fro = t.advance(ee.Number(timeWindow['before']), timeWindow['units'])
        to = t.advance(ee.Number(timeWindow['after']), timeWindow['units'])

        meanSpace2 = ee.ImageCollection(meanSpace).select(bandNamesRatio).filterDate(fro, to) \
                .filter(ee.Filter.eq('relativeOrbitNumber_start', image.get('relativeOrbitNumber_start')))

        b = image.select(bandNamesMean)

        return(b.multiply(meanSpace2.sum()).divide(meanSpace2.count()).rename(bandNames))

    return meanSpace.map(multitemporalDespeckleSingle)


def extr_MODIS_MOD13Q1_ts_GEE(lon,lat, bufferSize=20):

    ee.Reset()
    ee.Initialize()

    def createAvg(image):
        gee_roi = ee.Geometry.Point(lon, lat).buffer(bufferSize)

        # mask image
        immask = image.select('SummaryQA').eq(ee.Image(0))
        image = image.updateMask(immask)

        reduced_img_data = image.reduceRegion(ee.Reducer.median(), gee_roi, 30)
        return ee.Feature(None, {'result': reduced_img_data})

    # load collection
    gee_l8_collection = ee.ImageCollection('MODIS/006/MOD13Q1')

    # filter collection
    gee_roi = ee.Geometry.Point(lon, lat).buffer(bufferSize)
    gee_l8_fltd = gee_l8_collection.filterBounds(gee_roi)

    # extract time series
    gee_l8_mpd = gee_l8_fltd.map(createAvg)
    tmp = gee_l8_mpd.getInfo()

    EVI = np.array([x['properties']['result']['EVI'] for x in tmp['features']], dtype=np.float)

    ge_dates = np.array([datetime.strptime(x['id'][12::], '%Y_%m_%d') for x in tmp['features']])

    valid = np.where(np.isfinite(EVI))

    # cut out invalid values
    EVI = EVI[valid]
    ge_dates = ge_dates[valid]

    return ((ge_dates, {'EVI': EVI}))


def extr_MODIS_MOD13Q1_array_reGE(minlon, minlat, maxlon, maxlat, year, month, day):

    ee.Initialize()

    def mask(image):
        # mask image
        immask = image.select('SummaryQA').eq(ee.Image(0))
        image = image.updateMask(immask)
        return (image)

    def mosaic_custom(image, mosaic):
        tmpmosaic = ee.Image(mosaic)
        tmpimage = ee.Image(image)
        return(tmpmosaic.where(tmpimage.select('SummaryQA').eq(0), tmpimage))

    # load collection
    modis_collection = ee.ImageCollection('MODIS/006/MOD13Q1')

    # construct roi
    roi = ee.Geometry.Polygon([[minlon, minlat], [minlon, maxlat],
                               [maxlon, maxlat], [maxlon, minlat],
                               [minlon, minlat]])
    # filter
    doi = dt.date(year=year, month=month, day=day)
    sdate = doi - dt.timedelta(days=100)
    edate = doi + dt.timedelta(days=100)

    modis_fltd = modis_collection.filterBounds(roi).filterDate(sdate.strftime('%Y-%m-%d'), edate.strftime('%Y-%m-%d'))
    # modis_fltd_before = modis_collection.filterBounds(roi).filterDate(sdate.strftime('%Y-%m-%d'), doi.strftime('%Y-%m-%d'))
    # modis_fltd_after = modis_collection.filterBounds(roi).filterDate(doi.strftime('%Y-%m-%d'), edate.strftime('%Y-%m-%d'))
    #
    # # get EVI composites before and after the day-of-interest
    # modis_img_before = ee.Image(modis_fltd_before.iterate(mosaic_custom, ee.Image(modis_fltd_before.first())))
    # tmp = modis_fltd_after.toList(500).reverse()
    # if tmp.length().getInfo() == 0:
    #     raise Exception('Empty list!')
    # modis_img_after = ee.Image(tmp.iterate(mosaic_custom, ee.Image(tmp.get(0))))
    #
    # # interpolate EVI for the day-of-interest
    # doy_aft = modis_img_after.select('DayOfYear').where(modis_img_after.select('DayOfYear').lt(modis_img_before.select('DayOfYear')),
    #                                                     modis_img_after.select('DayOfYear').add(ee.Image(365)))
    # doy_bef = modis_img_before.select('DayOfYear').where(modis_img_before.select('DayOfYear').lt(doi.timetuple().tm_yday),
    #                                                      modis_img_before.select('DayOfYear').add(ee.Image(365)))
    #
    # doy_diff = doy_aft.subtract(doy_bef)
    # evi_diff = modis_img_after.select('EVI').subtract(modis_img_before.select('EVI'))
    # coeff = evi_diff.divide(doy_diff)
    #
    # mask_combo = modis_img_after.select('SummaryQA').eq(0).And(modis_img_before.select('SummaryQA').eq(0))
    # if doi.year != sdate.year:
    #     doi_nmbr = doi.timetuple().tm_yday + 365
    # else:
    #     doi_nmbr = doi.timetuple().tm_yday
    # doi_img = ee.Image(doi_nmbr).subtract(doy_bef)
    #
    # evi_interpolated = modis_img_before.select('EVI').add(doi_img.multiply(coeff)).updateMask(mask_combo).clip(roi)

    # create a list of availalbel dates
    tmp = modis_fltd.getInfo()
    tmp_ids = [x['properties']['system:index'] for x in tmp['features']]
    dates = np.array([dt.datetime.strptime(x, '%Y_%m_%d').date() for x in tmp_ids])

    # find the closest acquisitions
    doi = dt.date(year=year, month=month, day=day)
    doi_index = np.argmin(np.abs(dates - doi))
    date_selected = dates[doi_index]

    # filter collection for respective dates
    edate = date_selected + dt.timedelta(days=1)
    modis_fltd = modis_fltd.filterDate(date_selected.strftime('%Y-%m-%d'), edate.strftime('%Y-%m-%d'))

    # mosaic scenes
    modis_fltd = modis_fltd.map(mask)
    modis_mosaic = ee.Image(modis_fltd.mosaic().clip(roi))

    # return (evi_interpolated, doi)
    return (modis_mosaic, date_selected)


def extr_L8_ts_GEE(lon, lat, bufferSize=20):

    ee.Reset()
    ee.Initialize()

    def createAvg(image):
        gee_roi = ee.Geometry.Point(lon, lat).buffer(bufferSize)

        # mask image
        immask = image.select('cfmask').eq(ee.Image(0))
        image = image.updateMask(immask)

        reduced_img_data = image.reduceRegion(ee.Reducer.median(), gee_roi, 30)
        return ee.Feature(None, {'result': reduced_img_data})

    def setresample(image):
        image = image.resample()
        return (image)


    # load collection
    gee_l8_collection = ee.ImageCollection('LANDSAT/LC8_SR').map(setresample)

    # filter collection
    gee_roi = ee.Geometry.Point(lon, lat).buffer(bufferSize)
    gee_l8_fltd = gee_l8_collection.filterBounds(gee_roi)

    # extract time series
    gee_l8_mpd = gee_l8_fltd.map(createAvg)
    tmp = gee_l8_mpd.getInfo()

    b1 = np.array([x['properties']['result']['B1'] for x in tmp['features']], dtype=np.float)
    b2 = np.array([x['properties']['result']['B2'] for x in tmp['features']], dtype=np.float)
    b3 = np.array([x['properties']['result']['B3'] for x in tmp['features']], dtype=np.float)
    b4 = np.array([x['properties']['result']['B4'] for x in tmp['features']], dtype=np.float)
    b5 = np.array([x['properties']['result']['B5'] for x in tmp['features']], dtype=np.float)
    b6 = np.array([x['properties']['result']['B6'] for x in tmp['features']], dtype=np.float)
    b7 = np.array([x['properties']['result']['B7'] for x in tmp['features']], dtype=np.float)

    ge_dates = np.array([datetime.strptime(x['id'][9::], '%Y%j') for x in tmp['features']])

    valid = np.where(np.isfinite(b2))

    # cut out invalid values
    b1 = b1[valid]
    b2 = b2[valid]
    b3 = b3[valid]
    b4 = b4[valid]
    b5 = b5[valid]
    b6 = b6[valid]
    b7 = b7[valid]
    ge_dates = ge_dates[valid]

    return((ge_dates, {'B1': b1,
                       'B2': b2,
                       'B3': b3,
                       'B4': b4,
                       'B5': b5,
                       'B6': b6,
                       'B7': b7}))


def extr_L8_array_reGE(minlon, minlat, maxlon, maxlat, year, month, day):

    ee.Initialize()

    def setresample(image):
        image = image.resample()
        return (image)

    def mask(image):
        # mask image
        immask = image.select('cfmask').eq(ee.Image(0))
        image = image.updateMask(immask)
        return(image)


    # load collection
    gee_l8_collection = ee.ImageCollection('LANDSAT/LC8_SR').map(setresample)

    # construct roi
    roi = ee.Geometry.Polygon([[minlon, minlat], [minlon, maxlat],
                               [maxlon, maxlat], [maxlon, minlat],
                               [minlon, minlat]])

    # filter
    doi = dt.date(year=year, month=month, day=day)
    sdate = doi - dt.timedelta(days=30)
    edate = doi + dt.timedelta(days=30)
    gee_l8_fltd = gee_l8_collection.filterBounds(roi).filterDate(sdate.strftime('%Y-%m-%d'), edate.strftime('%Y-%m-%d'))

    # create a list of availalbel dates
    tmp = gee_l8_fltd.getInfo()
    tmp_ids = [x['properties']['system:index'] for x in tmp['features']]
    dates = np.array([dt.datetime.strptime(x[9::], '%Y%j').date() for x in tmp_ids])

    # find the closest acquisitions
    doi = dt.date(year=year, month=month, day=day)
    doi_index = np.argmin(np.abs(dates - doi))
    date_selected = dates[doi_index]

    # filter collection for respective dates
    edate = date_selected + dt.timedelta(days=1)
    gee_l8_fltd = gee_l8_fltd.filterDate(date_selected.strftime('%Y-%m-%d'), edate.strftime('%Y-%m-%d'))

    # mosaic scenes
    gee_l8_fltd = gee_l8_fltd.map(mask)
    gee_l8_mosaic = ee.Image(gee_l8_fltd.mosaic().clip(roi))

    return(gee_l8_mosaic)


def extr_L8_array(minlon, minlat, maxlon, maxlat, year, month, day, workdir, sampling):

    ee.Initialize()

    def setresample(image):
        image = image.resample()
        return (image)

    def mask(image):
        # mask image
        immask = image.select('cfmask').eq(ee.Image(0))
        image = image.updateMask(immask)
        return(image)


    # load collection
    gee_l8_collection = ee.ImageCollection('LANDSAT/LC8_SR').map(setresample)

    # construct roi
    roi = ee.Geometry.Polygon([[minlon, minlat], [minlon, maxlat],
                               [maxlon, maxlat], [maxlon, minlat],
                               [minlon, minlat]])

    # filter
    doi = dt.date(year=year, month=month, day=day)
    sdate = doi - dt.timedelta(days=30)
    edate = doi + dt.timedelta(days=30)
    gee_l8_fltd = gee_l8_collection.filterBounds(roi).filterDate(sdate.strftime('%Y-%m-%d'), edate.strftime('%Y-%m-%d'))

    # create a list of availalbel dates
    tmp = gee_l8_fltd.getInfo()
    tmp_ids = [x['properties']['system:index'] for x in tmp['features']]
    dates = np.array([dt.datetime.strptime(x[9::], '%Y%j').date() for x in tmp_ids])

    # find the closest acquisitions
    doi = dt.date(year=year, month=month, day=day)
    doi_index = np.argmin(np.abs(dates - doi))
    date_selected = dates[doi_index]

    # filter collection for respective dates
    edate = date_selected + dt.timedelta(days=1)
    gee_l8_fltd = gee_l8_fltd.filterDate(date_selected.strftime('%Y-%m-%d'), edate.strftime('%Y-%m-%d'))

    # mosaic scenes
    gee_l8_fltd = gee_l8_fltd.map(mask)
    gee_l8_mosaic = ee.Image(gee_l8_fltd.mosaic().clip(roi))

    GEtodisk(gee_l8_mosaic.select(['B1','B2','B3','B4','B5','B6','B7']), 'e_l8', workdir, sampling, roi)
    #return(gee_l8_mosaic)


def extr_SIG0_LIA_ts_GEE(lon, lat, bufferSize=20, maskwinter=True):

    ee.Reset()
    ee.Initialize()

    # def setresample(image):
    #     image = image.resample()
    #     return (image)

    def createAvg(image):
        gee_roi = ee.Geometry.Point(lon, lat).buffer(bufferSize)
        tmp = ee.Image(image)

        # load land cover info
        corine = ee.Image('users/felixgreifeneder/corine')

        # create lc mask
        valLClist = [10, 11, 12, 13, 18, 19, 20, 21, 26, 27, 28, 29]

        lcmask = corine.eq(valLClist[0]).bitwiseOr(corine.eq(valLClist[1])) \
            .bitwiseOr(corine.eq(valLClist[2])) \
            .bitwiseOr(corine.eq(valLClist[3])) \
            .bitwiseOr(corine.eq(valLClist[4])) \
            .bitwiseOr(corine.eq(valLClist[5])) \
            .bitwiseOr(corine.eq(valLClist[6])) \
            .bitwiseOr(corine.eq(valLClist[7])) \
            .bitwiseOr(corine.eq(valLClist[8])) \
            .bitwiseOr(corine.eq(valLClist[9])) \
            .bitwiseOr(corine.eq(valLClist[10])) \
            .bitwiseOr(corine.eq(valLClist[11]))

        # mask pixels
        vv = tmp.select('VV')
        vh = tmp.select('VH')
        lia = tmp.select('angle')
        maskvv = vv.gte(-25)
        maskvh = vh.gte(-25)
        masklia1 = lia.gt(10)
        masklia2 = lia.lt(50)
        masklia = masklia1.bitwiseAnd(masklia2)

        mask = maskvv.bitwiseAnd(maskvh)
        mask = mask.bitwiseAnd(masklia)
        mask = mask.bitwiseAnd(lcmask)
        tmp = tmp.updateMask(mask)

        # Conver to linear before averaging
        tmp = tmp.addBands(ee.Image(10).pow(tmp.select('VV').divide(10)))
        tmp = tmp.addBands(ee.Image(10).pow(tmp.select('VH').divide(10)))
        # tmp2 = ee.Image.cat([VVlin, VHlin, tmp.select('angle)')])
        tmp = tmp.select(['constant', 'constant_1', 'angle'], ['VV', 'VH', 'angle'])

        reduced_img_data = tmp.reduceRegion(ee.Reducer.median(), gee_roi, 30)
        totcount = ee.Image(1).reduceRegion(ee.Reducer.count(), gee_roi, 30)
        pcount = tmp.reduceRegion(ee.Reducer.count(), gee_roi, 30)
        return ee.Feature(None, {'result': reduced_img_data, 'count': pcount, 'totcount': totcount})

    # load S1 data
    gee_s1_collection = ee.ImageCollection('COPERNICUS/S1_GRD')#.map(setresample)

    # filter collection
    gee_roi = ee.Geometry.Point(lon, lat).buffer(bufferSize)

    gee_s1_filtered = gee_s1_collection.filter(ee.Filter.eq('instrumentMode', 'IW')) \
        .filterBounds(gee_roi) \
        .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING')) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
        #.filterDate('2014-01-01', opt_end='2015-12-31')
        #.filter(ee.Filter.dayOfYear(121,304)) # 1st of may to 31st of october

    if maskwinter == True:
        gee_s1_filtered = gee_s1_filtered.filter(ee.Filter.dayOfYear(121,304))

    # get the track numbers
    tmp = gee_s1_filtered.getInfo()
    track_series = np.array([x['properties']['relativeOrbitNumber_start'] for x in tmp['features']])
    available_tracks = np.unique(track_series)

    out_dict = {}
    for track_nr in available_tracks:

        #  filter for track
        gee_s1_track_fltd = gee_s1_filtered.filterMetadata('relativeOrbitNumber_start', 'equals', track_nr)

        gee_s1_mapped = gee_s1_track_fltd.map(createAvg)
        tmp = gee_s1_mapped.getInfo()
        # get vv
        vv_sig0 = 10*np.log10(np.array([x['properties']['result']['VV'] for x in tmp['features']], dtype=np.float))

        # get vh
        vh_sig0 = 10*np.log10(np.array([x['properties']['result']['VH'] for x in tmp['features']], dtype=np.float))

        ge_dates = np.array([datetime.strptime(x['id'][17:32], '%Y%m%dT%H%M%S') for x in tmp['features']])

        # get lia
        lia = np.array([x['properties']['result']['angle'] for x in tmp['features']])

        # get val_count
        val_count = np.array([np.float(x['properties']['count']['VV']) / np.float(x['properties']['totcount']['constant']) for x in tmp['features']], dtype=np.float)

        valid = np.where(val_count > 0.4)
        vv_sig0 = vv_sig0[valid]
        vh_sig0 = vh_sig0[valid]
        lia = lia[valid]
        ge_dates = ge_dates[valid]

        out_dict[str(int(track_nr))] =  (ge_dates, {'sig0': vv_sig0, 'sig02': vh_sig0, 'lia': lia})

    return (out_dict)


def extr_SIG0_LIA_ts_GEE_VV(lon, lat, bufferSize=20, maskwinter=True):

    ee.Reset()
    ee.Initialize()

    def setresample(image):
        image = image.resample()
        return (image)

    def createAvg(image):
        gee_roi = ee.Geometry.Point(lon, lat).buffer(bufferSize)
        tmp = ee.Image(image)

        # mask pixels
        lia = tmp.select('angle')
        masklia1 = lia.gt(10)
        masklia2 = lia.lt(50)
        masklia = masklia1.bitwiseAnd(masklia2)

        tmp = tmp.updateMask(masklia)

        # Conver to linear before averaging
        tmp = tmp.addBands(ee.Image(10).pow(tmp.select('VV').divide(10)))
        # tmp2 = ee.Image.cat([VVlin, VHlin, tmp.select('angle)')])
        tmp = tmp.select(['constant', 'angle'], ['VV', 'angle'])

        reduced_img_data = tmp.reduceRegion(ee.Reducer.median(), gee_roi, 10)
        return ee.Feature(None, {'result': reduced_img_data})

    # load S1 data
    gee_s1_collection = ee.ImageCollection('COPERNICUS/S1_GRD').map(setresample)

    # filter S1 collection
    gee_roi = ee.Geometry.Point(lon, lat).buffer(bufferSize)

    gee_s1_filtered = gee_s1_collection.filter(ee.Filter.eq('instrumentMode', 'IW')) \
        .filterBounds(gee_roi) \
        .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING')) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
        #.filterDate('2014-01-01', opt_end='2015-12-31')
        #.filter(ee.Filter.dayOfYear(121,304)) # 1st of may to 31st of october

    if maskwinter == True:
        gee_s1_filtered = gee_s1_filtered.filter(ee.Filter.dayOfYear(121,304))

    # apply lc mask
    #gee_s1_filtered.updateMask(lcmask)

    # .filterMetadata('relativeOrbitNumber_start', 'equals', track_nr) \
    # get the track numbers
    tmp = gee_s1_filtered.getInfo()
    track_series = np.array([x['properties']['relativeOrbitNumber_start'] for x in tmp['features']])
    available_tracks = np.unique(track_series)


    out_dict = {}
    for track_nr in available_tracks:

        #  filter for track
        gee_s1_track_fltd = gee_s1_filtered.filterMetadata('relativeOrbitNumber_start', 'equals', track_nr)

        gee_s1_mapped = gee_s1_track_fltd.map(createAvg)
        tmp = gee_s1_mapped.getInfo()
        # get vv
        vv_sig0 = 10*np.log10(np.array([x['properties']['result']['VV'] for x in tmp['features']], dtype=np.float))
        ge_dates = np.array([datetime.strptime(x['id'][17:32], '%Y%m%dT%H%M%S') for x in tmp['features']])

        # get lia
        lia = np.array([x['properties']['result']['angle'] for x in tmp['features']])

        out_dict[str(int(track_nr))] =  (ge_dates, {'sig0': vv_sig0, 'lia': lia})

    return (out_dict)


def extr_GEE_array(minlon, minlat, maxlon, maxlat, year, month, day, workdir, tempfilter=True, applylcmask=True, sampling=20, dualpol=True):

    def setresample(image):
        image = image.resample()
        return (image)

    def toln(image):

        tmp = ee.Image(image)

        # Convert to linear
        vv = ee.Image(10).pow(tmp.select('VV').divide(10))
        if dualpol == True:
            vh = ee.Image(10).pow(tmp.select('VH').divide(10))

        # Convert to ln
        out = vv.log()
        if dualpol == True:
            out = out.addBands(vh.log())
            out = out.select(['constant', 'constant_1'], ['VV', 'VH'])
        else:
            out = out.select(['constant'], ['VV'])

        return out.set('system:time_start', tmp.get('system:time_start'))

    def tolin(image):

        tmp = ee.Image(image)

        # Covert to linear
        vv = ee.Image(10).pow(tmp.select('VV').divide(10))
        if dualpol == True:
            vh = ee.Image(10).pow(tmp.select('VH').divide(10))

        # Convert to
        if dualpol == True:
            out = vv.addBands(vh)
            out = out.select(['constant', 'constant_1'], ['VV', 'VH'])
        else:
            out = vv.select(['constant'], ['VV'])

        return out.set('system:time_start', tmp.get('system:time_start'))

    def todb(image):

        tmp = ee.Image(image)

        return ee.Image(10).multiply(tmp.log10()).set('system:time_start', tmp.get('system:time_start'))

    ee.Initialize()

    # load S1 data
    gee_s1_collection = ee.ImageCollection('COPERNICUS/S1_GRD')

    # load land cover info
    corine = ee.Image('users/felixgreifeneder/corine')

    # create lc mask
    valLClist = [10, 11, 12, 13, 18, 19, 20, 21, 26, 27, 28, 29]

    lcmask = corine.eq(valLClist[0]).bitwiseOr(corine.eq(valLClist[1])) \
        .bitwiseOr(corine.eq(valLClist[2])) \
        .bitwiseOr(corine.eq(valLClist[3])) \
        .bitwiseOr(corine.eq(valLClist[4])) \
        .bitwiseOr(corine.eq(valLClist[5])) \
        .bitwiseOr(corine.eq(valLClist[6])) \
        .bitwiseOr(corine.eq(valLClist[7])) \
        .bitwiseOr(corine.eq(valLClist[8])) \
        .bitwiseOr(corine.eq(valLClist[9])) \
        .bitwiseOr(corine.eq(valLClist[10])) \
        .bitwiseOr(corine.eq(valLClist[11]))

    # for lc_id in range(1, len(valLClist)):
    #     tmpmask = corine.eq(valLClist[lc_id])
    #     lcmask = lcmask.bitwiseAnd(tmpmask)

    # construct roi
    roi = ee.Geometry.Polygon([[minlon, minlat], [minlon, maxlat],
                               [maxlon, maxlat], [maxlon, minlat],
                               [minlon, minlat]])

    # ASCENDING acquisitions
    gee_s1_filtered = gee_s1_collection.filter(ee.Filter.eq('instrumentMode', 'IW')) \
        .filterBounds(roi) \
        .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING')) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')).map(setresample)  # \

    # gee_s1_filtered_dsc = gee_s1_collection.filter(ee.Filter.eq('instrumentMode', 'IW')) \
    #     .filterBounds(roi) \
    #     .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING')) \
    #     .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))

    if dualpol == True:
        gee_s1_filtered = gee_s1_filtered.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
        # gee_s1_filtered_dsc = gee_s1_filtered_dsc.filter(
        #     ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))

    # create a list of availalbel dates
    tmp = gee_s1_filtered.getInfo()
    tmp_ids = [x['properties']['system:index'] for x in tmp['features']]
    dates = np.array([dt.date(year=int(x[17:21]), month=int(x[21:23]), day=int(x[23:25])) for x in tmp_ids])
    # descending
    # tmp = gee_s1_filtered_dsc.getInfo()
    # tmp_ids = [x['properties']['system:index'] for x in tmp['features']]
    # dates_dsc = np.array([dt.date(year=int(x[17:21]), month=int(x[21:23]), day=int(x[23:25])) for x in tmp_ids])

    # find the closest acquisitions
    doi = dt.date(year=year, month=month, day=day)
    doi_index = np.argmin(np.abs(dates - doi))
    date_selected = dates[doi_index]
    # descending
    # doi_index_dsc = np.argmin(np.abs(dates_dsc - doi))
    # date_selected_dsc = dates_dsc[doi_index_dsc]

    # filter imagecollection for respective date
    gee_s1_list = gee_s1_filtered.toList(500)
    doi_indices = np.where(dates == date_selected)[0]
    gee_s1_drange = ee.ImageCollection(gee_s1_list.slice(doi_indices[0], doi_indices[-1] + 1))
    s1_sig0 = gee_s1_drange.mosaic()
    s1_sig0 = ee.Image(s1_sig0.copyProperties(gee_s1_drange.first()))

    # fetch image from image collection
    # s1_sig0 = ee.Image(gee_s1_list.get(doi_index))
    s1_lia = s1_sig0.select('angle').clip(roi)
    # get the track number
    s1_sig0_info = s1_sig0.getInfo()
    track_nr = s1_sig0_info['properties']['relativeOrbitNumber_start']
    # descending
    # gee_s1_list_dsc = gee_s1_filtered_dsc.toList(500)
    # s1_sig0_dsc = ee.Image(gee_s1_list_dsc.get(doi_index_dsc))

    # despeckle
    if tempfilter == True:
        radius = 7
        units = 'pixels'
        gee_s1_linear = gee_s1_filtered.map(tolin)
        gee_s1_dspckld_vv = multitemporalDespeckle(gee_s1_linear.select('VV'), radius, units,
                                                   {'before': -12, 'after': 12, 'units': 'month'})
        gee_s1_dspckld_vv = gee_s1_dspckld_vv.map(todb)
        gee_s1_list_vv = gee_s1_dspckld_vv.toList(500)
        gee_s1_fltrd_vv = ee.ImageCollection(gee_s1_list_vv.slice(doi_indices[0], doi_indices[-1] + 1))
        s1_sig0_vv = gee_s1_fltrd_vv.mosaic()
        # s1_sig0_vv = ee.Image(gee_s1_list_vv.get(doi_index))

        # descending
        # gee_s1_linear_dsc = gee_s1_filtered_dsc.map(tolin)
        # gee_s1_dspckld_vv_dsc = multitemporalDespeckle(gee_s1_linear_dsc.select('VV'), radius, units,
        #                                                 {'before': -12, 'after': 12, 'units': 'month'})
        # gee_s1_dspckld_vv_dsc = gee_s1_dspckld_vv_dsc.map(todb)
        # gee_s1_list_vv_dsc = gee_s1_dspckld_vv_dsc.toList(500)
        # s1_sig0_vv_dsc = ee.Image(gee_s1_list_vv_dsc.get(doi_index_dsc))


        if dualpol == True:
            gee_s1_dspckld_vh = multitemporalDespeckle(gee_s1_linear.select('VH'), radius, units,
                                                       {'before': -12, 'after': 12, 'units': 'month'})
            gee_s1_dspckld_vh = gee_s1_dspckld_vh.map(todb)
            gee_s1_list_vh = gee_s1_dspckld_vh.toList(500)
            gee_s1_fltrd_vh = ee.ImageCollection(gee_s1_list_vh.slice(doi_indices[0], doi_indices[-1] + 1))
            s1_sig0_vh = gee_s1_fltrd_vh.mosaic()

        if dualpol == True:
            s1_sig0 = s1_sig0_vv.addBands(s1_sig0_vh).select(['constant', 'constant_1'], ['VV', 'VH'])
        else:
            s1_sig0 = s1_sig0_vv.select(['constant'], ['VV'])
            # s1_sig0_dsc = s1_sig0_vv_dsc.select(['constant'], ['VV'])

    # extract information
    if applylcmask == True:
        s1_sig0 = s1_sig0.updateMask(lcmask)
    # s1_sig0 = s1_sig0.clip(roi)
    s1_sig0_vv = s1_sig0.select('VV')
    s1_sig0_vv = s1_sig0_vv.clip(roi)
    if dualpol == True:
        s1_sig0_vh = s1_sig0.select('VH')
        s1_sig0_vh = s1_sig0_vh.clip(roi)
    # descending
    # s1_sig0_vv_dsc = s1_sig0_dsc.select('VV')
    # s1_sig0_vv_dsc = s1_sig0_vv_dsc.clip(roi)
    # s1_sig0diff = s1_sig0_vv.subtract(s1_sig0_vv_dsc)

    # compute temporal statistics
    gee_s1_filtered = gee_s1_filtered.filterMetadata('relativeOrbitNumber_start', 'equals', track_nr)
    gee_s1_ln = gee_s1_filtered.map(toln)
    if applylcmask == True:
        gee_s1_ln = gee_s1_ln.updateMask(lcmask)
    # gee_s1_ln = gee_s1_ln.clip(roi)
    k1vv = ee.Image(gee_s1_ln.select('VV').mean()).clip(roi)
    k2vv = ee.Image(gee_s1_ln.select('VV').reduce(ee.Reducer.stdDev())).clip(roi)
    if dualpol == True:
        k1vh = ee.Image(gee_s1_ln.select('VH').mean()).clip(roi)
        k2vh = ee.Image(gee_s1_ln.select('VH').reduce(ee.Reducer.stdDev())).clip(roi)

    # export
    s1_sig0_vv = s1_sig0_vv.reproject(s1_lia.projection(), scale=sampling)
    # s1_sig0diff = s1_sig0diff.reproject(s1_lia.projection(), scale=sampling)
    lia_exp = ee.batch.Export.image.toDrive(image=s1_lia, description='lia',
                                            fileNamePrefix='s1lia'+str(date_selected), scale=sampling)
    # sig0diff_exp = ee.batch.Export.image.toDrive(image=s1_sig0diff, description='diff',
    #                                              fileNamePrefix='s1diff'+str(date_selected), scale=sampling)

    sig0_vv_exp = ee.batch.Export.image.toDrive(image=s1_sig0_vv, description='svv_exp', #folder='GEE_output',
                                                fileNamePrefix='sig0vv'+str(date_selected), scale=sampling)
    k1vv_exp = ee.batch.Export.image.toDrive(image=k1vv, description='kvv_exp',  # folder='GEE_output',
                                             fileNamePrefix='k1vv' + str(int(track_nr)), scale=sampling)
    k2vv_exp = ee.batch.Export.image.toDrive(image=k2vv, description='k2vv_exp',  # folder='GEE_output',
                                             fileNamePrefix='k2vv' + str(int(track_nr)), scale=sampling)
    if dualpol == True:
        s1_sig0_vh = s1_sig0_vh.reproject(s1_lia.projection, scale=sampling)
        sig0_vh_exp = ee.batch.Export.image.toDrive(image=s1_sig0_vh, description='svh_exp', #folder='GEE_output',
                                                    fileNamePrefix='sig0vh'+str(date_selected), scale=sampling)
        k1vh_exp = ee.batch.Export.image.toDrive(image=k1vh, description='kvh_exp', #folder='GEE_output',
                                                    fileNamePrefix='k1vh'+str(int(track_nr)), scale=sampling)
        k2vh_exp = ee.batch.Export.image.toDrive(image=k2vh, description='k2vh_exp', #folder='GEE_output',
                                                 fileNamePrefix='k2vh'+str(int(track_nr)), scale=sampling)

    lia_exp.start()
    # sig0diff_exp.start()
    sig0_vv_exp.start()
    k1vv_exp.start()
    k2vv_exp.start()
    if dualpol == True:
        sig0_vh_exp.start()
        k1vh_exp.start()
        k2vh_exp.start()

    print('Exporting images to Google Drive ...')
    if dualpol == True:
        while (k2vh_exp.active() == True) or (k2vv_exp.active() == True) \
            or (k1vh_exp.active() == True) or (k1vv_exp.active() == True) \
            or (sig0_vh_exp.active() == True) or (sig0_vv_exp.active() == True) \
            or (lia_exp.active() == True):
            time.sleep(2)
        else:
            print('Export completed')
    else:
        while (k2vv_exp.active() == True) \
            or (k1vv_exp.active() == True) \
            or (sig0_vv_exp.active() == True) \
            or (lia_exp.active() == True):# \
            #or (sig0diff_exp.active() == True):
            time.sleep(2)
        else:
            print('Export completed')

    # initialise Google Drive
    drive_handler = gdrive()
    print('Downloading files ...')
    print('s1lia' + str(date_selected))
    drive_handler.download_file('s1lia' + str(date_selected) + '.tif',
                                workdir + 'b_s1lia' + str(date_selected) + '.tif')
    drive_handler.delete_file('s1lia' + str(date_selected) + '.tif')

    # print('sig0diff' + str(date_selected))
    # drive_handler.download_file('s1diff'+str(date_selected) + '.tif',
    #                             workdir + 's1diff'+str(date_selected) + '.tif')
    # drive_handler.delete_file('s1diff'+str(date_selected)+ '.tif')

    print('sig0vv'+str(date_selected))
    drive_handler.download_file('sig0vv'+str(date_selected)+'.tif', workdir + 'a_sig0vv'+str(date_selected) + '.tif')
    drive_handler.delete_file('sig0vv'+str(date_selected)+'.tif')
    print('k1vv' + str(int(track_nr)))
    drive_handler.download_file('k1vv' + str(int(track_nr)) + '.tif', workdir + 'c_k1vv' + str(int(track_nr)) + '.tif')
    drive_handler.delete_file('k1vv' + str(int(track_nr)) + '.tif')
    print('k2vv' + str(int(track_nr)))
    drive_handler.download_file('k2vv' + str(int(track_nr)) + '.tif', workdir + 'd_k2vv' + str(int(track_nr)) + '.tif')
    drive_handler.delete_file('k2vv' + str(int(track_nr)) + '.tif')
    if dualpol == True:
        print('sig0vh' + str(date_selected))
        drive_handler.download_file('sig0vh'+str(date_selected)+'.tif', workdir + 'sig0vh' + str(date_selected) + '.tif')
        drive_handler.delete_file('sig0vh' + str(date_selected)+'.tif')
        print('k1vh' + str(int(track_nr)))
        drive_handler.download_file('k1vh'+str(int(track_nr))+'.tif', workdir + 'k1vh' + str(int(track_nr)) + '.tif')
        drive_handler.delete_file('k1vh'+str(int(track_nr))+'.tif')
        print('k2vh' + str(int(track_nr)))
        drive_handler.download_file('k2vh' + str(int(track_nr))+'.tif', workdir + 'k2vh' + str(int(track_nr)) + '.tif')
        drive_handler.delete_file('k2vh'+str(int(track_nr))+'.tif')


def extr_GEE_array_reGE(minlon, minlat, maxlon, maxlat, year, month, day, tempfilter=True, applylcmask=True,
                       sampling=20, dualpol=True):

        def setresample(image):
            image = image.resample()
            return(image)

        def toln(image):

            tmp = ee.Image(image)

            # Convert to linear
            vv = ee.Image(10).pow(tmp.select('VV').divide(10))
            if dualpol == True:
                vh = ee.Image(10).pow(tmp.select('VH').divide(10))

            # Convert to ln
            out = vv.log()
            if dualpol == True:
                out = out.addBands(vh.log())
                out = out.select(['constant', 'constant_1'], ['VV', 'VH'])
            else:
                out = out.select(['constant'], ['VV'])

            return out.set('system:time_start', tmp.get('system:time_start'))

        def tolin(image):

            tmp = ee.Image(image)

            # Covert to linear
            vv = ee.Image(10).pow(tmp.select('VV').divide(10))
            if dualpol == True:
                vh = ee.Image(10).pow(tmp.select('VH').divide(10))

            # Convert to
            if dualpol == True:
                out = vv.addBands(vh)
                out = out.select(['constant', 'constant_1'], ['VV', 'VH'])
            else:
                out = vv.select(['constant'], ['VV'])

            return out.set('system:time_start', tmp.get('system:time_start'))

        def todb(image):

            tmp = ee.Image(image)

            return ee.Image(10).multiply(tmp.log10()).set('system:time_start', tmp.get('system:time_start'))

        ee.Initialize()

        # load S1 data
        gee_s1_collection = ee.ImageCollection('COPERNICUS/S1_GRD')

        # load land cover info
        corine = ee.Image('users/felixgreifeneder/corine')

        # create lc mask
        valLClist = [10, 11, 12, 13, 18, 19, 20, 21, 26, 27, 28, 29]

        lcmask = corine.eq(valLClist[0]).bitwiseOr(corine.eq(valLClist[1])) \
            .bitwiseOr(corine.eq(valLClist[2])) \
            .bitwiseOr(corine.eq(valLClist[3])) \
            .bitwiseOr(corine.eq(valLClist[4])) \
            .bitwiseOr(corine.eq(valLClist[5])) \
            .bitwiseOr(corine.eq(valLClist[6])) \
            .bitwiseOr(corine.eq(valLClist[7])) \
            .bitwiseOr(corine.eq(valLClist[8])) \
            .bitwiseOr(corine.eq(valLClist[9])) \
            .bitwiseOr(corine.eq(valLClist[10])) \
            .bitwiseOr(corine.eq(valLClist[11]))

        # for lc_id in range(1, len(valLClist)):
        #     tmpmask = corine.eq(valLClist[lc_id])
        #     lcmask = lcmask.bitwiseAnd(tmpmask)

        # construct roi
        roi = ee.Geometry.Polygon([[minlon, minlat], [minlon, maxlat],
                                   [maxlon, maxlat], [maxlon, minlat],
                                   [minlon, minlat]])

        # ASCENDING acquisitions
        gee_s1_filtered = gee_s1_collection.filter(ee.Filter.eq('instrumentMode', 'IW')) \
            .filterBounds(roi) \
            .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING')) \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')).map(setresample)

        if dualpol == True:
            gee_s1_filtered = gee_s1_filtered.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))

        # create a list of availalbel dates
        tmp = gee_s1_filtered.getInfo()
        tmp_ids = [x['properties']['system:index'] for x in tmp['features']]
        dates = np.array([dt.date(year=int(x[17:21]), month=int(x[21:23]), day=int(x[23:25])) for x in tmp_ids])

        # find the closest acquisitions
        doi = dt.date(year=year, month=month, day=day)
        doi_index = np.argmin(np.abs(dates - doi))
        date_selected = dates[doi_index]

        # filter imagecollection for respective date
        gee_s1_list = gee_s1_filtered.toList(1000)
        doi_indices = np.where(dates == date_selected)[0]
        gee_s1_drange = ee.ImageCollection(gee_s1_list.slice(doi_indices[0], doi_indices[-1]+1))
        s1_sig0 = gee_s1_drange.mosaic()
        s1_sig0 = ee.Image(s1_sig0.copyProperties(gee_s1_drange.first()))

        # fetch image from image collection
        s1_lia = s1_sig0.select('angle').clip(roi)
        # get the track number
        s1_sig0_info = s1_sig0.getInfo()
        track_nr = s1_sig0_info['properties']['relativeOrbitNumber_start']

        # despeckle
        if tempfilter == True:
            radius = 7
            units = 'pixels'
            gee_s1_linear = gee_s1_filtered.map(tolin)
            gee_s1_dspckld_vv = multitemporalDespeckle(gee_s1_linear.select('VV'), radius, units,
                                                       {'before': -12, 'after': 12, 'units': 'month'})
            gee_s1_dspckld_vv = gee_s1_dspckld_vv.map(todb)
            gee_s1_list_vv = gee_s1_dspckld_vv.toList(1000)
            gee_s1_fltrd_vv = ee.ImageCollection(gee_s1_list_vv.slice(doi_indices[0], doi_indices[-1]+1))
            s1_sig0_vv = gee_s1_fltrd_vv.mosaic()
            #s1_sig0_vv = ee.Image(gee_s1_list_vv.get(doi_index))

            if dualpol == True:
                gee_s1_dspckld_vh = multitemporalDespeckle(gee_s1_linear.select('VH'), radius, units,
                                                           {'before': -12, 'after': 12, 'units': 'month'})
                gee_s1_dspckld_vh = gee_s1_dspckld_vh.map(todb)
                gee_s1_list_vh = gee_s1_dspckld_vh.toList(1000)
                gee_s1_fltrd_vh = ee.ImageCollection(gee_s1_list_vh.slice(doi_indices[0], doi_indices[-1] + 1))
                s1_sig0_vh = gee_s1_fltrd_vh.mosaic()

            if dualpol == True:
                s1_sig0 = s1_sig0_vv.addBands(s1_sig0_vh).select(['constant', 'constant_1'], ['VV', 'VH'])
            else:
                s1_sig0 = s1_sig0_vv.select(['constant'], ['VV'])
                # s1_sig0_dsc = s1_sig0_vv_dsc.select(['constant'], ['VV'])

        # extract information
        if applylcmask == True:
            s1_sig0 = s1_sig0.updateMask(lcmask)
        # s1_sig0 = s1_sig0.clip(roi)
        s1_sig0_vv = s1_sig0.select('VV')
        s1_sig0_vv = s1_sig0_vv.clip(roi)
        if dualpol == True:
            s1_sig0_vh = s1_sig0.select('VH')
            s1_sig0_vh = s1_sig0_vh.clip(roi)

        # compute temporal statistics
        gee_s1_filtered = gee_s1_filtered.filterMetadata('relativeOrbitNumber_start', 'equals', track_nr)
        gee_s1_ln = gee_s1_filtered.map(toln)
        # gee_s1_ln = gee_s1_ln.clip(roi)
        k1vv = ee.Image(gee_s1_ln.select('VV').mean()).clip(roi)
        k2vv = ee.Image(gee_s1_ln.select('VV').reduce(ee.Reducer.stdDev())).clip(roi)
        if applylcmask == True:
            k1vv = k1vv.updateMask(lcmask)
            k2vv = k2vv.updateMask(lcmask)
        if dualpol == True:
            k1vh = ee.Image(gee_s1_ln.select('VH').mean()).clip(roi)
            k2vh = ee.Image(gee_s1_ln.select('VH').reduce(ee.Reducer.stdDev())).clip(roi)
        if applylcmask == True:
            k1vh = k1vh.updateMask(lcmask)
            k2vh = k2vh.updateMask(lcmask)

        # export
        if dualpol == False:
            #s1_sig0_vv = s1_sig0_vv.reproject(s1_lia.projection())
            return(s1_sig0_vv, s1_lia, k1vv, k2vv, roi)
        else:
            return(s1_lia, s1_sig0_vv, s1_sig0_vh, k1vv, k1vh, k2vv, k2vh, roi)


def get_s1_dates(minlon, minlat, maxlon, maxlat, tracknr=None, dualpol=True):

    ee.Initialize()

    # load S1 data
    gee_s1_collection = ee.ImageCollection('COPERNICUS/S1_GRD')

    # construct roi
    roi = ee.Geometry.Polygon([[minlon, minlat], [minlon, maxlat],
                               [maxlon, maxlat], [maxlon, minlat],
                               [minlon, minlat]])

    # ASCENDING acquisitions
    gee_s1_filtered = gee_s1_collection.filter(ee.Filter.eq('instrumentMode', 'IW')) \
        .filterBounds(roi) \
        .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING')) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))

    if dualpol == True:
        gee_s1_filtered = gee_s1_filtered.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))

    if tracknr is not None:
        gee_s1_filtered = gee_s1_filtered.filter(ee.Filter.eq('relativeOrbitNumber_start', tracknr))

    # create a list of availalbel dates
    tmp = gee_s1_filtered.getInfo()
    tmp_ids = [x['properties']['system:index'] for x in tmp['features']]
    dates = np.array([dt.date(year=int(x[17:21]), month=int(x[21:23]), day=int(x[23:25])) for x in tmp_ids])

    return(dates)


# extract time series of SIG0 and LIA from SGRT database
def extr_SIG0_LIA_ts(dir_root, product_id, soft_id, product_name, src_res, lon, lat, xdim, ydim,
                     pol_name=None, grid=None, subgrid='EU', hour=None, sat_pass=None, monthmask=None):
    #initialise grid
    alpGrid = Equi7.Equi7Grid(src_res)

    #identify tile
    if grid is None:
        Equi7XY = alpGrid.lonlat2equi7xy(lon, lat)
    elif grid == 'Equi7':
        Equi7XY = [subgrid, lon, lat]
    TileName = alpGrid.identfy_tile(Equi7XY[0], [Equi7XY[1],Equi7XY[2]])
    TileExtent = Equi7.Equi7Tile(TileName).extent
    #load tile
    TOI = SgrtTile.SgrtTile(dir_root=dir_root, product_id=product_id, soft_id=soft_id, product_name=product_name, ftile=TileName, src_res=src_res)

    # extract data
    x = int((Equi7XY[1] - TileExtent[0]) / src_res)
    y = int((TileExtent[3] - Equi7XY[2]) / src_res)

    # check if month mask is set
    if monthmask is None:
        monthmask = [1,2,3,4,5,6, 7, 8, 9,10,11,12]

    #extract data
    if pol_name is None:
        SIG0 = TOI.read_ts("SIG0_", x, y, xsize=xdim, ysize=ydim)
        LIA = TOI.read_ts("PLIA_", x, y, xsize=xdim, ysize=ydim)

        # check if date dublicates exist
        udates = np.unique(SIG0[0], return_index=True)
        days = np.array(SIG0[0])[udates[1]]
        data = np.array(SIG0[1])[udates[1],:,:]
        SIG0 = (days, data)
        udates = np.unique(LIA[0], return_index=True)
        days = np.array(LIA[0])[udates[1]]
        data = np.array(LIA[1])[udates[1],:,:]
        LIA = (days, data)

    elif len(pol_name) == 1:
        SIG0 = TOI.read_ts("SIG0_", x, y, xsize=xdim, ysize=ydim, pol_name=pol_name.upper())
        LIA = TOI.read_ts("PLIA_", x, y, xsize=xdim, ysize=ydim)

        # check if date dublicates exist
        udates = np.unique(SIG0[0], return_index=True)
        days = np.array(SIG0[0])[udates[1]]
        data = np.array(SIG0[1])[udates[1],:,:]
        SIG0 = (days, data)
        udates = np.unique(LIA[0], return_index=True)
        days = np.array(LIA[0])[udates[1]]
        data = np.array(LIA[1])[udates[1],:,:]
        LIA = (days, data)

    elif len(pol_name) == 2:
        SIG0 = TOI.read_ts("SIG0_", x, y, xsize=xdim, ysize=ydim,
                           pol_name=pol_name[0].upper(), sat_pass=sat_pass)
        SIG02 = TOI.read_ts("SIG0_", x, y, xsize=xdim, ysize=ydim,
                            pol_name=pol_name[1].upper(), sat_pass=sat_pass)
        LIA = TOI.read_ts("LIA__", x, y, xsize=xdim, ysize=ydim, sat_pass=sat_pass)

        # this is temporary: filter scenes based on time, TODO: change or remove
        if hour is not None:
            morning = np.where(np.array([SIG0[0][i].hour for i in range(len(SIG0[0]))]) == hour)[0]
            SIG0 = (np.array(SIG0[0])[morning], SIG0[1][morning])
            morning = np.where(np.array([SIG02[0][i].hour for i in range(len(SIG02[0]))]) == hour)[0]
            SIG02 = (np.array(SIG02[0])[morning], SIG02[1][morning])
            morning = np.where(np.array([LIA[0][i].hour for i in range(len(LIA[0]))]) == hour)[0]
            LIA = (np.array(LIA[0])[morning], LIA[1][morning])

        # filter months
        # TODO make an option
        summer = np.where(np.in1d(np.array([SIG0[0][i].month for i in range(len(SIG0[0]))]), monthmask))[0]
        SIG0 = (np.array(SIG0[0])[summer], SIG0[1][summer])
        summer = np.where(np.in1d(np.array([SIG02[0][i].month for i in range(len(SIG02[0]))]), monthmask))[0]
        SIG02 = (np.array(SIG02[0])[summer], SIG02[1][summer])
        summer = np.where(np.in1d(np.array([LIA[0][i].month for i in range(len(LIA[0]))]), monthmask))[0]
        LIA = (np.array(LIA[0])[summer], LIA[1][summer])

        # check if date dublicates exist
        # TODO average if two measurements in one day
        datedate = [SIG0[0][i].date() for i in range(len(SIG0[0]))]
        udates = np.unique(datedate, return_index=True)
        days = np.array(SIG0[0])[udates[1]]
        data = np.array(SIG0[1])[udates[1],:,:]
        SIG0 = (days, data)
        datedate = [SIG02[0][i].date() for i in range(len(SIG02[0]))]
        udates = np.unique(datedate, return_index=True)
        days = np.array(SIG02[0])[udates[1]]
        data = np.array(SIG02[1])[udates[1],:,:]
        SIG02 = (days, data)
        datedate = [LIA[0][i].date() for i in range(len(LIA[0]))]
        udates = np.unique(datedate, return_index=True)
        days = np.array(LIA[0])[udates[1]]
        data = np.array(LIA[1])[udates[1],:,:]
        LIA = (days, data)
    else:
        return None

    # format datelists and check if all dates are available for both SIG0 and LIA.
    # datelistSIG = []
    # datelistLIA = []
    # for i in range(len(SIG0[0])): datelistSIG.append(int(SIG0[0][i].toordinal()))
    # for i in range(len(LIA[0])): datelistLIA.append(int(LIA[0][i].toordinal()))
    datelistSIG = SIG0[0]
    if len(pol_name) == 2:
        datelistSIG2 = SIG02[0]
    else:
        datelistSIG2 = SIG02[0]

    datelistLIA = LIA[0]

    datelistFINAL = [x for x in datelistSIG if (x in datelistLIA) and (x in datelistSIG2)]

    SIG0out = [SIG0[1][x,:,:] for x in range(len(SIG0[0])) if datelistSIG[x] in datelistFINAL]
    LIAout = [LIA[1][x,:,:] for x in range(len(LIA[0])) if datelistLIA[x] in datelistFINAL]
    if len(pol_name) == 1:
        outtuple = (np.asarray(datelistFINAL), {'sig0': np.asarray(SIG0out), 'lia': np.asarray(LIAout)})
    elif len(pol_name) == 2:
        SIG0out2 = [SIG02[1][x,:,:] for x in range(len(SIG02[0])) if datelistSIG2[x] in datelistFINAL]
        outtuple = (np.asarray(datelistFINAL), {'sig0': np.asarray(SIG0out), 'sig02': np.asarray(SIG0out2), 'lia': np.asarray(LIAout)})

    #TOI = None

    return outtuple


def read_NORM_SIG0(dir_root, product_id, soft_id, product_name, src_res, lon, lat, xdim, ydim, pol_name=None, grid=None):
    # initialise grid
    alpGrid = Equi7.Equi7Grid(src_res)

    # identify tile
    if grid is None:
        Equi7XY = alpGrid.lonlat2equi7xy(lon, lat)
    elif grid == 'Equi7':
        Equi7XY = ['EU', lon, lat]
    TileName = alpGrid.identfy_tile(Equi7XY[0], [Equi7XY[1], Equi7XY[2]])
    TileExtent = Equi7.Equi7Tile(TileName).extent
    # load tile
    TOI = SgrtTile.SgrtTile(dir_root=dir_root, product_id=product_id, soft_id=soft_id, product_name=product_name,
                            ftile=TileName, src_res=src_res)
    TOI_LIA = SgrtTile.SgrtTile(dir_root=dir_root, product_id=product_id, soft_id='A0111', product_name='resampled',
                                ftile=TileName, src_res=src_res) # TODO allow to specify different versions

    # extract data
    x = int((Equi7XY[1] - TileExtent[0]) / src_res)
    y = int((TileExtent[3] - Equi7XY[2]) / src_res)


    # extract data
    if pol_name is None:
        SIG0 = TOI.read_ts("SIGNM", x, y, xsize=xdim, ysize=ydim)
        LIA = TOI_LIA.read_ts("PLIA_", x, y, xsize=xdim, ysize=ydim)

        # check if date dublicates exist
        udates = np.unique(SIG0[0], return_index=True)
        days = np.array(SIG0[0])[udates[1]]
        data = np.array(SIG0[1])[udates[1], :, :]
        SIG0 = (days, data)

    elif len(pol_name) == 1:
        SIG0 = TOI.read_ts("SIGNM", x, y, xsize=xdim, ysize=ydim, pol_name=pol_name.upper())
        LIA = TOI_LIA.read_ts("PLIA_", x, y, xsize=xdim, ysize=ydim)

        # check if date dublicates exist
        udates = np.unique(SIG0[0], return_index=True)
        days = np.array(SIG0[0])[udates[1]]
        data = np.array(SIG0[1])[udates[1], :, :]
        SIG0 = (days, data)

    elif len(pol_name) == 2:
        SIG0 = TOI.read_ts("SIGNM", x, y, xsize=xdim, ysize=ydim, pol_name=pol_name[0].upper(), sat_pass='A')
        SIG02 = TOI.read_ts("SIGNM", x, y, xsize=xdim, ysize=ydim, pol_name=pol_name[1].upper(), sat_pass='A')
        LIA = TOI_LIA.read_ts("PLIA_", x, y, xsize=xdim, ysize=ydim)

        # check if date dublicates exist
        udates = np.unique(SIG0[0], return_index=True)
        days = np.array(SIG0[0])[udates[1]]
        data = np.array(SIG0[1])[udates[1], :, :]
        SIG0 = (days, data)
        udates = np.unique(SIG02[0], return_index=True)
        days = np.array(SIG02[0])[udates[1]]
        data = np.array(SIG02[1])[udates[1], :, :]
        SIG02 = (days, data)
        udates = np.unique(LIA[0], return_index=True)
        days = np.array(LIA[0])[udates[1]]
        data = np.array(LIA[1])[udates[1], :, :]
        LIA = (days, data)

    else:
        return None

    # format datelist and, in case of dual-pol, check if dates are available for both
    # polarisations
    datelistSIG = SIG0[0]
    if len(pol_name) == 2:
        datelistSIG2 = SIG02[0]
    else:
        datelistSIG2 = SIG0[0]

    datelistLIA = LIA[0]

    datelistFINAL = [x for x in datelistSIG if (x in datelistLIA) and (x in datelistSIG2)]

    SIG0out = [SIG0[1][x, :, :] for x in range(len(SIG0[0])) if datelistSIG[x] in datelistFINAL]
    LIAout = [LIA[1][x, :, :] for x in range(len(LIA[0])) if datelistLIA[x] in datelistFINAL]
    if len(pol_name) == 1:
        outtuple = (np.asarray(datelistFINAL), {'sig0': np.asarray(SIG0out), 'lia': np.asarray(LIAout)})
    elif len(pol_name) == 2:
        SIG0out2 = [SIG02[1][x, :, :] for x in range(len(SIG02[0])) if datelistSIG2[x] in datelistFINAL]
        outtuple = (np.asarray(datelistFINAL),
                    {'sig0': np.asarray(SIG0out), 'sig02': np.asarray(SIG0out2), 'lia': np.asarray(LIAout)})

    # TOI = None

    return outtuple

#extract value of any given raster at band (band) at lat, lon
def extr_raster_pixel_values(filename, bandnr, lat, lon, dtype):
    #initialisiation
    dataset = gdal.Open(filename, GA_ReadOnly)
    geotransform = dataset.GetGeoTransform()

    originX = geotransform[0]
    originY = geotransform[3]
    pixelSize = geotransform[1]

    #translate lat lon to image coordinates
    imX = int((lon - originX) / pixelSize)
    imY = int((originY - lat) / pixelSize)

    #retrieve data
    band = dataset.GetRasterBand(bandnr)
    if dtype == 1:
        bandArr = band.ReadAsArray().astype(np.byte)
    elif dtype == 2:
        bandArr = band.ReadAsArray().astype(np.int)
    elif dtype == 3:
        bandArr = band.ReadAsArray().astype(np.float32)
    elif dtype ==4:
        bandArr = band.ReadAsArray().astype(np.float64)
    else:
        print('unknown datatype')
        return(-9999)

    band = None
    dataset = None

    return(bandArr[imY, imX])


def GEtodisk(geds, name, dir, sampling, roi):

    file_exp = ee.batch.Export.image.toDrive(image=geds, description='fileexp' + name,
                                             fileNamePrefix=name, scale=sampling, region=roi.toGeoJSON()['coordinates'],
                                             maxPixels=1000000000000)

    file_exp.start()

    while (file_exp.active() == True):
        time.sleep(2)
    else:
        print('Export completed')

    # initialise Google Drive
    drive_handler = gdrive()
    print('Downloading files ...')
    print(name)
    drive_handler.download_file(name + '.tif',
                                dir + name + '.tif')
    drive_handler.delete_file(name + '.tif')