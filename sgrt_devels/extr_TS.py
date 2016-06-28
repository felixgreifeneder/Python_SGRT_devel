__author__ = 'felix'

import numpy as np
import sgrt.common.grids.Equi7Grid as Equi7
import sgrt.common.utils.SgrtTile as SgrtTile
from osgeo import gdal
from osgeo.gdalconst import *
from netCDF4 import Dataset, num2date
from datetime import datetime

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


# extract time series of SIG0 and LIA from SGRT database
def extr_SIG0_LIA_ts(dir_root, product_id, soft_id, product_name, src_res, lon, lat, xdim, ydim, pol_name=None, grid=None, hour=None):
    #initialise grid
    alpGrid = Equi7.Equi7Grid(src_res)

    #identify tile
    if grid is None:
        Equi7XY = alpGrid.lonlat2equi7xy(lon, lat)
    elif grid == 'Equi7':
        Equi7XY = ['EU', lon, lat]
    TileName = alpGrid.identfy_tile(Equi7XY[0], [Equi7XY[1],Equi7XY[2]])
    TileExtent = Equi7.Equi7Tile(TileName).extent
    #load tile
    TOI = SgrtTile.SgrtTile(dir_root=dir_root, product_id=product_id, soft_id=soft_id, product_name=product_name, ftile=TileName, src_res=src_res)

    # extract data
    x = int((Equi7XY[1] - TileExtent[0]) / src_res)
    y = int((TileExtent[3] - Equi7XY[2]) / src_res)

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
        SIG0 = TOI.read_ts("SIG0_", x, y, xsize=xdim, ysize=ydim, pol_name=pol_name[0].upper())
        SIG02 = TOI.read_ts("SIG0_", x, y, xsize=xdim, ysize=ydim, pol_name=pol_name[1].upper())
        LIA = TOI.read_ts("LIA__", x, y, xsize=xdim, ysize=ydim)

        # this is temporary: filter scenes based on time, TODO: change or remove
        if hour is not None:
            morning = np.where(np.array([SIG0[0][i].hour for i in range(len(SIG0[0]))]) == hour)[0]
            SIG0 = (np.array(SIG0[0])[morning], SIG0[1][morning])
            morning = np.where(np.array([SIG02[0][i].hour for i in range(len(SIG02[0]))]) == hour)[0]
            SIG02 = (np.array(SIG02[0])[morning], SIG02[1][morning])
            morning = np.where(np.array([LIA[0][i].hour for i in range(len(LIA[0]))]) == hour)[0]
            LIA = (np.array(LIA[0])[morning], LIA[1][morning])


        # check if date dublicates exist
        udates = np.unique(SIG0[0], return_index=True)
        days = np.array(SIG0[0])[udates[1]]
        data = np.array(SIG0[1])[udates[1],:,:]
        SIG0 = (days, data)
        udates = np.unique(SIG02[0], return_index=True)
        days = np.array(SIG02[0])[udates[1]]
        data = np.array(SIG02[1])[udates[1],:,:]
        SIG02 = (days, data)
        udates = np.unique(LIA[0], return_index=True)
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

