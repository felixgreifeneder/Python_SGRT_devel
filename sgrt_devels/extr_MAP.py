__author__ = 'usergre'

import sgrt.common.grids.Equi7Grid as Equi7
import sgrt.common.utils.SgrtTile as SgrtTile
import numpy as np
from osgeo import gdal
from osgeo import ogr
import re
import os
import fnmatch


def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return(result)


def tile_to_raster(tile, dst):

    x_pixel = tile[0].shape[0]
    y_pixel = tile[0].shape[1]

    driver = gdal.GetDriverByName('GTiff')

    dataset = driver.Create(dst,
                            x_pixel,
                            y_pixel,
                            1,
                            gdal.GDT_Int16)

    dataset.SetGeoTransform(tile[1]['geotransform'])
    dataset.SetProjection(tile[1]['spatialreference'])
    dataset.GetRasterBand(1).WriteArray(tile[0])
    dataset.FlushCache()


def extr_mean_sig0_map(dir_root, product_id, soft_id, product_name,
                       src_res, maxlon, maxlat, minlon, minlat, workdir, outdir):

    #initialise grid
    alpGrid = Equi7.Equi7Grid(src_res)

    # Find overlapping tiles ---
    # create search polygon
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(minlon, maxlat)
    ring.AddPoint(maxlon, maxlat)
    ring.AddPoint(maxlon, minlat)
    ring.AddPoint(minlon, minlat)
    ring.AddPoint(minlon, maxlat)
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)

    ag_tilelist = alpGrid.search_tiles(poly)
    print(ag_tilelist)

    # iterate through the tiles
    for i in range(len(ag_tilelist)):
    #for i in range(2,3):
        tName = ag_tilelist[i]
        TOI = SgrtTile.SgrtTile(dir_root=dir_root, product_id=product_id,
                                soft_id=soft_id, product_name=product_name,
                                ftile=tName, src_res=src_res)
        tileGeotags = alpGrid.get_tile_geotags(tName)

        # create temporary tile to fill with mean values
        tmpMeanSig0 = np.zeros([8000,8000], dtype=np.float32)
        tmpMeanSig0[:,:] = -9999

        for k in range(0,8000,1000): #column
            for l in range(0,8000,1000): #row
                # read sig0 and lia
                sig0ts = TOI.read_ts("SIG0_", k, l, xsize=1000, ysize=1000)
                liats = TOI.read_ts("PLIA_", k, l, xsize=1000, ysize=1000)

                valT = (np.array([d.month for d in sig0ts[0]]) > 4) & (np.array([d.month for d in sig0ts[0]]) < 11)
                valsig0 = np.array(sig0ts[1][valT,:,:])
                vallia = np.array(liats[1][valT,:,:])
                del sig0ts, liats
                sig0mask = (valsig0 == -9999) | (vallia < 2000) | (vallia > 6000)
                valsig0 = valsig0.astype(np.float32)
                valsig0[sig0mask] = np.nan

                tmp = np.nanmean(valsig0, axis=0)
                tmp[np.isinf(tmp)] = -9999
                tmpMeanSig0[l:l+1000,k:k+1000] = tmp

                del tmp, valT, valsig0, vallia, sig0mask
                #
                # for m in range(1000): #column
                #     for n in range(1000): #row
                #         valT = (np.array([d.month for d in sig0ts[0]]) > 4) & (np.array([d.month for d in sig0ts[0]]) < 11) & \
                #                (sig0ts[1][:,n,m] != -9999) & (liats[1][:,n,m] > 2000) & (liats[1][:,n,m] < 6000)
                #         tmpMeanSig0[l+n, k+m] = sig0ts[1][valT,n,m].mean()

        # save mean tile to work dir
        tmpOut = tmpMeanSig0.astype(dtype=np.int)
        outfile = workdir + '/ASAR' + tName + '.tif'

        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(outfile,
                                8000,
                                8000,
                                1,
                                gdal.GDT_Int16)

        dataset.SetGeoTransform(tileGeotags['geotransform'])
        dataset.SetProjection(tileGeotags['spatialreference'])
        dataset.GetRasterBand(1).WriteArray(tmpOut)
        dataset.FlushCache()

    # mosaic tiles
    filelist = find('*.tif', workdir)
    fileliststring = ' '.join(filelist)

    outfile = outdir + '/ASARWS_mean_sig0.tif'
    outfile_latlon = outdir + '/ASARWS_mean_sig0_latlon.tif'
    os.system('python /usr/local/bin/gdal_merge.py -o ' + outfile + ' -of GTiff -ot Int16 -n -9999 -a_nodata -9999 ' + fileliststring)
    #reproject to lat lon
    wktstring = alpGrid.get_sgrid_projection('EU')
    os.system('/usr/local/bin/gdalwarp -s_srs ' +
              wktstring +
              ' -t_srs EPSG:4326 -ot Int16 -r near -srcnodata -9999 -dstnodata -9999 -of GTiff ' +
              outfile + ' ' + outfile_latlon)

    #os.system('cp ' + outfile + ' ' + outdir)
    os.system('rm ' + workdir + '/*.tif')

    print('done')





def extr_map(dir_root, product_id, soft_id, product_name, src_res,
                     maxlon, maxlat, minlon, minlat, siglia, date, workdir, outdir):
    # initialise grid
    alpGrid = Equi7.Equi7Grid(src_res)

    # Find overlapping tiles ----
    # create search polygon
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(minlon, maxlat)
    ring.AddPoint(maxlon, maxlat)
    ring.AddPoint(maxlon, minlat)
    ring.AddPoint(minlon, minlat)
    ring.AddPoint(minlon, maxlat)
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)

    ag_tilelist = alpGrid.search_tiles(poly)
    print(ag_tilelist)

    # Extract SIG0
    #SIG0tiles = np.zeros([3000,3000,len(ag_tilelist)], dtype=np.int)

    for i in range(len(ag_tilelist)):
        tName = ag_tilelist[i]
        TOI = SgrtTile.SgrtTile(dir_root=dir_root, product_id=product_id,
                                soft_id=soft_id, product_name=product_name,
                                ftile=tName, src_res=src_res)

        # find tiles
        pattern = '.'+date+'.*'+siglia+'.*'
        regex = re.compile(pattern)
        founds = [x for x in TOI._tile_files.keys() if regex.match(x)]

        if len(founds) == 0:
            print('No files found for ' + pattern)
        else:
            for j in range(len(founds)):
                tmp = TOI.read_tile(pattern=founds[j])
                #SIG0tiles[:,:,i] = tmp
                tile_to_raster(tmp, workdir + '/' + tName + '.tif')

    outfile = workdir + '/ASAR' + date + siglia + '.tif'
    outfile_latlon = outdir + '/ASAR' + date + siglia + '.tif'

    filelist = find('*.tif', workdir)
    fileliststring = ' '.join(filelist)

    # mosaic tiles
    os.system('python /usr/local/bin/gdal_merge.py -o ' + outfile + ' -of GTiff -ot Int16 -n -9999 -a_nodata -9999 ' + fileliststring)
    #reproject to lat lon
    wktstring = alpGrid.get_sgrid_projection('EU')
    os.system('/usr/local/bin/gdalwarp -s_srs ' +
              wktstring +
              ' -t_srs EPSG:4326 -ot Int16 -r near -srcnodata -9999 -dstnodata -9999 -of GTiff ' +
              outfile + ' ' + outfile_latlon)

    #os.system('cp ' + outfile + ' ' + outdir)
    os.system('rm ' + workdir + '/*.tif')

    print('done')
