# this script include functions to derive both soil moisture time series and maps

from sgrt_devels.compile_tset import Estimationset
import sgrt.common.grids.Equi7Grid as Equi7
import pickle

def extract_time_series(model_path, sgrt_root, out_path, lat, lon, grid, name=None):

    mlmodel = pickle.load(open(model_path, 'rb'))

    # initialise grid
    alpGrid = Equi7.Equi7Grid(10)

    # identify tile
    if grid is None:
        Equi7XY = alpGrid.lonlat2equi7xy(lon, lat)
    elif grid == 'Equi7':
        Equi7XY = ['EU', lon, lat]
    TileName = alpGrid.identfy_tile(Equi7XY[0], [Equi7XY[1], Equi7XY[2]])

    # initialise estimation set
    es = Estimationset(sgrt_root, [TileName[7:]],
                       sgrt_root+'Sentinel-1_CSAR/IWGRDH/parameters/datasets/sig0m/B0212/EQUI7_EU010M/',
                       sgrt_root+'Sentinel-1_CSAR/IWGRDH/ancillary/datasets/DEM/',
                       out_path,
                       mlmodel,
                       subgrid='EU',
                       uselc=True)

    es.ssm_ts_alternative(Equi7XY[1], Equi7XY[2], 3, name=name)