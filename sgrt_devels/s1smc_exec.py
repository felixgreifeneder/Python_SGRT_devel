__author__ = 'usergre'

import sys

sys.path.extend(['/home/usergre/winpycharm/sgrt_run', '/home/usergre/winpycharm/sgrt', '/home/usergre/winpycharm/Python_SGRT_devel'])

from sgrt_devels.compile_tset import Trainingset
from sgrt_devels.compile_tset import Estimationset
import pickle

t = Trainingset('/mnt/SAT4/DATA/S1_EODC/',
                '/mnt/SAT4/DATA/S1_EODC/Sentinel-1_CSAR/IWGRDH/parameters/datasets/sig0m/B0212/EQUI7_EU010M/',
                '/mnt/SAT4/DATA/S1_EODC/', '/mnt/SAT4/DATA/S1_EODC/Sentinel-1_CSAR/IWGRDH/ancillary/datasets/DEM/',
                '/mnt/SAT/Workspaces/GrF/Processing/S1ALPS/all/',
                uselc=True,
                subgrid='EU')

model = t.train_model_alternative()

# model = pickle.load(open('/mnt/SAT/Workspaces/GrF/Processing/S1ALPS/all/mlmodel.p', 'rb'))
#
es = Estimationset('/mnt/SAT4/DATA/S1_EODC/', ['E048N014T1', 'E048N015T1', 'E049N014T1', 'E049N015T1'],
                   '/mnt/SAT4/DATA/S1_EODC/Sentinel-1_CSAR/IWGRDH/parameters/datasets/sig0m/B0212/EQUI7_EU010M/',
                   '/mnt/SAT4/DATA/S1_EODC/Sentinel-1_CSAR/IWGRDH/ancillary/datasets/DEM/',
                   '/mnt/SAT/Workspaces/GrF/Processing/S1ALPS/all/',
                   model,
                   subgrid='EU',
                   uselc=True)

es.ssm_map()




