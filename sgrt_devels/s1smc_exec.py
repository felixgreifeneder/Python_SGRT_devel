__author__ = 'usergre'

import sys

sys.path.extend(['/raid0/Python_Devel/sgrt_run', '/raid0/Python_Devel/sgrt', '/raid0/Python_Devel/Python_SGRT_devel'])

from sgrt_devels.compile_tset import Trainingset
from sgrt_devels.compile_tset import Estimationset
import pickle
#
t = Trainingset('/mnt/SAT4/DATA/S1_EODC/',
                '/mnt/SAT4/DATA/S1_EODC/Sentinel-1_CSAR/IWGRDH/parameters/datasets/sig0m/B0212/EQUI7_AF010M/',
                '/mnt/SAT4/DATA/S1_EODC/', '/mnt/SAT4/DATA/S1_EODC/Sentinel-1_CSAR/IWGRDH/ancillary/datasets/DEM/',
                '/mnt/SAT/Workspaces/GrF/Processing/S1ALPS/israel/',
                uselc=False,
                subgrid='AF')

model = t.train_model()

# model = pickle.load(open('/mnt/SAT/Workspaces/GrF/Processing/S1ALPS/israel/mlmodel.p', 'rb'))

es = Estimationset('/mnt/SAT4/DATA/S1_EODC/', ['E069N084T1'],
                   '/mnt/SAT4/DATA/S1_EODC/Sentinel-1_CSAR/IWGRDH/parameters/datasets/sig0m/B0212/EQUI7_EU010M/',
                   '/mnt/SAT4/DATA/S1_EODC/Sentinel-1_CSAR/IWGRDH/ancillary/datasets/DEM/',
                   '/mnt/ProjectData/ECOPOTENTIAL/ISR/SOIL MOISTURE/S1_SMC/',
                   model,
                   subgrid='AF',
                   uselc=False)

es.ssm_map()




