__author__ = 'usergre'

import sys

sys.path.extend(['/raid0/Python_Devel/sgrt_run', '/raid0/Python_Devel/sgrt', '/raid0/Python_Devel/Python_SGRT_devel'])

from sgrt_devels.compile_tset import Trainingset
from sgrt_devels.compile_tset import Estimationset
import pickle
#
# t = Trainingset('/mnt/SAT4/DATA/S1_EODC/',
#                 '/mnt/SAT4/DATA/S1_EODC/Sentinel-1_CSAR/IWGRDH/parameters/datasets/sig0m/B0212/EQUI7_EU010M/',
#                 '/mnt/SAT4/DATA/S1_EODC/', '/mnt/SAT4/DATA/S1_EODC/Sentinel-1_CSAR/IWGRDH/ancillary/datasets/DEM/',
#                 '/mnt/SAT/Workspaces/GrF/Processing/S1ALPS/all_all/')
#
# model = t.train_model()
model = pickle.load(open('/mnt/SAT/Workspaces/GrF/Processing/S1ALPS/all_all/mlmodel.p', 'rb'))

es = Estimationset('/mnt/SAT4/DATA/S1_EODC/', ['E048N015T1'],
               '/mnt/SAT4/DATA/S1_EODC/Sentinel-1_CSAR/IWGRDH/parameters/datasets/sig0m/B0212/EQUI7_EU010M/',
               '/mnt/SAT4/DATA/S1_EODC/Sentinel-1_CSAR/IWGRDH/ancillary/datasets/DEM/',
               '/mnt/SAT/Workspaces/GrF/Processing/S1ALPS/all_all/',
               model)
# #
# # #extr ts
#i1
#es.ssm_ts(4811375, 1512390, 10)
#i3
#es.ssm_ts(4814294, 1512117, 10)
#p2
#es.ssm_ts(4814464, 1512364, 10)
#p3
#es.ssm_ts(4814518, 1512426, 10)
es.ssm_map('D20150704_170644')
#e051n015
#es.ssm_map('D20150718_165032')
