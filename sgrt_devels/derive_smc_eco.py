
import sys

sys.path.extend(['/home/usergre/winpycharm/sgrt_run', '/home/usergre/winpycharm/sgrt', '/home/usergre/winpycharm/Python_SGRT_devel'])

from sgrt_devels.compile_tset import Estimationset
from sgrt_devels.compile_tset import Trainingset
from sgrt_run.releases.B02.B02_workflow_sar_EURAC_param_retrieval import B02_workflow_sar_param_retrieval
import pickle

# Israel
# B02_workflow_sar_param_retrieval('/raid0/sgrt_config/process_israel_b02.xml')

# t = Trainingset('/mnt/SAT4/DATA/S1_EODC/',
#                 '/mnt/SAT4/DATA/S1_EODC/Sentinel-1_CSAR/IWGRDH/parameters/datasets/sig0m/B0212/EQUI7_EU010M/',
#                 '/mnt/SAT4/DATA/S1_EODC/', '/mnt/SAT4/DATA/S1_EODC/Sentinel-1_CSAR/IWGRDH/ancillary/datasets/DEM/',
#                 '/mnt/SAT/Workspaces/GrF/Processing/S1ALPS/israel/',
#                 uselc=False,
#                 subgrid='AF',
#                 tiles=['E069N085T1', 'E069N084T1'],
#                 months=[1,2,3,4,5,6,7,8,9,10,11,12])
#
# pickle.dump(t.sig0lia, open(t.outpath + 'sig0lia_dict.p', 'wb'))
#
# model = t.train_model()

model = pickle.load(open('/mnt/SAT/Workspaces/GrF/Processing/S1ALPS/israel/mlmodel.p', 'rb'))

es = Estimationset('/mnt/SAT4/DATA/S1_EODC/', ['E069N085T1', 'E069N084T1'],
                   '/mnt/SAT4/DATA/S1_EODC/Sentinel-1_CSAR/IWGRDH/parameters/datasets/sig0m/B0212/EQUI7_EU010M/',
                   '/mnt/SAT4/DATA/S1_EODC/Sentinel-1_CSAR/IWGRDH/ancillary/datasets/DEM/',
                   '/mnt/ProjectData/ECOPOTENTIAL/ISR - HARHANEGEV/SOIL MOISTURE/sentinel_1/',
                   model, subgrid="AF", uselc=False)

es.ssm_map()
#


# Portugal and Spain training
#B02_workflow_sar_param_retrieval('/raid0/sgrt_config/process_all_ecopo_b02.xml')

# t = Trainingset('/mnt/SAT4/DATA/S1_EODC/',
#                 '/mnt/SAT4/DATA/S1_EODC/Sentinel-1_CSAR/IWGRDH/parameters/datasets/sig0m/B0212/EQUI7_EU010M/',
#                 '/mnt/SAT4/DATA/S1_EODC/', '/mnt/SAT4/DATA/S1_EODC/Sentinel-1_CSAR/IWGRDH/ancillary/datasets/DEM/',
#                 '/mnt/SAT/Workspaces/GrF/Processing/S1ALPS/PTSP/',
#                 uselc=True,
#                 subgrid='EU',
#                 tiles=['E030N009T1', 'E030N010T1', 'E030N011T1',
#                        'E031N009T1', 'E031N010T1', 'E031N011T1',
#                        'E032N009T1', 'E032N010T1', 'E032N011T1', 'E032N014T1',
#                        'E034N007T1', 'E034N008T1'],
#                 months=[5,6,7,8,9])
#
# model = t.train_model()
#
# # Porugal
#
#
model = pickle.load(open('/mnt/SAT/Workspaces/GrF/Processing/S1ALPS/PTSP/mlmodel.p', 'rb'))
#

es = Estimationset('/mnt/SAT4/DATA/S1_EODC/', ['E032N014T1'],
                   '/mnt/SAT4/DATA/S1_EODC/Sentinel-1_CSAR/IWGRDH/parameters/datasets/sig0m/B0212/EQUI7_EU010M/',
                   '/mnt/SAT4/DATA/S1_EODC/Sentinel-1_CSAR/IWGRDH/ancillary/datasets/DEM/',
                   '/mnt/ProjectData/ECOPOTENTIAL/PT - PENEDA/SOIL MOISTURE/sentinel_1/',
                   model)

es.ssm_map()

es = Estimationset('/mnt/SAT4/DATA/S1_EODC/', ['E030N009T1', 'E030N010T1', 'E030N011T1',
                                               'E031N009T1', 'E031N010T1', 'E031N011T1',
                                               'E032N009T1', 'E032N010T1', 'E032N011T1'],
                   '/mnt/SAT4/DATA/S1_EODC/Sentinel-1_CSAR/IWGRDH/parameters/datasets/sig0m/B0212/EQUI7_EU010M/',
                   '/mnt/SAT4/DATA/S1_EODC/Sentinel-1_CSAR/IWGRDH/ancillary/datasets/DEM/',
                   '/mnt/ProjectData/ECOPOTENTIAL/PT - MONTADO/SOIL MOISTURE/sentinel_1/',
                   model)

es.ssm_map()

# Sierra Nevada

model = pickle.load(open('/mnt/SAT/Workspaces/GrF/Processing/S1ALPS/PTSP/mlmodel.p', 'rb'))
#

es = Estimationset('/mnt/SAT4/DATA/S1_EODC/', ['E034N007T1'],
                   '/mnt/SAT4/DATA/S1_EODC/Sentinel-1_CSAR/IWGRDH/parameters/datasets/sig0m/B0212/EQUI7_EU010M/',
                   '/mnt/SAT4/DATA/S1_EODC/Sentinel-1_CSAR/IWGRDH/ancillary/datasets/DEM/',
                   '/mnt/ProjectData/ECOPOTENTIAL/E - SIERRA NEVADA/SOIL MOISTURE/sentinel_1/',
                   model)

es.ssm_map()

es = Estimationset('/mnt/SAT4/DATA/S1_EODC/', ['E034N008T1'],
                   '/mnt/SAT4/DATA/S1_EODC/Sentinel-1_CSAR/IWGRDH/parameters/datasets/sig0m/B0212/EQUI7_EU010M/',
                   '/mnt/SAT4/DATA/S1_EODC/Sentinel-1_CSAR/IWGRDH/ancillary/datasets/DEM/',
                   '/mnt/ProjectData/ECOPOTENTIAL/E - SIERRA NEVADA/SOIL MOISTURE/sentinel_1/',
                   model)

es.ssm_map()

# -----------------------------------------------------

# t = Trainingset('/mnt/SAT4/DATA/S1_EODC/',
#                 '/mnt/SAT4/DATA/S1_EODC/Sentinel-1_CSAR/IWGRDH/parameters/datasets/sig0m/B0212/EQUI7_EU010M/',
#                 '/mnt/SAT4/DATA/S1_EODC/', '/mnt/SAT4/DATA/S1_EODC/Sentinel-1_CSAR/IWGRDH/ancillary/datasets/DEM/',
#                 '/mnt/SAT/Workspaces/GrF/Processing/S1ALPS/AT/',
#                 uselc=True,
#                 subgrid='EU',
#                 tiles=['E051N015T1'],
#                 months=[5,6,7,8,9])
#
# model = t.train_model()

# Northern Limestone

model = pickle.load(open('/mnt/SAT/Workspaces/GrF/Processing/S1ALPS/AT/mlmodel.p', 'rb'))
#

es = Estimationset('/mnt/SAT4/DATA/S1_EODC/', ['E051N015T1'],
                   '/mnt/SAT4/DATA/S1_EODC/Sentinel-1_CSAR/IWGRDH/parameters/datasets/sig0m/B0212/EQUI7_EU010M/',
                   '/mnt/SAT4/DATA/S1_EODC/Sentinel-1_CSAR/IWGRDH/ancillary/datasets/DEM/',
                   '/mnt/ProjectData/ECOPOTENTIAL/AU - LIMESTONE/SOIL MOISTURE/sentinel_1/',
                   model)

es.ssm_map()
# ------------------------------------------------

# Gran Paradiso

# t = Trainingset('/mnt/SAT4/DATA/S1_EODC/',
#                 '/mnt/SAT4/DATA/S1_EODC/Sentinel-1_CSAR/IWGRDH/parameters/datasets/sig0m/B0212/EQUI7_EU010M/',
#                 '/mnt/SAT4/DATA/S1_EODC/', '/mnt/SAT4/DATA/S1_EODC/Sentinel-1_CSAR/IWGRDH/ancillary/datasets/DEM/',
#                 '/mnt/SAT/Workspaces/GrF/Processing/S1ALPS/IT/',
#                 uselc=True,
#                 subgrid='EU',
#                 tiles=['E045N014T1'],
#                 months=[5,6,7,8,9])
#
# model = t.train_model()

model = pickle.load(open('/mnt/SAT/Workspaces/GrF/Processing/S1ALPS/IT/mlmodel.p', 'rb'))


es = Estimationset('/mnt/SAT4/DATA/S1_EODC/', ['E045N014T1'],
                   '/mnt/SAT4/DATA/S1_EODC/Sentinel-1_CSAR/IWGRDH/parameters/datasets/sig0m/B0212/EQUI7_EU010M/',
                   '/mnt/SAT4/DATA/S1_EODC/Sentinel-1_CSAR/IWGRDH/ancillary/datasets/DEM/',
                   '/mnt/ProjectData/ECOPOTENTIAL/IT - GRAN PARADISO/SOIL MOISTURE/sentinel_1/',
                   model)

es.ssm_map()
# ---------------------------------------------
