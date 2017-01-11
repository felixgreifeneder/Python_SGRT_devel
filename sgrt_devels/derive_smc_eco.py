
import sys

sys.path.extend(['/raid0/Python_Devel/sgrt_run', '/raid0/Python_Devel/sgrt', '/raid0/Python_Devel/Python_SGRT_devel'])

from sgrt_devels.compile_tset import Estimationset
import pickle


# Porugal


model = pickle.load(open('/mnt/SAT/Workspaces/GrF/Processing/S1ALPS/PT/mlmodel.p', 'rb'))
#

es = Estimationset('/mnt/SAT4/DATA/S1_EODC/', ['E032N014T1'],
                   '/mnt/SAT4/DATA/S1_EODC/Sentinel-1_CSAR/IWGRDH/parameters/datasets/sig0m/B0212/EQUI7_EU010M/',
                   '/mnt/SAT4/DATA/S1_EODC/Sentinel-1_CSAR/IWGRDH/ancillary/datasets/DEM/',
                   '/mnt/ProjectData/ECOPOTENTIAL/PT - MONTADO PENEDA/SOIL MOISTURE/S1_SMC/',
                   model)

es.ssm_map()


# Northern Limestone

model = pickle.load(open('/mnt/SAT/Workspaces/GrF/Processing/S1ALPS/AT/mlmodel.p', 'rb'))
#

es = Estimationset('/mnt/SAT4/DATA/S1_EODC/', ['E051N015T1'],
                   '/mnt/SAT4/DATA/S1_EODC/Sentinel-1_CSAR/IWGRDH/parameters/datasets/sig0m/B0212/EQUI7_EU010M/',
                   '/mnt/SAT4/DATA/S1_EODC/Sentinel-1_CSAR/IWGRDH/ancillary/datasets/DEM/',
                   '/mnt/ProjectData/ECOPOTENTIAL/AT - NORTHERN LIMESTONE/SOIL MOISTURE/S1_SMC/',
                   model)

es.ssm_map()
# ------------------------------------------------

# Gran Paradiso

model = pickle.load(open('/mnt/SAT/Workspaces/GrF/Processing/S1ALPS/IT/mlmodel.p', 'rb'))
#

es = Estimationset('/mnt/SAT4/DATA/S1_EODC/', ['E045N014T1'],
                   '/mnt/SAT4/DATA/S1_EODC/Sentinel-1_CSAR/IWGRDH/parameters/datasets/sig0m/B0212/EQUI7_EU010M/',
                   '/mnt/SAT4/DATA/S1_EODC/Sentinel-1_CSAR/IWGRDH/ancillary/datasets/DEM/',
                   '/mnt/ProjectData/ECOPOTENTIAL/IT - GRAN PARADISO/SOIL MOISTURE/S1_SMC/',
                   model)

es.ssm_map()
# ---------------------------------------------

# Sierra Nevada

model = pickle.load(open('/mnt/SAT/Workspaces/GrF/Processing/S1ALPS/SP/mlmodel.p', 'rb'))
#

es = Estimationset('/mnt/SAT4/DATA/S1_EODC/', ['E034N007T1'],
                   '/mnt/SAT4/DATA/S1_EODC/Sentinel-1_CSAR/IWGRDH/parameters/datasets/sig0m/B0212/EQUI7_EU010M/',
                   '/mnt/SAT4/DATA/S1_EODC/Sentinel-1_CSAR/IWGRDH/ancillary/datasets/DEM/',
                   '/mnt/ProjectData/ECOPOTENTIAL/E - SIERRA NEVADA/SOIL MOISTURE/S1_SMC/',
                   model)

es.ssm_map()

es = Estimationset('/mnt/SAT4/DATA/S1_EODC/', ['E034N008T1'],
                   '/mnt/SAT4/DATA/S1_EODC/Sentinel-1_CSAR/IWGRDH/parameters/datasets/sig0m/B0212/EQUI7_EU010M/',
                   '/mnt/SAT4/DATA/S1_EODC/Sentinel-1_CSAR/IWGRDH/ancillary/datasets/DEM/',
                   '/mnt/ProjectData/ECOPOTENTIAL/E - SIERRA NEVADA/SOIL MOISTURE/S1_SMC/',
                   model)

es.ssm_map()