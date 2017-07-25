__author__ = 'usergre'

import sys

sys.path.extend(['/home/usergre/winpycharm/sgrt_run',
                 '/home/usergre/winpycharm/sgrt',
                 '/home/usergre/winpycharm/Python_SGRT_devel',
                 '/home/usergre/winpycharm/ascat'])

from sgrt_devels.compile_tset import Trainingset
from sgrt_devels.compile_tset import Estimationset
import pickle
from sgrt_devels.derive_smc import extract_time_series_gee

#subsets = [36.39,-8.57,53.88,17.84]
# Subset central europe CE
subsets = [48.0, 8.0, 49.0, 9.0]

# subsets = [[36.39, -8.57, 53.88, -5.26875],
#            [36.39, -5.26876, 53.88, -1.9675],
#            [36.39, -1.9676, 53.88, 1.33375],
#            [36.39, 1.33376, 53.88, 4.635],
#            [36.39, 4.636, 53.88, 7.93625],
#            [36.39, 7.93626, 53.88, 11.2375],
#            [36.39, 11.2376, 53.88, 14.53875],
#            [36.39, 14.53875, 53.88, 17.84]]

# subsets = [[46.63,10.28,47.6,11.2],
#            [47.61, 10.28, 48.6, 11.2]]

# results = Parallel(n_jobs=8, verbose=5)(delayed(Trainingset)('/mnt/SAT4/DATA/S1_EODC/',
#                                                              '/mnt/SAT4/DATA/S1_EODC/Sentinel-1_CSAR/IWGRDH/parameters/datasets/sig0m/B0212/EQUI7_EU010M/',
#                                                              '/mnt/SAT4/DATA/S1_EODC/', '/mnt/SAT4/DATA/S1_EODC/Sentinel-1_CSAR/IWGRDH/ancillary/datasets/DEM/',
#                                                              '/mnt/SAT/Workspaces/GrF/Processing/S1ALPS/all_alps_plus_se_gee/mixed/t' + str(i) +'/',
#                                                               uselc=True,
#                                                               subgrid='EU',
#                                                               tiles=subsets[i],
#                                                               ssm_target='ASCAT',
#                                                               sig0_source='GEE') for i in range(8))

results = Trainingset('/mnt/SAT4/DATA/S1_EODC/',
                      '/mnt/SAT4/DATA/S1_EODC/Sentinel-1_CSAR/IWGRDH/parameters/datasets/sig0m/B0212/EQUI7_EU010M/',
                      '/mnt/SAT4/DATA/S1_EODC/', '/mnt/SAT4/DATA/S1_EODC/Sentinel-1_CSAR/IWGRDH/ancillary/datasets/DEM/',
                      '/mnt/SAT/Workspaces/GrF/Processing/S1ALPS/ASCAT/gee_subset_CE/',
                      uselc=True,
                      subgrid='EU',
                      tiles=subsets,
                      ssm_target='ASCAT',
                      sig0_source='GEE')
#
#pickle.dump(results.trainingdata, open(results.outpath + 'sig0lia_dict.p', 'wb'))

#model = results.train_model_alternative_linear()

#results = pickle.load(open('/mnt/SAT/Workspaces/GrF/Processing/S1ALPS/ASCAT/gee_subset_CE/mlmodel.p', 'rb'))
model = results.train_model()
#pickle.dump(results, open(results.outpath + 'mlmodel.p', 'wb'))

# es = Estimationset('/mnt/SAT4/DATA/S1_EODC/', ['E048N014T1'],
#                    '/mnt/SAT4/DATA/S1_EODC/Sentinel-1_CSAR/IWGRDH/parameters/datasets/sig0m/B0212/EQUI7_EU010M/',
#                    '/mnt/SAT4/DATA/S1_EODC/Sentinel-1_CSAR/IWGRDH/ancillary/datasets/DEM/',
#                    '/mnt/SAT/Workspaces/GrF/Processing/S1ALPS/all/pwise/',
#                    model,
#                    subgrid='EU',
#                    uselc=True)
#
# es.ssm_map()

# stations = {'Vipas2000': [4831141.21, 1516685.53],
#             'Vimes2000': [4830444.22, 1517335.96],
#             'Vimef2000': [4830996.06, 1516222.11],
#             'Vimef1500': [4836014.08, 1512440.54],
#             'Domes1500': [4874224.56, 1469179.81],
#             'Domef1500': [4874881.06, 1469495.06],
#             'Dopas2000': [4873472.41, 1463915.39],
#             'Nemef1500': [4935219.71, 1531329.21],
#             'Nemes1500': [4934388.83, 1531173.69],
#             'Domef2000': [4889902.12, 1484534.78],
#             'Domes2000': [4892088.35, 1481584.32],
#             'Nepas2000': [4944063.47, 1503903.05],
#             'Vimes1500': [4814092.66, 1512626.88],
#             'Nemef2000': [4944025.70, 1503497.29],
#             'Nemes2000': [4943880.04, 1503699.99]}

#mlmodel = pickle.load(open('/mnt/SAT/Workspaces/GrF/Processing/S1ALPS/all_alps_plus_se_gee/pwise/mlmodel_fltrd.p', 'rb'))

# for i in stations:
#     print(i)
#     extract_time_series_gee(model,
#                         '/mnt/SAT4/DATA/S1_EODC/',
#                         '/mnt/SAT/Workspaces/GrF/Processing/S1ALPS/monalisa/gee_allalps_pwise/',
#                         stations[i][1],
#                         stations[i][0],
#                         grid='Equi7',
#                         name=i)




