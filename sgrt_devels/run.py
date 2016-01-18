__author__ = 'usergre'

import devel_fgreifen2.SVR as tSVR
import devel_fgreifen2.derive_SMC as dSMC


def execBatch():

    tSVR.extract_params('/mnt/SAT/Workspaces/GrF/01_Data/ERA/netcdf-atls05-a562cefde8a29a7288fa0b8b7f9413f7-Uyu8oE.nc',
                       '/raid0/SVR_training/random_points_noforest.csv',
                       '/mnt/HiResAlp_ASARWS_Output',
                       '/mnt/SAT/Workspaces/GrF/01_Data/ANCILLARY/CORINE/g100_06_alps_latlon.tif',
                       '/mnt/SAT/Workspaces/GrF/01_Data/ANCILLARY/DEM/SRTM/srtm_alps_clipped.tif',
                       '/mnt/SAT/Workspaces/GrF/Processing/ASAR_ALPS/ERA100_AVG10')

    tSVR.merge_params('/mnt/SAT/Workspaces/GrF/Processing/ASAR_ALPS/ERA100_AVG10',
                      '/mnt/SAT/Workspaces/GrF/Processing/ASAR_ALPS/ERA100_AVG10')

    tSVR.train_SVR('/mnt/SAT/Workspaces/GrF/Processing/ASAR_ALPS/ERA100_AVG10/comtable.npz',
                   '/mnt/SAT/Workspaces/GrF/Processing/ASAR_ALPS/ERA100_AVG10')

    # dSMC.smc_map('/mnt/SAT/Workspaces/GrF/Processing/ASAR_ALPS/ASAR20091020SIG.tif',
    #              '/mnt/SAT/Workspaces/GrF/Processing/ASAR_ALPS/ASAR20091020LIA.tif',
    #              '/mnt/SAT/Workspaces/GrF/01_Data/ANCILLARY/DEM/SRTM/srtm_alps_clipped.tif',
    #              '/mnt/SAT/Workspaces/GrF/01_Data/ANCILLARY/NDVI/MCD43A4_005_2009_10_16.NDVI.tif',
    #              '/mnt/SAT/Workspaces/GrF/01_Data/ANCILLARY/CORINE/g100_06_alps_latlon.tif',
    #              '/mnt/SAT/Workspaces/GrF/Processing/ASAR_ALPS/ERA100_AVG10/grad_model.pkl',
    #              '/raid0/ASARALPS_work',
    #              '/mnt/SAT/Workspaces/GrF/Processing/ASAR_ALPS/ERA100_AVG10')