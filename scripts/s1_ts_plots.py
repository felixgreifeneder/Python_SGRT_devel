from sgrt_devels.extr_TS import extr_SIG0_LIA_ts
from sgrt_devels.extr_TS import extr_SIG0_LIA_ts_GEE
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ee
from datetime import datetime

# extract local s1 time series

s1_data_all = extr_SIG0_LIA_ts('/mnt/SAT4/DATA/S1_EODC/', 'S1AIWGRDH', 'A0111', 'resampled',
                               10, 11.328520, 46.438419, 11, 11, pol_name=['VV', 'VH'],
                               sat_pass='A', monthmask=[1,2,3,4,5,6,7,8,9,10,11,12])

vv_extr = np.array(s1_data_all[1]['sig0'], dtype=np.float32)
vv_extr[vv_extr == -9999] = np.nan
vv_avg = np.nanmean(vv_extr, axis=(1,2))
vh_extr = np.array(s1_data_all[1]['sig02'], dtype=np.float32)
vh_extr[vh_extr == -9999] = np.nan
vh_avg = np.nanmean(vh_extr, axis=(1,2))
#vv_seris = pd.Series(data=vv_avg, index=s1_data_all[0])
#vh_series = pd.Series(data=vh_avg, index=s1_data_all[0])
#combined_series = pd.concat([vv_seris, vh_series], axis=1)
df_s1 = pd.DataFrame(data={'vv': vv_avg/100., 'vh': vh_avg/100.}, index=s1_data_all[0])

# extract GEE s1 time series

ge_all = extr_SIG0_LIA_ts_GEE(11.328520, 46.438419, 117)

df_gee = pd.DataFrame(data={'gee_vv': ge_all[1]['sig0'], 'gee_vh': ge_all[1]['sig02']}, index=ge_all[0])
df_comb = pd.concat([df_s1, df_gee], axis=1, join='outer')
df_comb = df_comb.interpolate()


plt.figure()
df_comb.plot()
#ax1 = fig.add_subplot(1,1,1)

#ax1.plot(df_comb)
#ax1.plot(df_comb['gee_vv'])
#fig.show()

plt.savefig('/mnt/SAT/Workspaces/GrF/Processing/S1ALPS/s1_ts_plots/mazia.png', dpi=600)
plt.close()


