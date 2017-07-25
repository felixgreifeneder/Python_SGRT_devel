import pandas as pd
from sgrt_devels.extr_TS import extr_SIG0_LIA_ts_GEE_VV as extrts
from matplotlib import pyplot as plt
import datetime as dt
from pytesmo.time_series import anomaly

s1ts = extrts(35.35, 0.673, 2000, maskwinter=False)
#s1ts = extrts(36.0189, 0.425, 50, maskwinter=False)
s1pd = pd.Series(s1ts['130'][1]['sig0'], index=s1ts['130'][0])
s1pd.sort_index

# scaling
bcksct_min = s1pd.min()
bcksct_max = s1pd.max()
#bcksct_min = -14.8
#bcksct_max = -8.12
bcksct_rng = bcksct_max - bcksct_min

s1pd_scaled = (s1pd - bcksct_min) / bcksct_rng
s1pd_scaled = (s1pd_scaled * 35.) + 5.
# s1pd_scaled = (s1pd - bcksct_min) / bcksct_rng
# s1pd_scaled = (s1pd_scaled * 15.) + 3.
s1pd_scaled = s1pd_scaled.resample('D').mean()
s1pd_scaled.interpolate(inplace=True)

# anomalies
#bcksct_mean = s1pd_scaled.mean()
#s1pd_anom = (s1pd_scaled / bcksct_mean) - 1
climatology = anomaly.calc_climatology(s1pd_scaled, moving_avg_orig=5, moving_avg_clim=30)
s1pd_anom = anomaly.calc_anomaly(s1pd_scaled, climatology=climatology)


plt.figure(figsize=(6.3,3.7))
s1pd_scaled.plot()
plt.xlabel('Date', fontsize=10)
plt.ylabel('SMC [m3/m-3]')
plt.ylim((0,50))
plt.show()
plt.savefig('/mnt/SAT/Workspaces/GrF/02_Documents/ISRSE2017/s1ts.png', dpi=300)
plt.close()

plt.figure(figsize=(6.3,3.7))
s1pd_anom.plot()
timex = pd.date_range('30/9/2014', '30/4/2017', freq='D')
zerolist = pd.Series([0]*len(timex), index=timex)
zerolist.plot(linestyle='--', color='r')
#plt.plot(timex, [0]*len(timex), linestyle='--', color='r', marker='*')
plt.xlabel('Date', fontsize=10)
plt.ylabel('SMC anomaly')
plt.show()
plt.savefig('/mnt/SAT/Workspaces/GrF/02_Documents/ISRSE2017/s1ts_anomaly.png', dpi=300)
plt.close()
