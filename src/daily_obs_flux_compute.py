import os 
import pandas as pd

from coare35vn import coare35vn

beta_max_border, beta_min_border = 5.0, -5.0

temp_path = r'.\data\buoy_data'
files = os.listdir(temp_path)

temp = pd.read_excel(os.path.join(temp_path,files[0]))

df_all = pd.DataFrame(columns = temp.columns+['C35_SHF','C35_LHF'])

for file in files:
    df = pd.read_excel(os.path.join(temp_path,file))

    u = df['WSPD']
    t = df['ATMP']
    rh = df['RH']
    ts = df['WTMP']
    P = df['PRES']
    SW = df['SW']
    LW = df['LW']
    zu = df['z_wspd']
    zt = df['z_tair']
    zq = df['z_qair']
    zsst = df['z_tsea']
    zi = 600
    lat = df['lat']
    jcool = 0
    A = coare35vn(u,t,rh,ts,P,Rs = SW,Rl = LW,zu = zu,zt = zt,zq = zq,lat = lat,zi = zi,jcool = jcool)

    df['C35_SHF'] = A[:,2][0]
    df['C35_LHF'] = A[:,3][0]

    df_all = pd.concat([df_all,df],axis = 0)

df_all['C35_betac'] = df_all['C35_SHF']/df_all['C35_LHF']
df_all['C35_betac'] = df_all['C35_betac'][(df_all['C35_betac'] <=beta_max_border)&(df_all['C35_betac'] >=beta_min_border)]

df_all = df_all.reset_index()
df_all = df_all.drop(columns = ['index'])

df_all.to_pickle(os.path.join(r'.\data\pickle','OrignalBaseData.pickle'))

    