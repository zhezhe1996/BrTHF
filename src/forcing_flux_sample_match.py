import os 
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

import rasterio
from meteo import qsea,qair

df = pd.read_pickle(os.path.join(r'.\data\pickle','OrignalBaseData.pickle'))
df = df[(df['day_date']<=pd.to_datetime('2017-12-31'))&(df['day_date']>=pd.to_datetime('1993-01-01'))]
df = df.sort_values('day_date')

forincg_names = [
    'ERA5_SLP','ERA5_SW', 'ERA5_LW','ERA5_TA','ERA5_RH',
    'SMAP_SAL','SMAP_DOS',
    'AVSIO_ADT','AVSIO_SLA',
    'OSCAR_CS',
    'OISST_TS',
    'CCMP_WS',
    'GOWR_SWH', 'GOWR_Tp',
]
flux_names = ['OHF_SHF','OHF_LHF',
              'OAFlux_SHF','OAFlux_LHF',
              'JOFURO3_SHF','JOFURO3_LHF',
              'IFREMER_SHF','IFREMER_LHF',
              'ERA5_SHF','ERA5_LHF',
              'SeaFlux_SHF','SeaFlux_LHF',
              'MERRA2_SHF','MERRA2_LHF']

df_all = pd.DataFrame(columns = list(df.columns) + forincg_names + flux_names)

for year_inter in np.arange(1993,2017,1):

    df_year = df[(df['day_date']>=pd.to_datetime(str(year_inter)+'-01-01'))&(df['day_date']<pd.to_datetime(str(year_inter+1)+'-01-01'))]
    
    df_day_all = []
    for date in df_year['day_date'].unique():
        print(date)
        df_day = df_year[df_year['day_date']== date]

        coords = [(x,y) for x,y in zip(df_day.lon,df_day.lat)]
        date_str = str(date)[:10].replace('-','')
        year_str = date_str[:4]
        month_str = date_str[4:6]
        day_str = date_str[6:]

        # forcings
        variable_names = ['SLP','SW','LW','TA','RH']
        for variable_name in variable_names:
            product_name = 'ERA5'
            sclae_type = 'Daily'
            path = os.path.join(r'.\data\tif',
                                variable_name+' Conversion',
                                year_str,month_str,
                                product_name+'_'+variable_name+'_'+sclae_type+'.A'+date_str+'.tif')

            try:
                src = rasterio.open(path)
                df_day[product_name+'_'+variable_name] = [x[0].real for x in src.sample(coords)]
            except:
                df_day[product_name+'_'+variable_name] = np.nan

        variable_names = ['SAL','DOS']
        for variable_name in variable_names:
            product_name = 'SMAP'
            sclae_type = 'Daily'
            path = os.path.join(r'.\data\tif',
                                'SAL Conversion',variable_name,
                                year_str,month_str,
                                product_name+'_'+variable_name+'_'+sclae_type+'.A'+date_str+'.tif')
            try:
                src = rasterio.open(path)
                df_day[product_name+'_'+variable_name] =  [x[0].real for x in src.sample(coords)]
            except:
                df_day[product_name+'_'+variable_name] = np.nan

        variable_names = ['ADT','SLA']
        for variable_name in variable_names:
            product_name = 'AVSIO'
            sclae_type = 'Daily'
            path = os.path.join(r'.\data\tif',
                                product_name+' Conversion',variable_name,
                                year_str,month_str,
                                product_name+'_'+variable_name+'_'+sclae_type+'.A'+date_str+'.tif')
            try:
                src = rasterio.open(path)
                df_day[product_name+'_'+variable_name] =  [x[0].real for x in src.sample(coords)]
            except:
                df_day[product_name+'_'+variable_name] = np.nan

        variable_names = ['CS']
        for variable_name in variable_names:
            product_name = 'OSCAR'
            sclae_type = 'Daily'
            path = os.path.join(r'.\data\tif',
                                product_name+' Conversion',variable_name,
                                year_str,month_str,
                                product_name+'_'+variable_name+'_'+sclae_type+'.A'+date_str+'.nc.tif')
            try:
                src = rasterio.open(path)
                df_day[product_name+'_'+variable_name] =  [x[0].real for x in src.sample(coords)]
            except:
                df_day[product_name+'_'+variable_name] = np.nan

        variable_names = ['WS']
        for variable_name in variable_names:
            product_name = 'CCMP'
            sclae_type = 'Daily'
            path = os.path.join(r'.\data\tif',
                                product_name+' Conversion',variable_name,
                                year_str,month_str,
                                product_name+'_'+variable_name+'_'+sclae_type+'.A'+date_str+'.tif')
            try:
                src = rasterio.open(path)
                df_day[product_name+'_'+variable_name] =  [x[0].real for x in src.sample(coords)]
            except:
                df_day[product_name+'_'+variable_name] = np.nan

        variable_names = ['SST']
        for variable_name in variable_names:
            product_name = 'OISST'
            sclae_type = 'Daily'
            path = os.path.join(r'.\data\tif',
                                product_name+' Conversion',variable_name,
                                year_str,month_str,
                                'NOAA'+'_'+variable_name+'_'+sclae_type+'.A'+date_str+'.tif')
            try:
                src = rasterio.open(path)
                df_day[product_name+'_'+variable_name] =  [x[0].real for x in src.sample(coords)]
            except:
                df_day[product_name+'_'+variable_name] = np.nan

        variable_names = ['SWH','Tp']
        for variable_name in variable_names:
            product_name = 'GOWR'
            sclae_type = 'Daily'
            path = os.path.join(r'.\data\tif',
                                'Global Ocean Waves Reanalysis Conversion',variable_name,
                                year_str,month_str,
                                product_name+'_'+variable_name+'_'+sclae_type+'.A'+date_str+'.tif')
            try:
                src = rasterio.open(path)
                df_day[product_name+'_'+variable_name] =  [x[0].real for x in src.sample(coords)]
            except:
                df_day[product_name+'_'+variable_name] = np.nan

        # Fluxes
        variable_names = ['LHF','SHF']
        for variable_name in variable_names:
            product_name = 'OHF'
            sclae_type = 'Daily'
            path = os.path.join(r'.\data\tif',
                                product_name+' Conversion',variable_name,
                                year_str,month_str,
                                product_name+'_'+variable_name+'_'+sclae_type+'.A '+date_str+'.tif')# 命名有问题多了空格
            try:
                src = rasterio.open(path)
                df_day[product_name+'_'+variable_name] =  [x[0].real for x in src.sample(coords)]
            except:
                df_day[product_name+'_'+variable_name] = np.nan

        variable_names = ['LHF','SHF']
        for variable_name in variable_names:
            product_name = 'OAFlux'
            sclae_type = 'Daily'
            path = os.path.join(r'.\data\tif',
                                product_name+' Conversion',variable_name,
                                year_str,month_str,
                                product_name+'_'+variable_name+'_'+sclae_type+'.A'+date_str+'.tif')
            try:
                src = rasterio.open(path)
                df_day[product_name+'_'+variable_name] =  [x[0].real for x in src.sample(coords)]
            except:
                df_day[product_name+'_'+variable_name] = np.nan
        
        variable_names = ['LHF','SHF']
        for variable_name in variable_names:
            product_name = 'JOFURO3'
            sclae_type = 'Daily'
            path = os.path.join(r'.\data\tif',
                                product_name+' Conversion',variable_name,
                                year_str,month_str,
                                product_name+'_'+variable_name+'_'+sclae_type+'.A'+date_str+'.tif')
            try:
                src = rasterio.open(path)
                df_day[product_name+'_'+variable_name] =  [x[0].real for x in src.sample(coords)]
            except:
                df_day[product_name+'_'+variable_name] = np.nan
        
        variable_names = ['LHF','SHF']
        for variable_name in variable_names:
            product_name = 'IFREMER'
            sclae_type = 'Daily'
            path = os.path.join(r'.\data\tif',
                                product_name+' Conversion',variable_name,
                                year_str,month_str,
                                product_name+'_'+variable_name+'_'+sclae_type+'.A'+date_str+'.tif')
            try:
                src = rasterio.open(path)
                df_day[product_name+'_'+variable_name] =  [x[0].real for x in src.sample(coords)]
            except:
                df_day[product_name+'_'+variable_name] = np.nan
        
        variable_names = ['LHF','SHF']
        for variable_name in variable_names:
            product_name = 'ERA5'
            sclae_type = 'Daily'
            path = os.path.join(r'.\data\tif',
                                product_name+' Conversion',variable_name,
                                year_str,month_str,
                                product_name+'_'+variable_name+'_'+sclae_type+'.A'+date_str+'.tif')
            try:
                src = rasterio.open(path)
                df_day[product_name+'_'+variable_name] =  [x[0].real for x in src.sample(coords)]
            except:
                df_day[product_name+'_'+variable_name] = np.nan
                
        variable_names = ['LHF','SHF']
        for variable_name in variable_names:
            product_name = 'SeaFlux'
            sclae_type = 'Daily'
            path = os.path.join(r'.\data\tif',
                                product_name+' Conversion',variable_name,
                                year_str,month_str,
                                product_name+'_'+variable_name+'_'+sclae_type+'.A'+date_str+'.tif')
            try:
                src = rasterio.open(path)
                df_day[product_name+'_'+variable_name] =  [x[0].real for x in src.sample(coords)]
            except:
                df_day[product_name+'_'+variable_name] = np.nan
                
        variable_names = ['LHF','SHF']
        for variable_name in variable_names:
            product_name = 'MERRA2'
            sclae_type = 'Daily'
            path = os.path.join(r'.\data\tif',
                                product_name+' Conversion',variable_name,
                                year_str,month_str,
                                product_name+'_'+variable_name+'_'+sclae_type+'.A'+date_str+'.tif')
            try:
                src = rasterio.open(path)
                df_day[product_name+'_'+variable_name] =  [x[0].real for x in src.sample(coords)]
            except:
                df_day[product_name+'_'+variable_name] = np.nan
        df_day_all.append(df_day)

    df_day_all_concat = df_day_all[0]
    for item in df_day_all[1:]:
        df_day_all_concat = pd.concat([df_day_all_concat,item],axis = 0)

    df_all = pd.concat([df_all,df_day_all_concat],axis = 0)

df_all = df.dropna(subset = flux_names+forincg_names)

df_all = df_all.reset_index()
df_all = df_all.drop(columns = ['index'])

df_all['ERA5_QA'] = [qair(item[0],item[1],item[2])[0] for item in df_all[['ERA5_TA','ERA5_SLP','ERA5_RH']].values]
df_all['OISST_QS'] = [qsea(item[0],item[1]) for item in df_all[['OISST_SST','ERA5_SLP']].values]

df_all['t_diff_OISST_ERA5'] = df_all['OISST_SST'] - df_all['ERA5_TA']
df_all['q_diff_OISST_ERA5'] = df_all['OISST_QS'] - df_all['ERA5_QA']

df_all.to_pickle(r'.\data\pickle\BaseData.pickle')

