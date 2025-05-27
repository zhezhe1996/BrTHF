import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers
from tensorflow.keras import backend as K

from tensorflow.keras.regularizers import l1,l2
from sklearn.metrics import mean_squared_error

import os 
import numpy as np
import pandas as pd
plt.rc('font',family='Times New Roman')

import warnings
warnings.filterwarnings("ignore")

import rasterio

from meteo import qair,qsea

from tensorflow.keras.models import load_model

from model_construction import custom_loss,custome_metrics1,custome_metrics2,custome_metrics3

width,height = 720,1440

BrTHF = load_model(os.path.join(r'.\model',
                    'Model_for_BrTHF.h5'),
                   custom_objects = {'custom_loss': custom_loss,
                                     'custome_metrics1':custome_metrics1,
                                     'custome_metrics2':custome_metrics2,
                                     'custome_metrics3':custome_metrics2
                                    })

dates = pd.date_range('1993-01-01','2017-12-31')

for date in dates:
    
    img_list = []

    date_str = str(date)[:10].replace('-','')
    year_str = date_str[:4]
    month_str = date_str[4:6]
    day_str = date_str[6:]

    variable_names = ['SLP','SW','LW','TA','RH']
    for variable_name in variable_names:
        product_name = 'ERA5'
        sclae_type = 'Daily'
        path = os.path.join(r'.\data\tif',
                            variable_name+' Conversion',
                            year_str,month_str,
                            product_name+'_'+variable_name+'_'+sclae_type+'.A'+date_str+'.tif')
        src = rasterio.open(path)
        img = src.read(1)
        img_list.append(img)

    variable_names = ['SAL','DOS']
    for variable_name in variable_names:
        product_name = 'SMAP'
        sclae_type = 'Daily'
        path = os.path.join(r'.\data\tif',
                            'SAL Conversion',variable_name,
                            year_str,month_str,
                            product_name+'_'+variable_name+'_'+sclae_type+'.A'+date_str+'.tif')
        src = rasterio.open(path)
        img = src.read(1)
        img_list.append(img)

    variable_names = ['ADT','SLA']
    for variable_name in variable_names:
        product_name = 'AVSIO'
        sclae_type = 'Daily'
        path = os.path.join(r'.\data\tif',
                            product_name+' Conversion',variable_name,
                            year_str,month_str,
                            product_name+'_'+variable_name+'_'+sclae_type+'.A'+date_str+'.tif')
        src = rasterio.open(path)
        img = src.read(1)
        img_list.append(img)

    variable_names = ['CS']
    
    for variable_name in variable_names:
        product_name = 'OSCAR'
        sclae_type = 'Daily'
        path = os.path.join(r'.\data\tif',
                            product_name+' Conversion',variable_name,
                            year_str,month_str,
                            product_name+'_'+variable_name+'_'+sclae_type+'.A'+date_str+'.nc.tif')
        src = rasterio.open(path)
        img = src.read(1)
        img_list.append(img)

    variable_names = ['WS']
    
    for variable_name in variable_names:
        product_name = 'CCMP'
        sclae_type = 'Daily'
        path = os.path.join(r'.\data\tif',
                            product_name+' Conversion',variable_name,
                            year_str,month_str,
                            product_name+'_'+variable_name+'_'+sclae_type+'.A'+date_str+'.tif')
        src = rasterio.open(path)
        img = src.read(1)
        img_list.append(img)

    variable_names = ['SST']
    for variable_name in variable_names:
        product_name = 'OISST'
        sclae_type = 'Daily'
        path = os.path.join(r'.\data\tif',
                            product_name+' Conversion',variable_name,
                            year_str,month_str,
                            'NOAA'+'_'+variable_name+'_'+sclae_type+'.A'+date_str+'.tif')
        src = rasterio.open(path)
        img = src.read(1)
        img_list.append(img)

    variable_names = ['SWH','Tp']
    for variable_name in variable_names:
        product_name = 'GOWR'
        sclae_type = 'Daily'
        path = os.path.join(r'.\data',
                            'Global Ocean Waves Reanalysis Conversion',variable_name,
                            year_str,month_str,
                            product_name+'_'+variable_name+'_'+sclae_type+'.A'+date_str+'.tif')
        src = rasterio.open(path)
        img = src.read(1)
        img_list.append(img)

    forcing_array = np.array(img_list)
    forcing_array_re = forcing_array.reshape(len(forcing_array),-1).T
    forcing_array_re_nona = forcing_array_re[~np.isnan(forcing_array_re).any(axis = 1)]
    
    forcing_names = ['ERA5_SLP', 'ERA5_SW', 'ERA5_LW', 'ERA5_TA','ERA5_RH',
                     'SMAP_SAL','SMAP_DOS',
                     'AVSIO_ADT','AVSIO_SLA'
                     'OSCAR_CS',
                     'CCMP_WS', 
                     'GOWR_SWH', 'GOWR_Tp'
                    ]

    forcing_df = pd.DataFrame(forcing_array_re_nona,columns = forcing_names)

    forcing_df['ERA5_QA'] = [qair(item[0],item[1],item[2])[0] for item in forcing_df[['ERA5_TA','ERA5_SLP','ERA5_RH']].values]
    forcing_df['OISST_QS'] = [qsea(item[0],item[1]) for item in forcing_df[['OISST_SST','ERA5_SLP']].values]
    
    forcing_df['OISST_QS'] = qsea(forcing_df['OISST_SST'],forcing_df['ERA5_SLP'])
    
    forcing_df['t_diff_OISST_ERA5'] = forcing_df['OISST_SST'] - forcing_df['ERA5_TA'] 
    forcing_df['q_diff_OISST_ERA5'] = forcing_df['OISST_QS'] - forcing_df['ERA5_QA'] 

    train_idx = [
        'ERA5_SLP','ERA5_SW', 'ERA5_LW',
        'SMAP_SAL',
        'AVSIO_ADT',
        'OSCAR_CS',
        'CCMP_WS', 
        'GOWR_SWH', 'GOWR_Tp',
        't_diff_OISST_ERA5','q_diff_OISST_ERA5'
    ]

    test_X = forcing_df[train_idx]

    forcing_df[['BrTHF_SHF','BrTHF_LHF']] = BrTHF.predict(test_X,batch_size = 16384)
    forcing_df['BrTHF_betac'] = forcing_df['BrTHF_SHF']/forcing_df['BrTHF_LHF']

    temp = np.zeros((width,height),dtype = float)
    temp[:] = np.nan
    temp_re = temp.reshape(-1)
    temp_re[~np.isnan(forcing_array_re).any(axis = 1)] = forcing_df['BrTHF_SHF']
    SHF = temp_re.reshape(width,height)

    temp = np.zeros((width,height),dtype = float)
    temp[:] = np.nan
    temp_re = temp.reshape(-1)
    temp_re[~np.isnan(forcing_array_re).any(axis = 1)] = forcing_df['BrTHF_LHF']
    LHF = temp_re.reshape(width,height)

    out_meta = src.meta.copy()
    out_meta.update({'driver':'GTiff',
                   'width':width,
                   'height':height,
                   'count':1,
                   'dtype':'float32',
                   'crs':out_meta['crs'],
                   'transform':out_meta['transform'],
                   'nodata':np.nan,
                   'compress':'lzw'})

    product_name = 'BrTHF'
    sclae_type = 'Daily'

    variable_name = 'SHF'
    save_path = os.path.join(r'.\data\tif',
                        product_name+' Conversion',variable_name,
                        year_str,month_str)

    filename = product_name+'_'+variable_name+'_'+sclae_type+'.A'+date_str+'.tif'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_file_path = os.path.join(save_path,filename)

    with rasterio.open(fp=save_file_path,
                 mode='w',**out_meta) as dst:
                 dst.write(SHF, 1) 

    variable_name = 'LHF'
    save_path = os.path.join(r'.\data\tif',
                        product_name+' Conversion',variable_name,
                        year_str,month_str)

    filename = product_name+'_'+variable_name+'_'+sclae_type+'.A'+date_str+'.tif'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_file_path = os.path.join(save_path,filename)

    with rasterio.open(fp=save_file_path, 
                 mode='w',**out_meta) as dst:
                 dst.write(LHF, 1) 