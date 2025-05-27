import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers
from tensorflow.keras import backend as K

from tensorflow.keras.regularizers import l1,l2
from sklearn.metrics import mean_squared_error

import os 
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')

import warnings
warnings.filterwarnings("ignore")

from sklearn.utils import resample

beta_max_border, beta_min_border = 5.0, -5.0

def RMSE(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
    
def BIAS(y_obs, y_sim):   
    return np.nanmean(y_obs - y_sim)

def custome_metrics1(y_true, y_pred):
    metrics = K.mean(K.square(y_pred[:,0]-y_true[:,0]))#*10
    return metrics

def custome_metrics2(y_true, y_pred):
    metrics = K.mean(K.square(y_pred[:,1]-y_true[:,1]))
    return metrics

def custome_metrics3(y_true, y_pred):
    metrics = K.mean(K.square(tf.clip_by_value(y_pred[:,0]/y_pred[:,1], 
                                               beta_min_border,
                                               beta_max_border)-y_true[:,0]/y_true[:,1]))
    return metrics

def custom_loss(y_true, y_pred):
    pred_SHF = y_pred[:, 0]
    pred_LHF = y_pred[:, 1]
    true_SHF = y_true[:, 0]
    true_LHF = y_true[:, 1]

    pred_Bowen = pred_SHF /(pred_LHF + 1e-6)  
    true_Bowen = true_SHF /(true_LHF + 1e-6)

    SHF_loss = tf.reduce_mean(tf.square(pred_SHF - true_SHF))
    LHF_loss = tf.reduce_mean(tf.square(pred_LHF - true_LHF))
    bowen_loss = tf.reduce_mean(tf.square(tf.clip_by_value(pred_Bowen, beta_min_border,beta_max_border) - true_Bowen))

    total_loss = 1 * LHF_loss + 5 * SHF_loss + 250 * bowen_loss
    
    return total_loss

def build_cnn_model(input_shape):
    inputs = layers.Input(shape=(input_shape,)) 
    gradients = layers.Input(shape=(3,))  

    x = layers.Dense(32, activation='relu', kernel_regularizer=l2(0.05))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=l2(0.05))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(16, activation='relu', kernel_regularizer=l2(0.05))(x)
    outputs = layers.Dense(2)(x)

    model = models.Model(inputs=[inputs, gradients],outputs=outputs)
    
    return model

k_fold = 10

df = pd.read_pickle(r'.\data\pickle\BaseData.pickle')

df_all_valide = pd.DataFrame()
for idx in range(k_fold)[0:]:
    
    train = df.copy()
    test = df.copy()
    
    ID_df = pd.read_pickle(os.path.join(r'.\data\pickle',str(idx).rjust(2,'0')+'.pickle'))

    train = train[~train['ID'].isin(ID_df['ID'])]
    test = test[test['ID'].isin(ID_df['ID'])]


    train_idx = [
        'ERA5_SLP','ERA5_SW', 'ERA5_LW',
        'SMAP_SAL',
        'AVSIO_ADT',
        'OSCAR_CS',
        'CCMP_WS',
        'GOWR_SWH', 'GOWR_Tp',
        't_diff_OISST_ERA5','q_diff_OISST_ERA5'
    ]
    
    pred_idx = ['C35_SHF','C35_LHF']
    
    train_X = train[train_idx]
    train_Y = train[pred_idx]
    test_X = test[train_idx]
    test_Y = test[pred_idx]
    
    input_shape = len(train_idx)  
    model = build_cnn_model(input_shape)
    
    initial_lr = 0.001
    
    def lr_schedule(epoch, lr):
        if epoch > 0 and epoch % 35 == 0:
            return lr * 0.1
        return lr
    
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr),
        loss=custom_loss,
        metrics=[custome_metrics1,
                 custome_metrics2,
                 custome_metrics3])
    
    history = model.fit(train_X, train_Y, 
                        epochs = 100,
                        batch_size= 256,
                        callbacks=[lr_scheduler],
                        validation_data = (test_X,test_Y)
                       )

    model.save(os.path.join(r'.\model','fina_lmodel_'+str(idx).rjust(2,'0')+'.h5'),save_format='h5')

    test[['BrTHF_SHF','BrTHF_LHF']] = model.predict(test_X,batch_size= 10000)
    test['BrTHF_betac'] = test['BrTHF_SHF']/test['BrTHF_LHF']

    df_all_valide = pd.concat([df_all_valide,test],axis = 0)

df_all_valide.to_pickle(r'.\data\pickle\ModelOutput.pickle')