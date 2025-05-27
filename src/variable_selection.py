import os

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor

df = pd.read_pickle(r'.\data\pickle\BaseData.pickle')

train_idx = [
    'ERA5_SLP','ERA5_SW', 'ERA5_LW','ERA5_TA','ERA5_QA',
    'SMAP_SAL','SMAP_DOS',
    'AVSIO_ADT','AVSIO_SLA',
    'OSCAR_CS',
    'OISST_TS','OISST_QS',
    'CCMP_WS',
    'GOWR_SWH', 'GOWR_Tp',
    't_diff_OISST_ERA5','q_diff_OISST_ERA5'
]

pred_idx = ['C35_SHF','C35_LHF']

for item in pred_idx:

    X = df[train_idx]
    y = df[item]

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    feature_importances_regressor = rf.feature_importances_

    importance_df_regressor = pd.DataFrame({
        'Feature': train_idx,
        'Importance': feature_importances_regressor
    })

    importance_df_regressor = importance_df_regressor.sort_values(by='Importance', ascending=False)

    importance_df_regressor.to_excel(os.path.join(r'.\data\xls',item.replace('C35','')+'VariableImportance.xlsx'))
