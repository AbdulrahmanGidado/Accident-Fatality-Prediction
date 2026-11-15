
#!/usr/bin/env python
# coding: utf-8

import pickle

import pandas as pd
import numpy as np
import sklearn
import xgboost as xgb

from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline


print(f'pandas=={pd.__version__}')
print(f'numpy=={np.__version__}')
print(f'sklearn=={sklearn.__version__}')
print(f'xgboost=={xgb.__version__}')


def load_data():
    df = pd.read_csv('RTA Dataset.csv')

    df.drop([
        'Number_of_casualties',
        'Casualty_class',
        'Sex_of_casualty',
        'Age_band_of_casualty',
        'Casualty_severity',
        'Work_of_casuality',
        'Fitness_of_casuality',
        'Cause_of_accident',
        'Service_year_of_vehicle',
        'Defect_of_vehicle'
    ], axis=1, inplace=True)

    df.columns = df.columns.str.lower().str.replace(' ', '_')

    df['hour'] = pd.to_datetime(
        df['time'], format='%H:%M:%S'
    ).dt.hour

    df.drop('time', axis=1, inplace=True)

    cat_cols = df.select_dtypes(include=['object', 'str']).columns.tolist()

    for col in cat_cols:
        df[col] = (
            df[col]
            .str.replace(r'\s+', ' ', regex=True)
            .str.strip()
        )

    df['area_accident_occured'] = df[
        'area_accident_occured'
    ].replace('Rural village areasOffice areas', 'Unknown')

    df['pedestrian_movement'] = df[
        'pedestrian_movement'
    ].str.replace(
        'statioNot a Pedestrianry',
        'stationary',
        regex=False
    )


    df['fatality'] = (
        df['accident_severity'] != 'Slight Injury'
    ).astype(int)

    df.drop('accident_severity', axis=1, inplace=True)

    cat_cols = df.select_dtypes(include=['object', 'str']).columns.tolist()

    df[cat_cols] = df[cat_cols].fillna('Unknown')

    num_cols = [
        'number_of_vehicles_involved',
        'hour'
    ]

    df[num_cols] = df[num_cols].fillna(0)

    return df


def train_model(df):

    categorical = df.select_dtypes(include=['object', 'str']).columns.tolist()

    numerical = [
        'number_of_vehicles_involved',
        'hour'
    ]

    y_train = df.fatality

    train_dict = df[categorical + numerical].to_dict(
        orient='records'
    )

    pipeline = make_pipeline(
    DictVectorizer(),
    xgb.XGBClassifier(
        learning_rate=0.1,
        max_depth=12,
        min_child_weight=1,
        objective='binary:logistic',
        eval_metric='auc',
        n_estimators=100,
        n_jobs=8,
        random_state=1,
        verbosity=1
    )
        )

    pipeline.fit(train_dict, y_train)

    return pipeline


def save_model(pipeline, output_file):
    with open(output_file, 'wb') as f_out:
        pickle.dump(pipeline, f_out)


df = load_data()

pipeline = train_model(df)

save_model(pipeline, 'model.bin')

print('Model saved to model.bin')
