#!/usr/bin/env python3
import pandas as pd
import pickle
import os
from sklearn.externals import joblib
from taskB_ml_baseline_utils import FeatureMaker

if __name__ == '__main__':
    DATA_FILE = os.environ.get('INPUT', 'train.csv')
    PREDICTION_FILE = os.environ.get('OUTPUT', 'data/result.csv')

    # load data
    df = pd.DataFrame.from_csv(DATA_FILE, sep=',', index_col=None)
    df = df[['paragraph_id', 'question_id', 'paragraph', 'question']]

    # load model and other dependent files
    model = joblib.load('model.pkl')
    with open("idfs.pkl", "rb") as f:
        idfs = pickle.load(f)

    # build features
    maker = FeatureMaker(df, idfs)
    X_test = maker.make()

    # predict
    X_test['prediction'] = model.predict(X_test[X_test.columns.difference(['question_id', 'target', 'paragraph_id'])])

    # prepare result
    result = X_test.groupby(['paragraph_id', 'question_id'])['prediction'].idxmax().apply(lambda x: maker.get_span(x)).rename("answer")

    # save to specific file
    result.to_csv(PREDICTION_FILE, header=True)
