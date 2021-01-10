import ast
from typing import List

import numpy as np
import pandas as pd
import wfdb
import tensorflow as tf

from classifiers.inception import build_model


def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path + f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path + f) for f in df.filename_hr]

    return np.array([signal for signal, meta in data])


path = '/Users/viachaslau_kastyniuk/Downloads/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/'
sampling_rate = 100
# sampling_rate = 500

# load and convert annotation data
Y = pd.read_csv(path + 'ptbxl_database.csv', index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))


# Load scp_statements.csv for diagnostic aggregation
agg_df = pd.read_csv(path + 'scp_statements.csv', index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]


def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(key)
            # tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))


# Apply diagnostic superclass
Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)


Y = Y[Y.diagnostic_superclass.map(lambda d: len(d)) > 0]


# Load raw signal data
Y = Y.head(1000)
X = load_raw_data(Y, sampling_rate, path)






# Split data into train and test
test_fold = 10

diagnosis = agg_df.index.values


def to_categorical(arr: List[str]) -> np.array:
    return np.array([int(x in arr) for x in diagnosis], dtype=np.int)


# Train
x_train = X[np.where(Y.strat_fold != test_fold)]
x_train = tf.cast(x_train, tf.float32)

y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass.values
y_train = [to_categorical(r) for r in y_train]
y_train = tf.cast(y_train, tf.float32)


# # Test
x_test = X[np.where(Y.strat_fold == test_fold)]
x_test = tf.cast(x_test, tf.float32)

y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass.values
y_test = [to_categorical(r) for r in y_test]
y_test = tf.cast(y_test, tf.float32)


model = build_model(x_train.shape[-2:], num_classes=y_train.shape[-1], num_modules = 6, use_residual = True)

# if batch_size is None:
batch_size = int(min(x_train.shape[0] / 10, 16))
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=50,
                    # validation_data=(x_test, y_test)
                    )
print('\nhistory dict:', history.history)

# model.save('model.hdf5')
#
# model = tf.keras.models.load_model('model.hdf5')
# model.predict(x_test, batch_size=batch_size)
