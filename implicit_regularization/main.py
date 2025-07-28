import os
os.environ['PYTHONIOENCODING'] = 'UTF-8'
import json
import random
import math
import numpy as np
import tensorflow as tf
from pathlib import Path
import datetime
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt


BATCH_SIZE = 32
EPOCHS = 2
N_TRAINING_DATAPOINTS = 10000
N_VAL_DATAPOINTS = 1000
N_TEST_DATAPOINTS = 2000

n_training_steps_per_epoch = math.ceil(N_TRAINING_DATAPOINTS/BATCH_SIZE)
n_val_steps_per_epoch = math.ceil(N_VAL_DATAPOINTS/BATCH_SIZE)


def generate_toy_data(n_datapoints, path_output, datapoints_per_file):

    path_output = Path(__file__).resolve().parent / path_output
    path_output.mkdir(parents=True, exist_ok=True)

    beta = [1.14, 0.31, -0.31, -0.23, -3.13]
    X = np.stack([
            np.ones((n_datapoints,)),
            np.random.normal(loc=0, scale=1, size=n_datapoints),
            np.random.normal(loc=10, scale=3, size=n_datapoints),
            np.random.normal(loc=5, scale=4, size=n_datapoints),
            np.random.normal(loc=15, scale=6, size=n_datapoints),
        ], axis=1)
    epsilon = np.random.normal(loc=0, scale=3, size=n_datapoints)
    y = X @ beta + epsilon

    toy_data = list()
    for i, (xi, yi) in enumerate(zip(X, y)):
        xiyi = {'i': f'{i:010}','x_1': xi[1],'x_2': xi[2],'x_3': xi[3],'x_4': xi[4],'y': yi}
        toy_data.append(xiyi)
        
        if (i+1) % datapoints_per_file == 0:
            with open(f'{path_output}/data_{i:010}.json', 'w') as f:
                json.dump(toy_data, f, indent=4)
            toy_data = list()


def make_data(n, f_train, f_test):
    X, y = make_regression(n_samples=n, n_features=4, n_informative=4, n_targets=1)
    X_train = X[:int(0.8*f_train*n)]
    X_val = X[int(0.8*f_train*n):-int(f_test*n)]
    X_test = X[-int(f_test*n):]

    y_train = y[:int(0.8*f_train*n)]
    y_val = y[int(0.8*f_train*n):-int(f_test*n)]
    y_test = y[-int(f_test*n):]

    return X_train, y_train, X_val, y_val, X_test, y_test


class BatchGenerator:

    def __init__(self, data_folder):
        self.data_folder = Path(data_folder)
        self.make_shuffled_datapoint_paths()


    def make_shuffled_datapoint_paths(self):    
        json_files = list(self.data_folder.glob("*.json"))
        self.datapoint_paths = list()
        for file in json_files:
            with open(file, 'r') as f:
                data = json.load(f) # this could be a memory problem if each json is too big to be fully loaded to memory. could be done line by line
                for datapoint in data:
                    self.datapoint_paths.append((file, datapoint['i']))
        random.shuffle(self.datapoint_paths)


    def generate_batch(self, batch_size=32):

        for i in range(0, len(self.datapoint_paths), batch_size):

            batch_refs = self.datapoint_paths[i: (i + batch_size)]
            X_batch, y_batch = list(), list()

            # make a dict that holds file: [list of datapoints], so we only read each file once every time we generate a batch
            files_datapoints = dict()
            for file_path, datapoint_id in batch_refs:
                files_datapoints.setdefault(file_path, []).append(datapoint_id)

            # go to each file and grab the corresponding datapoints
            for file_path, datapoint_list in files_datapoints.items():
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    X_batch.extend(
                        [np.array([datapoint['x_1'], datapoint['x_2'], datapoint['x_3'], datapoint['x_4']]) for datapoint in data if datapoint['i'] in datapoint_list]
                    )
                    y_batch.extend(datapoint['y'] for datapoint in data if datapoint['i'] in datapoint_list)

            yield np.array(X_batch, dtype=np.float32), np.array(y_batch, dtype=np.float32)


if __name__ == '__main__':

    #generate_toy_data(N_TRAINING_DATAPOINTS, 'data/train', datapoints_per_file=5)
    #generate_toy_data(N_VAL_DATAPOINTS, 'data/val', datapoints_per_file=5)
    #generate_toy_data(N_TEST_DATAPOINTS, 'data/test', datapoints_per_file=5)
    #X_train, y_train, X_val, y_val, X_test, y_test = make_data(n=100, f_train=0.7, f_test=0.3)
    #train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(BATCH_SIZE).repeat()
    #val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE).repeat()
    #test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE).repeat()

    bg_train = BatchGenerator('data/train')
    bg_test = BatchGenerator('data/test')
    bg_val = BatchGenerator('data/val')

    train_ds = tf.data.Dataset.from_generator(
        lambda: bg_train.generate_batch(batch_size=BATCH_SIZE),
        output_signature=(
            tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
            tf.TensorSpec(shape=(None, ), dtype=tf.float32),
        )
    ).repeat().prefetch(tf.data.AUTOTUNE)

    test_ds = tf.data.Dataset.from_generator(
        lambda: bg_test.generate_batch(batch_size=BATCH_SIZE),
        output_signature=(
            tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
            tf.TensorSpec(shape=(None, ), dtype=tf.float32),
        )
    )

    val_ds = tf.data.Dataset.from_generator(
        lambda: bg_val.generate_batch(batch_size=BATCH_SIZE),
        output_signature=(
            tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
            tf.TensorSpec(shape=(None, ), dtype=tf.float32),
        )
    ).repeat().prefetch(tf.data.AUTOTUNE)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(4,)),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(1),
    ])

    log_dir = f"logs/fit/" + datetime.datetime.now().strftime('%Y%m%d%H%M')
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    model.compile(optimizer='adam', loss='mse')

    model.fit(
        train_ds, 
        validation_data=val_ds, 
        validation_steps=n_val_steps_per_epoch,
        epochs=EPOCHS, 
        steps_per_epoch=n_training_steps_per_epoch,
        callbacks=[tb_callback], 
        verbose=1
    )

    print(model.count_params())

    # beta = [1.14, 0.31, -0.31, -0.23, -3.13]


    y_pred = model.predict(test_ds)
    y_test = np.concatenate([y.numpy() for X, y in test_ds])

    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    plt.savefig(f"y_test_vs_y_pred_{datetime.datetime.now().strftime('%Y%m%d%H%M')}.png")

