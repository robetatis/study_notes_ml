import json
import random
import math
import numpy as np
#import tensorflow as tf
from pathlib import Path
from collections import deque
import matplotlib.pyplot as plt


BATCH_SIZE = 32
EPOCHS = 5

def generate_toy_data(n_datapoints, path_output):

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
        
        if (i+1) % 100 == 0:
            with open(f'{path_output}/data_{i:010}.json', 'w') as f:
                json.dump(toy_data, f, indent=4)
            toy_data = list()


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
        self.n_datapoints = len(self.datapoint_paths)
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

    #generate_toy_data(10000, 'data/train')
    #generate_toy_data(3000, 'data/val')
    #generate_toy_data(1000, 'data/test')

    bg_train = BatchGenerator('data/train')
    bg_test = BatchGenerator('data/test')
    bg_val = BatchGenerator('data/val')

    train_ds = tf.data.Dataset.from_generator(
        lambda: bg_train.generate_batch(batch_size=32),
        output_signature=(
            tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
            tf.TensorSpec(shape=(None, ), dtype=tf.float32),
        )
    ).repeat().prefetch(tf.data.AUTOTUNE)

    test_ds = tf.data.Dataset.from_generator(
        lambda: bg_test.generate_batch(batch_size=32),
        output_signature=(
            tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
            tf.TensorSpec(shape=(None, ), dtype=tf.float32),
        )
    ).repeat().prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_generator(
        lambda: bg_val.generate_batch(batch_size=32),
        output_signature=(
            tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
            tf.TensorSpec(shape=(None, ), dtype=tf.float32),
        )
    ).repeat().prefetch(tf.data.AUTOTUNE)


    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(4,)),
        tf.keras.layers.Dense(1),
    ])

    tb_callback = tf.keras.callbacks.TensorBoard(log_dir='logs/', histogram_freq=1)

    model.compile(optimizer='adam', loss='mse', metrics=['mse'])

    #steps_per_epoch = math.ceil(bg_train.n_datapoints/BATCH_SIZE)

    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[tb_callback], verbose=1)


