import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.api import VAR
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler


def load_data(path):
    return pd.read_csv(
        path, 
        parse_dates=['start_time'], 
        index_col='start_time', 
        usecols=[
            'start_time', 
            'start_lat', 
            'start_lng', 
        ]
    )


def summarize_data(df):
    print(f"Total no. events = {df.shape[0]:,}")
    print(f"Time range = {df.index.min().strftime('%Y-%m-%d %H:%M')} - {df.index.max().strftime('%Y-%m-%d %H:%M')}")


def build_time_components(df):

    df['hour_of_day'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek # (Monday=0, Sunday=6)
    df['week_of_year'] = df.index.isocalendar().week
    df['month_of_year'] = df.index.month
    df['day_of_month'] = df.index.day


def summary_plots(df):

    build_time_components(df)
    
    fig, ax = plt.subplots(1, 4, figsize=(10, 3))

    ax[0].set_title('hour_of_day')
    ax[0].set_ylabel('n_rides')
    ax[0].hist(df['hour_of_day'], bins=24, range=(0, 24), edgecolor='black')
    ax[0].set_xticks(np.arange(0, 25, 2))
    ax[0].set_xticklabels(np.arange(0, 25, 2))

    ax[1].set_title('day_of_week')
    ax[1].set_ylabel('n_rides')
    ax[1].hist(df['day_of_week'], bins=7, range=(0, 7), edgecolor='black')
    ax[1].set_xticks(np.arange(0, 7, 1) + 0.5)
    ax[1].set_xticklabels(['Mo', 'Tu', 'We', 'Th', 'Fr.', 'Sa.', 'Su.'])

    ax[2].set_title('week_of_year')
    ax[2].set_ylabel('n_rides')
    ax[2].hist(df['week_of_year'], bins=5, range=(9, 14), edgecolor='black')
    ax[2].set_xticks(np.arange(9, 14, 1) + 0.5)
    ax[2].set_xticklabels(np.arange(9, 14, 1))

    sample_size = 1000 # take sub-sample for 2D kernel-density plot
    indices = np.random.choice(df.shape[0], size=sample_size, replace=False)
    df_sample = df.iloc[indices, :]

    contours = sns.kdeplot(
        x=df_sample['start_lng'], y=df_sample['start_lat'],
        fill=False, cmap="coolwarm", linewidths=1, thresh=0.0, levels=20,
        ax=ax[3]
    )
    ax[3].scatter(x=df['start_lng'], y=df['start_lat'], s=0.1, c='gray')

    plt.tight_layout()
    plt.show()


def map_hotspot(df):

    fig, ax = plt.subplots()

    ax.set_title('Event density')
    ax.scatter(x=df['start_lng'], y=df['start_lat'], s=0.1, c='gray')
    
    sample_size = 1000 # take sub-sample for 2D kernel-density plot
    indices = np.random.choice(df.shape[0], size=sample_size, replace=False)
    df_sample = df.iloc[indices, :]
    contours = sns.kdeplot(
        x=df_sample['start_lng'], y=df_sample['start_lat'],
        fill=False, cmap="coolwarm", linewidths=1, thresh=0.0, levels=40, 
        ax=ax
    )

    ax.set_xlim(24.6, 24.85)
    ax.set_ylim(59.40, 59.47)
    
    plt.tight_layout()
    plt.show()


def grid_data(df, nrow, ncol, tbin):

    df['time_bin'] = df.index.floor(tbin)

    lat_bins = np.linspace(df['start_lat'].min(), df['start_lat'].max(), num= nrow + 1)
    lng_bins = np.linspace(df['start_lng'].min(), df['start_lng'].max(), num= ncol + 1)

    df['lat_bin'] = pd.cut(df['start_lat'], bins=lat_bins)
    df['lng_bin'] = pd.cut(df['start_lng'], bins=lng_bins)

    gridded_data = df \
        .groupby(['time_bin', 'lat_bin', 'lng_bin'], observed=False) \
        .agg(n_events=('start_lat', 'count')) \
        .reset_index()

    return gridded_data


def widen_grid_data(gridded_data):
    gridded_data_wide = gridded_data.pivot(index='time_bin', columns=['lat_bin', 'lng_bin'], values='n_events')
    gridded_data_wide.columns = [f'lat_bin_{lat}_lng_bin_{lng}' for lat, lng in gridded_data_wide.columns]

    return gridded_data_wide


def make_3d_array(gridded_data_wide, nrow, ncol):
    ntimesteps = gridded_data_wide.shape[0]
    grid = list()
    for t in range(ntimesteps):
        grid_t = list()
        for i in range(nrow):
            grid_t.append(gridded_data_wide.iloc[t, (i*ncol):(i*ncol + ncol)])
        grid_t = np.vstack(grid_t)
        grid.append(grid_t)
    grid = np.stack(grid, axis=0)
    return grid


def plot_grid(df, grid_3d, nrow, ncol, zmax):
    
    lat_bins = np.linspace(df['start_lat'].min(), df['start_lat'].max(), num= nrow + 1)
    lng_bins = np.linspace(df['start_lng'].min(), df['start_lng'].max(), num= ncol + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title('n_events_total')
    for lng in lng_bins: ax.axvline(x=lng, linestyle='--', color='black', linewidth=0.5)
    for lat in lat_bins: ax.axhline(y=lat, linestyle='--', color='black', linewidth=0.5)
    im = ax.imshow(
        grid_3d.sum(axis=0), 
        cmap='coolwarm', origin='lower', 
        vmin=0, vmax=zmax, alpha=0.4,
        extent=(lng_bins[0], lng_bins[-1], lat_bins[0], lat_bins[-1])
    )
    ax.scatter(x=df['start_lng'], y=df['start_lat'], s=0.1, c='gray')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.show()


def plot_timeseries_cells(gridded_data_wide):

    fig, ax = plt.subplots(figsize=(20, 3))

    for _, col in gridded_data_wide.items():
        ax.plot(col, linewidth=1, alpha=0.3, color='black')

    ax.plot(gridded_data_wide.sum(axis=1), alpha=0.4, color='red', label='total')    
    ax.set_ylabel('No. events')
    ax.set_title('No. events by cell')
    plt.legend()
    plt.show()


def split_data(df, f_train, f_val):

    i_train = int(f_train*len(df))
    i_val =  int((f_train + f_val)*len(df))

    train_df_original = df[:i_train] 
    val_df_original = df[i_train:i_val]
    test_df_original = df[i_val:]

    train_timestamps = df.index[:i_train] 
    val_timestamps = df.index[i_train:i_val]
    test_timestamps = df.index[i_val:]

    train_mean = train_df_original.mean(axis=0)
    train_sd = train_df_original.std(axis=0)

    train_df = (train_df_original - train_mean) / train_sd
    val_df = (val_df_original - train_mean) / train_sd
    test_df = (test_df_original - train_mean) / train_sd

    print(f'train: {train_df_original.shape}\nval: {val_df_original.shape}\ntest: {test_df_original.shape}')

    return {
        'train_df_original': train_df_original,
        'val_df_original': val_df_original,
        'test_df_original': test_df_original,
        'train_timestamps': train_timestamps,
        'val_timestamps': val_timestamps,
        'test_timestamps': test_timestamps,
        'train_mean': train_mean,
        'train_sd': train_sd,
        'train_df': train_df,
        'val_df': val_df,
        'test_df': test_df,
    }


class WindowGenerator:


    def __init__(self, input_width, label_width, offset, train_df, val_df, test_df, label_columns, batch_size):

        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        self.batch_size = batch_size

        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        self.input_width = input_width
        self.label_width = label_width
        self.offset = offset

        self.total_window_size = input_width + offset

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.label_slice = slice(self.label_start, None) # None -> slice goes until end of iterable on which it's used
        self.label_indices = np.arange(self.total_window_size)[self.label_slice]


    def __repr__(self):
        return '\n'.join([
            f'Total window size = {self.total_window_size}',
            f'Input indices = {self.input_indices}',
            f'Label indices = {self.label_indices}',
            f'Label columns = {self.label_columns}'
        ])


    def split_window(self, features):
        
        # (1st dimension is batch, 2nd is timestep, 3rd is variable)
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.label_slice, :]
        
        # keep only desired output columns
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1
            )

        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        return inputs, labels


    def plot(self, model=None, plot_col='n_events_total', max_subplots=3):
        
        inputs, labels = self.example
        
        plt.figure(figsize=(12, 5))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n+1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(
                self.input_indices, 
                inputs[n, :, plot_col_index],
                label='Inputs', 
                marker='.', 
                zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(
                self.label_indices, 
                labels[n, :, label_col_index],
                edgecolors='k', 
                label='Labels', 
                c='#2ca02c', 
                s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(
                    self.label_indices, 
                    predictions[n, :, label_col_index],
                    marker='X', 
                    edgecolors='k', 
                    label='Predictions',
                    c='#ff7f0e', 
                    s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time [h]')
        plt.tight_layout()


    def make_dataset(self, data):
        data = np.array(data, dtype=float)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=self.batch_size
        )
        ds = ds.map(self.split_window)
        return ds
    

    def make_dataset_no_shuffle(self, data):
        data = np.array(data, dtype=float)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=False,
            batch_size=self.batch_size
        )
        ds = ds.map(self.split_window)
        return ds


    @property
    def train_ds(self):
        return self.make_dataset(self.train_df)
    

    @property
    def val_ds(self):
        return self.make_dataset(self.val_df)
    

    @property
    def test_ds(self):
        return self.make_dataset(self.test_df)
    

    @property
    def test_ds_no_shuffle(self):
        return self.make_dataset_no_shuffle(self.test_df)


    @property
    def example(self):
        return next(iter(self.test_ds))


def compile_and_fit(model, window, max_epochs, patience):
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        mode='min'
    )

    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()]
    )

    history = model.fit(
        window.train_ds,
        validation_data=window.val_ds,
        epochs=max_epochs,
        callbacks=[early_stopping]
    )

    return history


def make_ypred(model, w, train_mean, train_sd):
    ypred_scaled = model.predict(w.test_ds_no_shuffle)
    ypred = ypred_scaled*train_sd + train_mean
    ypred = tf.squeeze(ypred, axis=-1).numpy()
    reconstructed = np.full((len(w.test_df),), np.nan)
    start_index = w.input_width
    for i, prediction in enumerate(ypred):
        reconstructed[start_index + i] = prediction[0]

    return reconstructed





def check_stationarity_of_single_series(ts, i, j):
    result = adfuller(ts)
    print(f'ADF Statistic: {result[0]:.2f}')
    print(f'p-value: {result[1]:.2f}')


def check_ac_structure(data):

    fig, ax = plt.subplots(1, 2, figsize=(15, 3))
    plot_acf(data, lags=150, ax=ax[0])
    plot_pacf(data, lags=150, ax=ax[1])
    for i in [0, 24*1, 24*2, 24*3, 24*4, 24*5, 24*6]:
        ax[0].axvline(x=i, color='gray', linestyle='--', alpha=0.7)
        ax[1].axvline(x=i, color='gray', linestyle='--', alpha=0.7)
    ax[0].set_xticks([0, 24*1, 24*2, 24*3, 24*4, 24*5, 24*6])
    ax[1].set_xticks([0, 24*1, 24*2, 24*3, 24*4, 24*5, 24*6])
    plt.show()


def compute_sape(ypred, ytest):
    '''
    Compute symmetrical absolute percentage errors
    '''

    def smape_i(ypred_i, ytest_i):
        if (ypred_i != 0.0) & (ytest_i != 0.0):
            return 100.0 * ((np.abs(ypred_i) - np.abs(ytest_i)) / (np.abs(ytest_i) + np.abs(ypred_i)))
        else:
            return 0.0
    return np.array(list(map(smape_i, ypred, ytest)))


def model_var(gridded_data_wide, split, maxlags):

    gridded_data_wide = gridded_data_wide.asfreq('h') 

    train = gridded_data_wide.iloc[:-split, :]
    test = gridded_data_wide.iloc[-split:, :]

    model = VAR(train)
    model_fit = model.fit(maxlags=maxlags)

    n_forecast = len(test)
    forecast = model_fit.forecast(train.values[-model_fit.k_ar:], steps=n_forecast)

    ypred = forecast.flatten()
    ytest = np.array(test).flatten()
    rmse = root_mean_squared_error(ytest, ypred)
    sape = compute_sape(ypred, ytest)

    print(f'{model_fit.params.size:,} params. on {train.size:,} datapoints')
    
    return {
        'rmse': rmse, 
        'model_fit': model_fit, 
        'train': train, 
        'test': test, 
        'forecast': forecast,
        'ypred': ypred, 
        'ytest': ytest,
        'sape': sape,
        'gridded_data_wide': gridded_data_wide
    }


def plot_var(results, error_threshold):

    fig = plt.figure(figsize=(12, 6))

    gs = fig.add_gridspec(2, 2)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.scatter(results['ytest'], results['ypred'], color='gray', s=5, alpha=0.4)
    ax0.plot([0, 350], [0, 350], 'k--')
    ax0.set_title('Pred. vs. Obs.')
    ax0.set_xlabel('ytest')
    ax0.set_ylabel('ypred')
    ax0.text(x=10, y=300, s=f"rmse = {results['rmse']:.2f}") 

    cell = 50
    ax2 = fig.add_subplot(gs[1, :])
    ax2.set_title(f"{results['gridded_data_wide'].columns[cell]}")
    ax2.plot(results['test'].iloc[:, cell], 'k-', label='ytest')
    ax2.plot(results['test'].index, results['forecast'][:, cell], 'r-', label='ypred')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d %H:%M'))
    ax2.tick_params(axis='x', rotation=90)
    ax2.axhline(y=0, color='gray')
    ax2.legend()

#    sape_bins = np.arange(0, 105, 5)
#    ax1 = fig.add_subplot(gs[0, 1])
#    ax1.set_title(f"SAPE distribution: {100.0*len(results['sape'][results['sape'] < error_threshold])/len(results['sape']):.1f}% < {error_threshold}")
#    ax1.hist(results['sape'], bins=sape_bins, edgecolor='black')
#    ax1.set_xticks(sape_bins)
#    ax1.tick_params(axis='x', rotation=90)
#    ax1.set_xlabel('SAPE')
#    ax1.set_ylabel('Abs. frequency')
    
    plt.tight_layout()
    plt.show()


def model_lstm(gridded_data_wide, test_size, sample_size):

    train_size = len(gridded_data_wide) - test_size

    gridded_data_wide_train = gridded_data_wide[:train_size]
    gridded_data_wide_test = gridded_data_wide[train_size:]

    scaler = StandardScaler()
    gridded_data_wide_train_scaled = scaler.fit_transform(gridded_data_wide_train)
    gridded_data_wide_test_scaled = scaler.transform(gridded_data_wide_test)

    Xtrain, ytrain, Xtest, ytest  = [], [], [], []

    for i in range(gridded_data_wide_train_scaled.shape[0] - sample_size):
        Xtrain.append(gridded_data_wide_train_scaled[i:i + sample_size])
        ytrain.append(gridded_data_wide_train_scaled[i + sample_size])

    for i in range(gridded_data_wide_test.shape[0] - sample_size):
        Xtest.append(gridded_data_wide_test_scaled[i:i + sample_size])
        ytest.append(gridded_data_wide_test_scaled[i + sample_size])

    Xtrain, ytrain, Xtest, ytest = np.array(Xtrain), np.array(ytrain), np.array(Xtest), np.array(ytest)

    n_features = gridded_data_wide.shape[1]

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(sample_size, n_features)),
        tf.keras.layers.LSTM(10, 'relu'),
        tf.keras.layers.Dense(n_features)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    history = model.fit(Xtrain, ytrain, epochs=8, batch_size=20, validation_split=0.1, verbose=0)

    ypred = model.predict(Xtest)

    ypred_original = scaler.inverse_transform(ypred)
    ytest_original = scaler.inverse_transform(ytest)

    ypred_original_flattened = ypred_original.flatten()
    ytest_original_flattened = ytest_original.flatten()

    rmse = root_mean_squared_error(ytest_original, ypred_original)
    sape = compute_sape(ypred_original_flattened, ytest_original_flattened) 
    
    print(f'{model.count_params():,} params. on {Xtrain.size:,} datapoints')

    return {
        'rmse': rmse, 
        'model_fit': model, 
        'history': history, 
        'ypred': ypred_original, 
        'ytest': ytest_original,
        'gridded_data_wide': gridded_data_wide,
        'sape': sape,
    }


def compute_sape(ypred, ytest):
    '''
    Compute symmetrical absolute percentage errors
    '''

    def smape_i(ypred_i, ytest_i):
        if (ypred_i != 0.0) & (ytest_i != 0.0):
            return 100.0 * ((np.abs(ypred_i) - np.abs(ytest_i)) / (np.abs(ytest_i) + np.abs(ypred_i)))
        else:
            return 0.0
    return np.array(list(map(smape_i, ypred, ytest)))


def model_fnn(gridded_data_wide, test_size, lags):

    train_size = len(gridded_data_wide) - test_size

    gridded_data_wide_train = gridded_data_wide[:train_size]
    gridded_data_wide_test = gridded_data_wide[train_size:]

    scaler = StandardScaler()
    gridded_data_wide_train_scaled = scaler.fit_transform(gridded_data_wide_train)
    gridded_data_wide_test_scaled = scaler.transform(gridded_data_wide_test)

    Xtrain, ytrain = [], []
    Xtest, ytest = [], []
    maxlag = max(lags)

    for lag in lags: 
        Xtrain.append(gridded_data_wide_train_scaled[(maxlag - lag):-lag])
        Xtest.append(gridded_data_wide_test_scaled[(maxlag - lag):-lag])

    ytrain = np.array(gridded_data_wide_train_scaled[maxlag:])
    ytest = np.array(gridded_data_wide_test_scaled[maxlag:])
    
    Xtrain = np.stack([arr for arr in Xtrain])
    Xtest = np.stack([arr for arr in Xtest])
    
    # put time steps as 1st dimension, then put all features from all lags into simple dimension
    Xtrain_transposed = Xtrain.transpose(1, 0, 2).reshape(Xtrain.shape[1], -1) 
    Xtest_transposed = Xtest.transpose(1, 0, 2).reshape(Xtest.shape[1], -1)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(Xtrain_transposed.shape[1], )),
        tf.keras.layers.Dense(100, activation='linear'),
        tf.keras.layers.Dense(ytrain.shape[1], activation='linear'),
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    print(f'{model.count_params():,} params. on {Xtrain.size:,} datapoints')

    history = model.fit(Xtrain_transposed, ytrain, epochs=8, batch_size=20, validation_split=0.1, verbose=0)
    
    ypred = model.predict(Xtest_transposed)

    ypred_original = scaler.inverse_transform(ypred)
    ytest_original = scaler.inverse_transform(ytest)

    ypred_original_flattened = ypred_original.flatten()
    ytest_original_flattened = ytest_original.flatten()

    rmse = root_mean_squared_error(ytest_original_flattened, ypred_original_flattened)
    sape = compute_sape(ypred_original_flattened, ytest_original_flattened) 

    return {
        'rmse': rmse, 
        'model_fit': model, 
        'history': history, 
        'ypred': ypred_original, 
        'ytest': ytest_original,
        'gridded_data_wide': gridded_data_wide,
        'sape': sape,
    }



def plot_nn(results, error_threshold, cell):

    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 5)

    ax0 = fig.add_subplot(gs[0, :3])
    ax0.scatter(results['ytest'], results['ypred'], color='gray', s=5, alpha=0.4)
    ax0.plot([0, 350], [0, 350], 'k--')
    ax0.set_title('Pred. vs. Obs.')
    ax0.set_xlabel('ytest')
    ax0.set_ylabel('ypred')
    ax0.text(x=10, y=300, s=f"rmse = {results['rmse']:.2f}") 

#    sape_bins = np.arange(0, 105, 5)
#    ax1 = fig.add_subplot(gs[0, 3:])
#    ax1.set_title(f'SAPE distribution: {100.0*len(results['sape'][results['sape'] < error_threshold])/len(results['sape']):.1f}% < {error_threshold}%')
#    ax1.hist(results['sape'], bins=sape_bins, edgecolor='black')
#    ax1.set_xticks(sape_bins)
#    ax1.tick_params(axis='x', rotation=90)
#    ax1.set_xlabel('SAPE')
#    ax1.set_ylabel('Abs. frequency')
#
    ax2 = fig.add_subplot(gs[1, :4])
    ax2.set_title(f"{results['gridded_data_wide'].columns[cell]}")
    ax2.plot(results['gridded_data_wide'].iloc[-len(results['ytest']):, cell], 'k-', label='ytest')
    ax2.plot(results['gridded_data_wide'].iloc[-len(results['ytest']):, cell].index, results['ypred'][:, cell], 'r-', label='ypred')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d %H:%M'))
    ax2.tick_params(axis='x', rotation=90)
    ax2.axhline(y=0, color='gray')
    ax2.legend()

    ax3 = fig.add_subplot(gs[1, 4:])
    ax3.plot(results['history'].history['loss'], label='train_loss')
    ax3.plot(results['history'].history['val_loss'], label='val_loss')
    ax3.legend()    
    
    plt.tight_layout()
    plt.show()


