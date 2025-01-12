import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import imageio
import requests
from io import StringIO
from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.api import VAR
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler


def get_url(path):

    path = Path(path).expanduser()

    with open(path, 'r') as f:
        url = f.read()
    
    return url


def download_data(url):

    path = Path('data/demand.csv')

    if not path.exists:

        try:
            response = requests.get(url)
            response.raise_for_status()  # Check if the request was successful

            csv_data = StringIO(response.text)
            df = pd.read_csv(csv_data)

            df.to_csv('data/demand.csv')

        except requests.exceptions.RequestException as e:
            print(f"Error downloading the CSV file: {e}")


def load_data(path):
    return pd.read_csv(
        path, 
        parse_dates=['start_time'], 
        index_col='start_time', 
        usecols=[
            'start_time', 
            'start_lat', 
            'start_lng', 
            'end_lat', 
            'end_lng', 
            'ride_value'
        ]
    )


def build_time_components(df):

    df['hour_of_day'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek # (Monday=0, Sunday=6)
    df['week_of_year'] = df.index.isocalendar().week
    df['month_of_year'] = df.index.month
    df['day_of_month'] = df.index.day


def summarize_data(df):
    print(f"No. ride requests = {df.shape[0]:,}")
    print(f"Time range = {df.index.min().strftime('%Y-%m-%d %H:%M')} - {df.index.max().strftime('%Y-%m-%d %H:%M')}")
    print(f"Min. ride value = {df['ride_value'].min():.2f}")
    print(f"Max. ride value = {df['ride_value'].max():.2f}")
    print(f"Mean. ride value = {df['ride_value'].mean():.2f}")
    print(f"Median. ride value = {df['ride_value'].quantile(0.5):.2f}")


def rides_per_capita_month():
    print(f'620k rides in March 2022 in a city of 420k inhabitants ~ {627210/426000:.2f} rides per capita-month')


def bin_ride_value(df, bin_size):
    hist, bin_edges = np.histogram(df, bins=np.arange(0, df['ride_value'].max(), bin_size))
    for i in range(len(hist)):
        print(f"Range {bin_edges[i]:.1f} - {bin_edges[i+1]:.1f} --- {hist[i]}")


def bin_by_cut_value(df, cut_value):
    normal_range = df[df['ride_value'] <= cut_value]
    outliers =  df[df['ride_value'] > cut_value]

    perc_normal_range = 100.0 * (normal_range.shape[0] / (normal_range.shape[0] + outliers.shape[0]))
    perc_outliers = 100.0 * (outliers.shape[0] / (normal_range.shape[0] + outliers.shape[0]))
    perc_revenue_outliers = 100.0 * df[df['ride_value'] > cut_value]['ride_value'].sum()/df['ride_value'].sum()

    print(f"No. datapoints below cut value = {normal_range.shape[0]}; ({perc_normal_range:.2f}%)")
    print(f"No. datapoints above cut value = {outliers.shape[0]}; ({perc_outliers:.2f}%)")
    print(f"Perc. revenue above cut value = {perc_revenue_outliers:.2f}%")


def summary_plots(df):

    build_time_components(df)
    
    fig, ax = plt.subplots(2, 3, figsize=(10, 5))

    ax[0, 0].set_title('hour_of_day')
    ax[0, 0].set_ylabel('n_rides')
    ax[0, 0].hist(df['hour_of_day'], bins=24, range=(0, 24), edgecolor='black')
    ax[0, 0].set_xticks(np.arange(0, 25, 2))
    ax[0, 0].set_xticklabels(np.arange(0, 25, 2))

    ax[0, 1].set_title('day_of_week')
    ax[0, 1].set_ylabel('n_rides')
    ax[0, 1].hist(df['day_of_week'], bins=7, range=(0, 7), edgecolor='black')
    ax[0, 1].set_xticks(np.arange(0, 7, 1) + 0.5)
    ax[0, 1].set_xticklabels(['Mo', 'Tu', 'We', 'Th', 'Fr.', 'Sa.', 'Su.'])

    ax[0, 2].set_title('week_of_year')
    ax[0, 2].set_ylabel('n_rides')
    ax[0, 2].hist(df['week_of_year'], bins=5, range=(9, 14), edgecolor='black')
    ax[0, 2].set_xticks(np.arange(9, 14, 1) + 0.5)
    ax[0, 2].set_xticklabels(np.arange(9, 14, 1))

    ax[1, 0].set_title('ride_value, 0 - 10 Eur.')
    ax[1, 0].set_ylabel('n_rides')
    ax[1, 0].set_xlabel('ride_value')
    ax[1, 0].hist(df['ride_value'], bins=np.arange(0, 11, 1), edgecolor='black')
    ax[1, 0].set_xticks(np.arange(0, 11, 1))
    ax[1, 0].set_xticklabels(np.arange(0, 11, 1))

    ax[1, 1].set_title('ride_value, >10 Eur.')
    ax[1, 1].set_ylabel('n_rides')
    ax[1, 1].hist(df['ride_value'], bins=np.arange(10, 3200, 50), edgecolor='black')
    ax[1, 1].set_xlabel('ride_value')
    ax[1, 2].set_title('Ride-value-weighted density')
    
    sample_size = 1000 # take sub-sample for 2D kernel-density plot
    indices = np.random.choice(df.shape[0], size=sample_size, replace=False)
    df_sample = df.iloc[indices, :]

    contours = sns.kdeplot(
        x=df_sample['start_lng'], y=df_sample['start_lat'], weights=df_sample['ride_value'],
        fill=False, cmap="coolwarm", linewidths=1, thresh=0.0, levels=20,
        ax=ax[1, 2]
    )
    ax[1, 2].scatter(x=df['start_lng'], y=df['start_lat'], s=0.1, c='gray')

    plt.tight_layout()
    plt.show()


def map_hotspot(df):

    fig, ax = plt.subplots()

    ax.set_title('Ride-value-weighted density')
    ax.scatter(x=df['start_lng'], y=df['start_lat'], s=0.1, c='gray')
    
    sample_size = 1000 # take sub-sample for 2D kernel-density plot
    indices = np.random.choice(df.shape[0], size=sample_size, replace=False)
    df_sample = df.iloc[indices, :]
    contours = sns.kdeplot(
        x=df_sample['start_lng'], y=df_sample['start_lat'], weights=df_sample['ride_value'],
        fill=False, cmap="coolwarm", linewidths=1, thresh=0.0, levels=40, 
        ax=ax
    )

    ax.set_xlim(24.6, 24.85)
    ax.set_ylim(59.40, 59.47)
    
    plt.tight_layout()
    plt.show()


def grid_data(df, nrow, ncol):

    df['time_bin'] = df.index.floor('h')

    lat_bins = np.linspace(df['start_lat'].min(), df['start_lat'].max(), num= nrow + 1)
    lng_bins = np.linspace(df['start_lng'].min(), df['start_lng'].max(), num= ncol + 1)

    df['lat_bin'] = pd.cut(df['start_lat'], bins=lat_bins)
    df['lng_bin'] = pd.cut(df['start_lng'], bins=lng_bins)

    gridded_data = df \
        .groupby(['time_bin', 'lat_bin', 'lng_bin'], observed=False) \
        .agg(
            n_rides=('ride_value', 'count'),
            mean_ride_value=('ride_value', 'mean'),
            min_ride_value=('ride_value', 'min'),
            max_ride_value=('ride_value', 'max'),
            q50_ride_vaue=('ride_value', lambda x: x.quantile(0.5)),
            q75_ride_vaue=('ride_value', lambda x: x.quantile(0.75)),
        ) \
        .reset_index()

    return gridded_data


def widen_grid_data(gridded_data):
    gridded_data_wide = gridded_data.pivot(index='time_bin', columns=['lat_bin', 'lng_bin'], values='n_rides')
    gridded_data_wide.columns = [f'lat_bin_{lat}_lng_bin_{lng}' for lat, lng in gridded_data_wide.columns]

    return gridded_data_wide


def make_3d_array(gridded_data_wide, ncol, nrow):
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


def plot_grid(df, nrides_total_grid, nrow, ncol):
    
    lat_bins = np.linspace(df['start_lat'].min(), df['start_lat'].max(), num= nrow + 1)
    lng_bins = np.linspace(df['start_lng'].min(), df['start_lng'].max(), num= ncol + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title('n_rides overall')
    for lng in lng_bins: ax.axvline(x=lng, linestyle='--', color='black', linewidth=0.5)
    for lat in lat_bins: ax.axhline(y=lat, linestyle='--', color='black', linewidth=0.5)
    im = ax.imshow(
        nrides_total_grid, 
        cmap='coolwarm', origin='lower', 
        vmin=0, vmax=100, alpha=0.4,
        extent=(lng_bins[0], lng_bins[-1], lat_bins[0], lat_bins[-1])
    )
    ax.scatter(x=df['start_lng'], y=df['start_lat'], s=0.1, c='gray')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.show()


def plot_timeseries_cells(gridded_data_wide_nonzero):

    fig, ax = plt.subplots(figsize=(20, 3))

    for _, col in gridded_data_wide_nonzero.items():
        ax.plot(col, linewidth=1, alpha=0.5, color='black')

    ax.plot(gridded_data_wide_nonzero.sum(axis=1), color='red', label='total')    
    ax.set_ylabel('No. rides')
    ax.set_title('No. rides per hour, all nonzero grid cells')
    plt.legend()
    plt.show()


def animate_gridded_data(gridded_data):

    images = []
    timestamps = gridded_data['time_bin'].unique()
    for i, ts in enumerate(timestamps):
        data_for_ts = gridded_data[gridded_data['time_bin'] == ts]   
        grid = data_for_ts.pivot(index='lat_bin', columns='lng_bin', values='n_rides')    

        fig, ax = plt.subplots(figsize=(6, 6))

        im = ax.imshow(grid, cmap='coolwarm', origin='lower', vmin=0, vmax=350)
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        ax.set_title(f'No. rides at {ts}')
        ax.set_xlim(-1, 19)
        ax.set_ylim(-1, 19)

        plt.tight_layout()

        filename = f'results/gif_frames/frame_{i}.png'
        plt.savefig(filename)

        plt.close()

        images.append(imageio.imread(filename))

    imageio.mimsave('results/no_rides.gif', images, fps=2)


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


