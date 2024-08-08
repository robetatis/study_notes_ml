import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data_size = 100
epsilon = np.random.normal(0, 30, data_size)
x_pop = np.linspace(-5, 5, 1000)
f = Polynomial([0.1, -0.3, 0.4, -1.4])

x_data = np.random.choice(x_pop, data_size)
y_data = f(x_data) + epsilon

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=0)

flexibility = np.arange(0, 20, 1)
models = list()
mse = list()
bias = list()
variance = list()
for flex in flexibility:
    model = Polynomial.fit(x_train, y_train, flex)
    models.append(model)
    mse.append(mean_squared_error(y_test, model(x_test)))
    # compute variance: loop over multiple training sets, compute 
    # compute bias



fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].scatter(x_train, y_train, label='train', s=5)
ax[0].scatter(x_test, y_test, label='test', s=5)
ax[0].plot(x_pop, models[1](x_pop), c='green', linewidth=1, label='order_1')
ax[0].plot(x_pop, models[2](x_pop), c='red', linewidth=1, label='order_2')

ax[1].scatter(flexibility, mse, s=20)
ax[1].plot(flexibility, mse, linestyle='--', linewidth=1)
ax[1].set_title('Test MSE')
ax[0].legend()

plt.savefig('plt.png')


