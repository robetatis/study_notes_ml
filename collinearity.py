import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


n = 1000
beta0 = 1.4
beta1 = 2
beta2 = -3
epsilon = np.random.normal(0, 4, n) 

x1 = np.random.uniform(10, 100, n)
x2 = np.random.uniform(5, 224, n)
y = beta0 + beta1*x1 - beta2*x2 + epsilon

x2_collinear = 1.3 + 0.8*x1 + np.random.normal(0, 4, n) 
y_collinear = 1.4 + 2*x1 - 3*x2_collinear + np.random.normal(0, 4, n) 

beta1_hat = np.arange(-1, 4, 0.01)
beta2_hat = np.arange(-6, 1, 0.01)

beta1_grid, beta2_grid = np.meshgrid(beta1_hat, beta2_hat)
beta_grid = np.column_stack([beta1_grid.ravel(), beta2_grid.ravel()])

yhat = beta0 + beta_grid[:, 0]*x1 + beta_grid[:, 1]

print(yhat)
exit()

print(beta_grid[0][0])
exit()

fig, ax = plt.subplots()
ax.scatter(x1, x2)
plt.savefig('collinearity.png')