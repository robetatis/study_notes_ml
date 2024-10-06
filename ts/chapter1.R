# https://online.stat.psu.edu/stat510/lesson/1

# in time series analysis, data are not necessarily independent and not necessarily
# iid

# first look: trend, seasonality (-> calendar time has effect), long-run cycle, 
# sigma = const, outliers, abrupt breaks in mean or variance

# example: annual earthquake series
library(dplyr)
earthquakes <- read.table(
  file = 'data/earthquakes.csv',
  header = TRUE,
  sep = ','
)
earthquakes$Date <- as.Date(earthquakes$Date, format='%m/%d/%Y')
earthquakes$Year <- format(earthquakes$Date, format='%Y')
earthquakes %>%
  group_by(Year) %>%
  count() %>%
  na.omit() -> earthquakes_by_year
par(mfcol = c(1, 3))
plot(earthquakes_by_year$Year, earthquakes_by_year$n, type='o', pch=20)
abline(h=mean(earthquakes_by_year$n))

# features of the series:
#   - upward trend
#   - no clear seasonality
#   - maybe 1 outlier in 2011 (712 earthquakes)
#   - more erratic towards right side, perhaps variance is higher there

# example: the AR(1) model:
# x_t = delta + phi_1*x_tminus1 + w_t
# assumptions:
#   - w_t --> iid N(0, sigma_w^2)
#   - w_t independent of x

# lag-1 plot
n <- nrow(earthquakes_by_year)
x_t <- earthquakes_by_year$n[2:n]
x_tminus1 <- earthquakes_by_year$n[1:(n-1)]
plot(x_tminus1, x_t)

# assuming we can just use OLS on x_t ~ x_tminus1 (for different reasons this is 
# not correct, discussed below), we'd do this:
ols <- lm(x_t ~ x_tminus1)
summary(ols)
abline(ols)

# OLS results 
# model: x_t = 141.5 + 0.672 * x_tminus1, F(1, 50) = 22.81, p<0.001
# residual standard error = 91.4
# slope is significant, so lag1 variable is useful
# R^2 = 0.29
# residual analysis:
plot(ols$fitted.values, ols$residuals)
abline(h=0)
# there's a two major outliers at fitted values ~460 (low leverage) and ~700 

# example with trend and seasonality: quarterly beer production in australia
library(fpp)
data(ausbeer)
ausbeer <- data.frame(
  t = as.numeric(time(ausbeer))[1:72],
  tsquared = as.numeric(time(ausbeer))[1:72]^2,
  quarter = cycle(ausbeer)[1:72],
  x_t = as.numeric(ausbeer)[1:72]
)

# features of the series:
#   - clear upward trend
#   - clear seasonality, with values increasing from Q2 to Q4 and then going back 
#     down
#   - 1 outliers at end of series
#   - variance seems slightly larger toward end of series

# one could manually devise a model based on classical regression, and try
# to reproduce with it the observed trend and seasonality:
#   - for linear trend, use t as predictor, t + t^2 for quadratic trend
#   - for quarterly seasonality, we can define indicator variables Sj, where
#     j = 1, 2, 3, 4 (the quarters) and Sj = 1 if value is in quarter j, 0 otherwise
#   - model: x_t = beta1*t + beta2*t^2 + alpha1*S1 + alpha2*S2 + alpha3*S3 + alpha4*S4 + w_t

# create Sj
for(j in 1:4){ 
  varname <- paste0('S', j)
  varval <- assign(varname, ifelse(ausbeer$quarter == j, 1, 0))
}
ausbeer <- cbind(ausbeer, S1, S2, S3, S4)
i_train <- 1:50
i_test <- 51:nrow(ausbeer)
ausbeer_train <- ausbeer[i_train, ]
ausbeer_test <- ausbeer[i_test, ]

# fit model
ols_trend_seasonal <- lm(
  formula = x_t ~ 0 + t + tsquared + S1 + S2 + S3 + S4, 
  data = ausbeer_train)
summary(ols_trend_seasonal)

# predict and test
newdata <- ausbeer_test[, c(-3, -4)]
x_t_pred <- predict(ols_trend_seasonal, newdata)

par(mfcol = c(1, 3))
plot(ausbeer$t, ausbeer$x_t, 
     type='o', pch=20, xlab='t', ylab='x_t', 
    main='Original data + prediction')
lines(ausbeer_test$t, x_t_pred, col='red', pch=20)
plot(ols_trend_seasonal$fitted.values, ols_trend_seasonal$residuals,
     xlab='t', ylab='x_t', 
     main='Training residuals')
abline(h=0)
plot(ausbeer_test$x_t, x_t_pred,
     main='Obs. vs. Pred, test data')
abline(a=0, b=1)


# **********************************************
# Sample autocorrelation function ACF
# *********************************************

# x_t vs. x_tminus1 vs. x_tminus2 vs. x_tminus3, ...
# reveals possible structure of series
# after model-fitting, ACF of residuals can be useful -> there should be no 
# significant association at any lag

# autocorrelation = cov(x_t, x_tminus1)/sd(x_t)*sd(x_tminus1)

# ACF of residuals of the above 2 models
par(mfcol=c(1, 2))
acf(ols$residuals, xlim=c(1, 20), ylim=c(-1, 1), main='ACF residuals earthquakes')
acf(ols_trend_seasonal$residuals, xlim=c(1, 20), ylim=c(-1, 1), main='ACF residuals beer')


# ********************
# stationarity
# *******************

# related to how distribution of stochastic process {X_t}, and its mean, variance, 
# and covariance at a particular lag, vary over time

# strict/strong stationarity: 
#   - distribution of stochastic process {Xt} does not change in time
#   - variance may be infinite!

# weak stationarity: 
#   - mean, variance and autocorrelation do not change in time; the actual shape 
#     of the distribution may change
#   - variance must be finite!

# relation strong / weak stationarity:
#   - if it weren't for the infinite variance, strong would imply weak, but weakly
#     stationary series must have finite variance
#   - obviously, weak does NOT imply strong, since distribution may change while
#     mean, variance and autocorrelation stay the same


# KPSS stationarity test: kwiatkowski-phillips-schmidt-shin test
# tests for stationarity with respect to level value and/or trend
# H0: series is (level or trend) stationary
# H1: series has unit root

# grab data
x_t <- read.table(
  file = 'data/unemployment.csv',
  header = TRUE,
  sep = '\t'
)
x_t <- x_t[[2]]
tt <- 1:length(x_t)

# visualize
plot(1:length(x_t), x_t, type='o', pch=20)
abline(h=mean(x_t), col='blue')
abline(lm(x_t ~ tt), col='red')

# compute test statistic
kpss <- function(x_t){
  
  # compute base quantities
  n <- length(x_t)
  mean_x = mean(x_t)
  
  tt <- 1:length(x_t)
  model <- lm(x_t ~ tt)
  
  ei_mean <- x_t - mean_x
  ei_trend <- model$residuals
  
  S_t_mean <- cumsum(ei_mean)
  S_t_trend <- cumsum(ei_trend)
  
  lambda_mean <- sqrt(sum(ei_mean^2)/(n - 1))
  lambda_trend <- sqrt(sum(ei_trend^2)/(n - 2))
  
  kpss_mean <- sum(S_t_mean^2)/(n^2 * lambda_mean^2)
  kpss_trend <- sum(S_t_trend^2)/(n^2 * lambda_trend^2)
  
  return (c(kpss_mean=kpss_mean, kpss_trend=kpss_trend))
}
kpss(x_t)

# with tseries library
library(tseries)
tseries::kpss.test(x_t, 'Level')
tseries::kpss.test(x_t, 'Trend')

# kpss test for white noise
x_t <- rnorm(100)
tseries::kpss.test(x_t, 'Level')
plot(x_t, type='o', pch=20)

# kpss test for random walk
w_t <- runif(100, -1, 1)
x_t <- numeric(length = 100)
x_t[1] <- 0
for(i in 2:length(w_t)){
  x_t[i] <- x_t[i-1] + w_t[i]
}
tseries::kpss.test(x_t, 'Level')
plot(x_t, type='o', pch=20)



# ********************
# The AR(1) model
# *******************

# x_t = delta + phi_1*x_tminus1 + w_t

# assumptions:
#   - x_t weakly stationary
#   - w_t --> iid N(0, sigma_w^2)
#   - w_t independent of x

# properties:
#   - expected value E(x_t):
#     E(x_t) = E(delta + phi_1*x_tminus1 + w_t)
#     E(x_t) = E(delta) + E(phi_1*x_tminus1) + E(w_t)
#     E(x_t) = delta + phi_1*E(x_tminus1) + 0 
#     Due to stationarity, E(x_t) = E(x_tminus1). Then,
#     E(x_t) = delta / (1 - phi_1)
#
#   - variance Var(x_t):
#     Var(x_t) = Var(delta + phi_1*x_tminus1 + w_t)
#     ...assuming x_t and w_t (errors) are independent, and noting that delta and
#        phi_1 are constants:
#     Var(x_t) = 0 + Var(phi_1*x_tminus1) + Var(w_t)
#     Var(x_t) = phi_1^2*Var(x_tminus1) + sigma_w^2
#     ...assuming series is stationary, Var(x_t) = Var(x_tminus1)
#     Var(x_t) = sigma_w^2 / (1 - phi_1^2)
#
#   - autocovariance Cov(x_t, x_tminush) (h = lag):
#     ...writing x_t by substituting x_minus1 (use model without delta for simplicity)
#        going from x_t back to x_tminush
#     x_t = phi_1*x_tminus1 + w_t
#     x_t = phi_1^2*x_tminus2 + phi_1*w_tminus1 + w_t
#     x_t = phi_1^3*x_tminus3 + phi_1 2*w_tminus2 + phi_1*w_tminus1 + w_t
#     ... at t_minush:
#     x_t = phi_1^h*x_tminush + sum{i=0...h-1}(phi_1^i*w_tminusi)
#     ... substituting this for x_t in definition of cov:
#     Cov(x_t, x_tminush) = Cov(phi_1^h*x_tminush + sum{i=0...h-1}(phi_1^i*w_tminusi),
#                               x_tminush)
#     Cov(x_t, x_tminush) = Cov(phi_1^h*x_tminush, x_tminush) +
#                           Cov(sum{i=0...h-1}(phi_1^i*w_tminusi), x_tminush)
#     ... noting that the second Cov is zero, since the w_t are iid and independent
#         of the x_t, we get:
#     Cov(x_t, x_tminush) = phi_1^h*Cov(x_tminush, x_tminush) + 0
#     Cov(x_t, x_tminush) = phi_1^h*Var(x_tminush)
#     ... for AR(1), h = 1:
#     Cov(x_t, x_tminus1) = (phi_1*sigma_w^2)/(1 - phi_1^2)
#
#   - lag-1 autocorrelation = phi_1*Var(x_t)/Var(x_t) = phi_1


# ACF patterns with phi_1 > 0 and phi_1 < 0:

# set parameters
delta <- 20
phi_1_pos <- 0.6
phi_1_neg <- -0.6
n <- 50

# initialize series
x_pos <- double(length = n)
x_neg <- double(length = n)
x_pos[1] <- runif(1, min=0, max=10)
x_neg[1] <- runif(1, min=0, max=10)

# compute series
for(i in seq(2, n, by=1)){
  x_pos[i] <- delta + phi_1_pos*x_pos[i-1] + rnorm(n=1, mean=0, sd=10)
  x_neg[i] <- delta + phi_1_neg*x_neg[i-1] + rnorm(n=1, mean=0, sd=10)
}

# plot
par(mfrow=c(2, 2), mar=c(3, 3, 3, 1))
plot(x_pos, type='o', pch=20, main='phi_1 = 0.6')
acf(x_pos, xlim=c(1, 20), ylim=c(-1, 1), main='ACF pos')
plot(x_neg, type='o', pch=20, main='phi_1 = -0.6')
acf(x_neg, xlim=c(1, 20), ylim=c(-1, 1), main='ACF neg')

# for phi_1 > 0, ACF tapers towards 0 relatively quickly
# for phi_1 < 0, same happens, but more slowly and with alternating sign for ACF

# theoretically, for an AR(1) process, the phi for higher lags should be the 
# successive powers of phi_1

# for non-stationary data, use 1-order differencing to stabilize series
# example: ACF for daily cardiovascular mortality rate in LA county, 1970, 1979
x_t = scan('data/cmort.dat')
y_t <- diff(x_t, 1) # compute 1st-order difference series

par(mfcol=c(2, 2), mar=c(3, 2, 3, 1))
plot(x_t, type='o', pch=20, main='raw data')
plot(y_t, type='o', pch=20, main='differences')
acf(x_t, xlim=c(1, 24), ylim=c(-1, 1), main='ACF, raw')
acf(y_t, xlim=c(1, 24), ylim=c(-1, 1), main='ACF, differences')





