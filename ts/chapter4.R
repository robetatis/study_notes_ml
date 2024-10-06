# https://online.stat.psu.edu/stat510/lesson/4

# ****************************************
# Seasonal ARIMA models
# ****************************************

# there's a clear repetitive pattern every S time periods

# in seasonal ARIMA, there's seasonal AR and MA terms that use x_t and w_t at lag
# values that are multiples of S

# examples:
#   - x_tminus12 in a seasonal AR(1) model (time unit 1 month)
#   - x_tminus12 and x_tminus24 in a seasonal AR(2) (time unit 1 month)


# seasonal differencing:

# seasonality causes instationarity. e.g., distribution parameters in summer will
# not be the same as in winter 
# -> formulas for E(x_t), Var(x_t), etc. break down!

# with backshift operator: x_t - x_tminus12 = (1 - B^12)*x_t

# we may need both de-trending and seasonal differencing:
# (1-B^12)(1-B)*x_t = (x_t - x_tminus1) - (x_tminus12 - x_tminus13)
# what this does is 'break down' variation in x_t into trend and seasonality, i.e., 
# into short-term and long-term effects

# the model:

# ARIMA(p,d,q) x (P, D, Q)S; where p, d, q = non-seasonal orders; P, D, Q = seasonal
# orders

# (1)PHI(B^S)*phi(B)(x_t - miu) = THETA(B^S)*theta(B)*w_t
# seasonal_AR x non-seasonal_AR = seasonal_MA x non-seasonal_MA

# non-seasonal components:
#   - AR: phi(B) = 1 - phi_1*B - ... - phi_p*B^p
#   - MA: theta(B) = 1 + theta_1*B + ... theta_q*B^q
# seasonal components:
#   - AR_S: PHI(B^S) = 1 - PHI_1*B^S - ... - PHI_P*B^PS
#   - MA_S: THETA(B^S) = 1 + THETA_1*B^S + ... + THETA_Q*B^QS
# PS = P times S, i.e., a multiple of S. Same for Q (the no. of seasons backwards)


# example: ARIMA(0, 0, 1) x (0, 0, 1)12 
#   -> terms: 1 non-sesaonal MA, 1 seasonal MA, seasonality = 12 t units
# model: x_t - miu = THETA(B^12)*theta(B)*w_t   (no AR terms)
# x_t - miu = (1 + THETA_1*B^12)(1 + theta_1*B)*w_t
# x_t - miu = (1 + theta_1*B + THETA_1*B^12 + THETA_1*theta_1*B^13)*w_t
# x_t - miu = w_t + theta_1*w_tminus1 + THETA_1*w_tminus12 + THETA_1*theta_1*w_tminus13

# generate synthetic data
x_t <- as.numeric(astsa::sarima.sim(ma=0.7, sma=0.6, S=12, n=1000)) + 10

# check data, ACF and PACF
par(mfcol=c(1, 3))
plot(x_t, type='o', pch=20)
acf(x_t, xlim=c(1, 40), ylim=c(-1, 1), main='ACF')
pacf(x_t, xlim=c(1, 40), ylim=c(-1, 1), main='PACF')

# ACF:
#   - significant lags at 1, 12 and 13, as expected. 
#   - strangely, there's also significant spike at lag 11
#   - slightly larger ACF again at lag 24, i.e., at lag 2*S (S=12)

# PACF:
#   - 'seasonal' tapering -> tapering (with alternating signs) first after
#      lag 1, and then again after lag 11, then again after lag 24 
#      i.e., at multiples of S

# fit model
x_t_hat <- astsa::sarima(x_t, p=0, d=0, q=1, Q=1, S=12)
print(x_t_hat)


# another example: ARIMA(1, 0, 0) x (1, 0, 0)12
#   -> terms: 1 non-sesaonal AR, 1 seasonal AR, seasonality = 12 t units
# model: PHI(B^12)phi(B)(x_t - miu) = w_t
# setting x_t - miu = z_t:
# (1 - PHI_1(B^12))(1 - phi_1(B))(z_t) = w_t
# (1 - phi_1(B) - PHI_1(B^12) + PHI_1*phi_1(B^13))(z_t) = w_t
# z_t = phi_1*z_tminus1 + PHI_1*z_tminus12 - PHI_1*phi_1*z_tminus13 + w_t

x_t <- as.numeric(astsa::sarima.sim(ar=0.6, sar=-0.5, S=12, n=1000))

par(mfcol=c(1, 3))
plot(x_t, type='o', pch=20)
acf(x_t, xlim=c(1, 40), ylim=c(-1, 1), main='ACF')
pacf(x_t, xlim=c(1, 40), ylim=c(-1, 1), main='PACF')

# ACF:
#   - seasonal tapering, autocorrelation starts positive at lag 1, then tapers
#     and starts having negative values towards lag 12

# PACF:
#   - peaks at lags 1, 12 and 13, and also 11


# ****************************************
# identifying a seasonal model
# ****************************************

# 1. plot raw data, search for trend and seasonality
# 2. ACF and PACF for raw data:
#    depending on whether there's AR and/or MA terms, ACF or PACF will show seasonality
#    seasonality -> gradual oscillations with peaks at multiples of S
#    if AR, PACF will show significant peaks only at relevant lags, ACF will taper seasonally
#    if MA, ACF will show significant peaks only at relevant lags, PACF will taper seasonally
# 3. do necessary differencing, seasonal and non-seasonal. look again at ACF, PACF
#    after seasonal differencing, analyze ACF and PACF the same way as for a 
#      no-trend, non-seasonal process (ACF/PACF, which tapers? which spikes?)

# example ACF and PACF before/after seasonal differencing:
x_t <- as.numeric(astsa::sarima.sim(ar=0.6, sar=0.5, S=12, n=1000))
x_t_diff <- diff(x_t, lag=12)

par(mfrow=c(1, 3))
plot(x_t_diff, type='o', pch=20, main='x_t_diff')
acf(x_t_diff, xlim=c(1, 40), ylim=c(-1, 1), main='ACF diff')
pacf(x_t_diff, xlim=c(1, 40), ylim=c(-1, 1), main='PACF diff')

# example seasonal model, Colorado river monthly flow:

# grab data
x_t <- scan('data/coloradoflow.dat')
month <- seq(
  as.Date('1950-01-01', format='%Y-%m-%d'), 
  as.Date('1999-12-31', format='%Y-%m-%d'), 
  by='month') # make monthly time axis

df <- data.frame(
  t=month,
  month=format(month, format='%m'), # make month variable for grouping
  x_t=x_t) # put together into data frame

# aggregate data by month
df_monthly <- data.frame(
  month=seq(1, 12, by=1),
  mean=aggregate(df$x_t, by=list(df$month), FUN=mean)$x,
  min=aggregate(df$x_t, by=list(df$month), FUN=min)$x,
  max=aggregate(df$x_t, by=list(df$month), FUN=max)$x 
) 

# plot raw data and monthly data
par(mfcol=c(1, 2))
plot(x_t, type='o', pch=20)
plot(df_monthly$mean, type='o', pch=20, ylim=c(0, 10))
lines(df_monthly$min, col='gray')
lines(df_monthly$max, col='gray')

# ... there seems to be seasonality, -> S=12 seems reasonable. to explore further 
#     components, take lag-12 seasonal differences and look at ACF and PACF:
x_t_diff <- diff(x_t, lag=12)

# plot ACF and PACF for raw and 12-differenced data
windows(12, 6)
par(mfrow=c(2, 3))
plot(x_t, type='o', pch=20, main='x_t')
acf(x_t, xlim=c(1, 50), ylim=c(-1, 1), main='ACF'); abline(v=0, col='blue')
pacf(x_t, xlim=c(1, 50), ylim=c(-1, 1), main='PACF'); abline(v=0, col='blue')
plot(x_t_diff, type='o', pch=20, main='x_t_diff')
acf(x_t_diff, xlim=c(1, 50), ylim=c(-1, 1), main='ACF diff'); abline(v=0, col='blue')
pacf(x_t_diff, xlim=c(1, 50), ylim=c(-1, 1), main='PACF diff'); abline(v=0, col='blue')

# non-seasonal components (look tapering/spikes ACF/PACF over multiples of h)
#   - in diff ACF, we see tapering around lags 1, 12, 24
#   - in diff PACF, we see spikes around lags 1, 12, 24
#   -> suggests non-seasonal AR(1)
# seasonal components (look tapering/spikes ACF/PACF over multiples of S):
#   - in diff ACF, we see spike at lag 1*S=12 (neg) and cut-off after that
#   - in diff PACF, we see tapering over 1*S, 2*S, 3*S
#   -> suggests seasonal MA(1)

# overall, seems like we could use ARIMA(1,0,0)x(0,1,1)12:
model <- astsa::sarima(
  x_t, 
  p=1, d=0, q=0, 
  P=0, D=1, Q=1, S=12)
print(model)

# check out residuals vs. fit
x_t_hat <- x_t - model$fit$residuals
ei <- as.numeric(model$fit$residuals)
plot(x_t_hat, ei)

# -> there's larger Var(e) for larger values -> non-constant variance -> need
#    to fix it by transformation, or ARCH model

# the qq plot for residuals also does not look normal -> large residuals on both
# ends -> extreme values in actual distribution of ei are more likely -> larger 
# variance -> 'fatter tails'
n <- length(ei)
qobs <- sort((ei - mean(ei))/sd(ei))
pobs <- (1:n)/(n)
qtheoretical <- qnorm(pobs)

par(mfcol=c(1, 2))
hist(ei)
plot(qtheoretical, qobs)
abline(a=0, b=1)

# make forecast
astsa::sarima.for(x_t, 24, 1,0,0,0,1,1,12)


# example beer production in Australia
library(fpp)
data(ausbeer)
ausbeer <- as.numeric(ausbeer[1:72])

plot(ausbeer, type='o', pch=20)

# obvious quarterly seasonality. since data is quarterly, S=4. 
# obvious upward trend
# -> need lag-4 differencing for seasonality and lag-1 differencing for trend

ausbeer_diff1 <- diff(ausbeer, lag=1)
ausbeer_diff1_diff4 <- diff(ausbeer_diff1, lag=4)

windows(12, 6)
par(mfrow=c(2, 3))
plot(ausbeer, type='o', pch=20, main='Original series')
acf(ausbeer, xlim=c(1, 20), ylim=c(-1, 1))
pacf(ausbeer, xlim=c(1, 20), ylim=c(-1, 1))
plot(ausbeer_diff1_diff4, type='o', pch=20, main='Diff 1, diff 4')
acf(ausbeer_diff1_diff4, xlim=c(1, 20), ylim=c(-1, 1))
text(x=1:20, y=-0.9, labels=1:20)
pacf(ausbeer_diff1_diff4, xlim=c(1, 20), ylim=c(-1, 1)); abline(v=0, col='blue')
text(x=1:20, y=-0.9, labels=1:20)

dev.off()

# non-seasonal -> PACF has spikes at lags 1 and 2, then cuts off
#                 ACF has intermittent tapering around, but not exactly,
#                 at multiples of S
#                 -> suggests AR(2)
# seasonal -> nothing seasonal in PACF
#             ACF has significant spike at lag 4, and a weird spike at lag 9
#             -> suggests seasonal MA(1) term with S=4

# overall, seems we could use ARIMA(2,1,0)x(0,1,1)4
x_t <- ausbeer
model <- astsa::sarima(
  x_t, 
  p=2, d=1, q=0, 
  P=0, D=1, Q=1, S=4)
print(model)

# check out residuals vs. fit
x_t_hat <- x_t - as.numeric(model$fit$residuals)
ei <- as.numeric(model$fit$residuals)
plot(x_t_hat, ei)
abline(h=0)

# -> Var(e) looks good

# qq plot for residuals
n <- length(ei)
qobs <- sort((ei - mean(ei))/sd(ei))
pobs <- (1:n)/(n)
qtheoretical <- qnorm(pobs, mean=0, sd=1) # mean=0 and sd=1 because we standardize ei !!

par(mfcol=c(1, 2))
hist(ei)
plot(qtheoretical, qobs)
abline(a=0, b=1)

# make forecast
astsa::sarima.for(x_t, 16, 2,1,0,0,1,1,4)


