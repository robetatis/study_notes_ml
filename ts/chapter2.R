# https://online.stat.psu.edu/stat510/lesson/2

# *************************************************
# moving average models
# *************************************************

# moving average term -> past error term * coefficient
# w_t -> iid N(0, sigma_w^2)
#
# MA(q) -> q-order moving average model:
#   - 1st order -> x_t = miu + w_t + theta_1*w_tminus1
#   - 2nd order -> x_t = miu + w_t + theta_1*w_tminus1 + theta_2*w_tminus2
#   - q-order   -> x_t = miu + w_t + sum(theta_i*w_tminusi, i=1...q)

# there's really no 'moving average' in MA models; they should be called 'moving
# sum', since what we're doing is taking a weighted sum 

# how are the thetas estimated if we don't have w_t? -> iterative process (Box
# et al. Time Series Analysis: Forecasting and Control):
# 1. A system of q eqns relating theta_k and rho_k is written, where the rho_k
#    are estimated as the sample autocorrelations at lags k = 1....q.
#    Solving this system gives PRELIMINARY values for the theta_k's
# 2. Equations for the w_t's are written by rearranging the model equation:
#    w_t = x_t - miu - theta_1*w_tminus1
#    for t=1 -> w_1 = x_1 - miu - theta_1*w_0
#    for t=2 -> w_2 = x_2 - miu - theta_1*w_1
#    ...
#    for t=T -> w_T = x_T - miu - theta_1*w_Tminus1
#    These can be solved recursively from t=1 to t=T by finding w_0, which is done:
#     - simply by setting w_0 = 0 and then computing forward from t=1 to t=T
#     - through backforecasting
# 3. This gives a preliminary model with preliminary theta_k's and w_t's. Another
#    iteration is needed where theta_k's are re-computed using the previous w_t's
# Another way (Hannan–Rissanen Algorithm) is to:
# 1. fit an AR model
# 2. Compute w_t as the difference between x_t and the AR's estimated x_t
# 3. Use those w_t's to regress x_t against the estimated w_t's
# 4. Further iterations can be done using the result of the last step


# example of MA(1) process
n <- 100
miu <- 10
theta_1 <- 0.7
rho_1_theoretical <- theta_1/(1 + theta_1^2)
x_0 <- 10
w_t <- rnorm(n=n, mean=0, sd=1)
w_tminus1 <- w_t[1:(n - 1)]
x_t <- numeric(length = n)
x_t[1] <- x_0
x_t[2:n] <- miu + theta_1*w_tminus1 + w_t[2:n]
x_t <- x_t[2:n]

par(mfcol=c(1, 2))
plot(x_t, type='o')
acf(x_t, xlim=c(1, 20), ylim=c(-1, 1))
abline(h=rho_1_theoretical, col='gray')

# same as above, but using stats package:
acfMA1 <- ARMAacf(ma=c(0.7), lag.max=10) # theoretical ACF
x_t <- as.numeric(arima.sim(n=150, list(ma=c(0.7)))) + 10 # simulated series
acfMA1_empirical <- acf(x_t, plot=FALSE) # empirical ACF
lags <- 0:10

par(mfcol=c(1, 3))
plot(time(x_t), x_t, type='o', pch=20, main='simulated MA(1)')
plot(lags, acfMA1, xlim=c(1, 10), type='h', ylim=c(-1, 1), main='theoretical ACF')
abline(h=0)
plot(acfMA1_empirical, xlim=c(1, 20), type='h', main='sample ACF', ylim=c(-1, 1))


# properties of the MA(1) model:
#   - expected value E(x_t):
#     E(x_t) = E(miu + w_t + theta_1*w_tminus1)
#     E(x_t) = miu
#   - Variance Var(x_t):
#     Var(x_t) = Var(miu + w_t + theta_1*w_tminus)
#     ...assuming w_t and w_tminus1 are independent, and noting that miu and
#        phi_1 are constants:
#     Var(x_t) = Var(w_t) + theta_1^2*Var(w_tminus)
#     Var(x_t) = sigma_w^2*(1 + theta_1^2)
#   - covariance Cov(x_t, x_tminush): (h is lag)
#     = E[(miu + w_t + theta_1*w_tminus1 - miu) * (miu + w_tminush + theta_1*w_tminush_minus1 - miu)]
#     = E(w_t + theta_1*w_tminus1) * (w_tminush + theta_1*w_tminush_minus1)]
#     = E(w_t*w_tminush + theta_1*w_t*w_tminush_minus1 + theta_1*w_tminus1*w_tminush + theta_1^2*w_tminus1*w_tminush_minus1)
#
#     ...for h = 1:
#     = E(w_t*w_tminus1 + theta_1*w_t*w_tminus2 + theta_1*w_tminus1^2 + theta_1^2*w_tminus1*w_tminus2)
#     = E(w_t*w_tminus1) + theta_1*E(w_t*w_tminus2) + theta_1*E(w_tminus1^2) + theta_1^2*E(w_tminus1*w_tminus2)
#     ...assuming independence of errors -> E(w_t*w_tminus1) = E(w_t)E(w_tminus1) + Cov(w_t, w_tminus1) = 0 + 0:
#     = theta_1*E(w_tminus1^2) 
#     ...since Var(w_t) = E(w_t^2) - (E(w_t))^2 = E(w_t^2):
#     = theta_1*sigma_w^2
# 
#   - autocorrelation lag 1:
#     rho = cov/var = theta_1*sigma_2^2 / (sigma_w^2*(1 + theta_1^2))
#     rho = theta_1/(1 + theta_1^2)
#
#     ...for h = 2 (or higher):
#     = E(w_t*w_tminus2 + theta_1*w_t*w_tminus3 + theta_1*w_tminus1*w_tminus2 + theta_1^2*w_tminus1*w_tminus3)
#     = 0 (due to independence of errors)
#
#     ...this last result is key because it shows that any sample ACF with large 
#        values at lag 1 and low values for lags >= 2 suggests an MA(1) process 
#
#     ...this carries over for higher-order MA models -> nonzero autocorrelations
#        for the first q lags and autocorrelations = 0 for all lags > q.
#
#   - invertibility AR(1) -> MA(inf), doing infinite substitution (simplify 
#     derivation  by using zero-mean model, but same holds without that):
#     x_t = w_t + phi_1*x_tminus1
#     x_t = phi_1^2*x_tminus2 + phi_1*w_tminus1 + w_t
#     x_t = phi_1^3*x_tminus3 + phi_1^2*w_tminus2 + phi_1*w_tminus1 + w_t
#     x_t = phi_1^4*x_tminus4 + phi_1^3*w_tminus3 + phi_1^2*w_tminus2 + phi_1*w_tminus1 + w_t
#     ....
#     x_t = w_t + phi_1*w_tminus1 + phi_1^2*w_tminus2 + phi_1^3*x_tminus3 + ....
#     the AR(k) term phi_1^k*x_tminusk (the 1st one) converges to zero as we go
#     infinitely back in time, which means x_t is dominated only by the w_t (MA) 
#     terms     

# *****************************************************************************
# |phi| < 1 is a requirement for invertibility, since for phi > 1 the AR
# term doesn't vanish as we go back infinintely over time

theta <- seq(-5, 5, by=0.05)
theta_i <- -0.5
rho1_i <- theta_i/(1 + theta_i^2)
rho2_i <- (1/theta_i)/(1 + (1/theta_i)^2)
rho1 <- theta/(1 + theta^2)
rho2 <- (1/theta)/(1 + (1/theta)^2)
plot(theta, rho1)
lines(theta, rho2)
abline(h=0, v=0, col='gray')
abline(v=c(theta_i, 1/theta_i), h=c(rho1_i, rho2_i), col='red')
# ********************************************************************

# MA(2) process 

# properties:
#   - E(x_t) = miu
#   - Var(x_t) = sigma_w^2*(1 + theta_1^2 + theta_2^2)
#   - ACF: rho_1 = (theta_1 + theta_1*theta_2) / (1 + theta_1^2 + theta_2^2)
#          rho_2 = theta_2/(1 + theta_1^2 + theta_2^2)
#          rho_h = 0 for h > 2 (similar to the MA(1) model)

# example of MA(2) model:
# x_t = miu + theta_1*w_tminus1 + theta_2*w_tminus2

n <- 100
miu <- 10
theta_1 <- 0.5
theta_2 <- 0.3
rho_1_theoretical <- (theta_1 + theta_1*theta_2) / (1 + theta_1^2 + theta_2^2)
rho_2_theoretical <- theta_2/(1 + theta_1^2 + theta_2^2)
x_0 <- 10
x_1 <- 10
w_t <- rnorm(n=n, mean=0, sd=1)
w_tminus1 <- w_t[2:(n - 1)]
w_tminus2 <- w_t[1:(n - 2)]
x_t <- numeric(length = n)
x_t[1] <- x_0
x_t[2] <- x_0
x_t[3:n] <- miu + theta_1*w_tminus1 + theta_2*w_tminus2 + w_t[3:n]
x_t <- x_t[3:n]

par(mfcol=c(1, 2))
plot(x_t, type='o')
acf(x_t, xlim=c(1, 20), ylim=c(-1, 1))
abline(h=rho_1_theoretical, col='gray')
abline(h=rho_2_theoretical, col='gray')


# ********************************************************************
# partial autocorrelation PACF
# ********************************************************************

# cov(x_t, x_tminush | x_tminush_minus1, x_tminush_minus2, ...x_tminus1)
# i.e., covariance of original and lagged series taking into account the effect
# of all lags in between

# it follows that for lag 1, partial autocorrelation is the same as autocorrelation
# for higer lags, the PACF gives the autocorrelation at the target lag taking into
# account the effect of the lags in between

# ********************************************************************
# identifying MA and AR models with ACF and PACF
# ********************************************************************

# - we saw earlier that the ACF for MA models is non-zero only for lags
#   1 to q, and then drops abruptly to zero. so, whenever the ACF has 
#   this pattern, it's likely a MA(q) process.
# - if this pattern is observed for the PACF, it's likely an AR(p) process

# -----|---------------------|--------------------|
#      | AR                  | MA                 |
# -----|---------------------|--------------------|
# ACF  | tapers              | shuts off after q  |
# PACF | shuts off after q   | tapers             |
# -----|---------------------|--------------------|

ar_process <- arima.sim(model=list(ar=c(0.7)), n=100, sd=0.2)
ma_process <- arima.sim(model=list(ma=c(0.7)), n=100, sd=0.2)

par(mfrow = c(2, 3))
plot(ar_process, type='o', pch=20, main='AR(1)')
acf(ar_process, main='ACF AR(1)', ylim=c(-1, 1), xlim=c(1, 20))
pacf(ar_process, main='PACF AR(1)', ylim=c(-1, 1), xlim=c(1, 20))
plot(ma_process, type='o', pch=20, main='MA(1)')
acf(ma_process, main='ACF MA(1)', ylim=c(-1, 1), xlim=c(1, 20))
pacf(ma_process, main='PACF MA(1)', ylim=c(-1, 1), xlim=c(1, 20))


# notational conventions
# ******************************

# backshift operator: 
#   - B^k*x_t = x_tminus_k, 
#   - B^k*w_t = w_tminus_k
#   - B^k*theta = theta (constants don't move in time)

# AR polynomial operator:
#   - PHI(B) = 1 - phi_1*B - phi_1*B^2 - ... - phi_q*B^q (q = max. lag) 
#   - AR(1) model can be written as 
#     PHI(B)*x_t = delta + w_t, expanding left-hand side:
#     (1 - phi_1*B)*x_t = delta + w_t, applying operator:
#     x_t - phi_1*x_tminus1 = delta + w_t, rearranging:
#     x_t = delta + phi_1*x_tminus1 + w_t
#   - AR(2) model: 
#     PHI(B)*x_t = delta + w_t 
#     (1- phi_1*B - phi_2*B^2)*x_t = delta + w_t
#     x_t - phi_1*x_tminus1 - phi_2*x_tminus2 = delta + w_t
#     x_t = delta + phi_1*x_tminus1 + phi_2*x_tminus2 + w_t


# MA polynomial operator:
#   - same as AR operator, but applied on w_t and with changed sign
#   - MA(1) model can be written as:
#     x_t = miu + w_t + theta_1*w_tminus1
#     x_t = miu + (1 + theta_1*B)*w_t
#     THETA(B)w*t = x_t - miu, q=1
#   - MA(2):
#     x_t = miu + w_t + theta_1*w_tminus1 + theta_2*w_tminus2
#     x_t = miu + (1 + theta_1*B + theta_2*B^2)*w_t
#     THETA(B)*w_t = x_t - miu, q=2

# ARMA model:
#   x_t = delta + phi_1*x_tminus1 + ... + phi_p*x_tminusp + 
#               + theta_1*w_tminus1 + ... + theta_q*w_tminusq
#               + w_t
#   x_t - delta - phi_1*x_tminus1 - ... - phi_p*x_tminusp = 
#               theta_1*w_tminus1 + ... + theta_q*w_tminusqs
#               + w_t
#   noting that delta = miu*(1 - phi_1 - phi_2 - ... - phi_p), with miu = E(x_t),
#   and that miu*(1 - phi_1 - phi_2 - ... - phi_p) = PHI(B)*(miu)
#   and also that PHI(B)() is a linear operator, and thus 
#     PHI(B)(x_t) - PHI(B)(miu) = PHI(B)(x_t - miu), we then get
#   (1 - phi_1*B - ... - phi_p*B^p)*(x_t - miu) =
#               (1 + theta_1*B + ... + theta_q*B^q)*w_t
#   finally,
#   PHI(B)*(x_t - miu) = THETA(B)*w_t

# differencing:
#   - x_t - x_tminusk = (1 - B^k)*x_t = ▽_k*x_t
#   - ▽^2*x_t = (1 - B)^2*x_t = (1 - 2B + B^2)*x_t = x_t - 2*x_tminus1 + x_tminus2



