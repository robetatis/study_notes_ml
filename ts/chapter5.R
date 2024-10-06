# https://online.stat.psu.edu/stat510/lesson/5

# *************************************************
# decomposition
# *************************************************

# used to separate trend, seasonal factors, long-run cycles, holiday effects, etc.

# structures:
#   - additive: x_t = trend + seasonality + random. used for stable seasonality
#   - multiplicative: x_t = trend * seasonality * random. used for increasing seasonality

# examples: australian beer prod., johnson&johnson earnings
library(fpp); data(ausbeer)

par(mfrow=c(1, 2))
plot(JohnsonJohnson, type='o', pch=20, main='J&J - increasing seasonality')
plot(ausbeer, type='o', pch=20, main='Beer - constant seasonality')

# steps for decomposition:

# 1. estimate trend. use either smoothing procedure or some regression
# 2. de-trend. subtract for additive, divide for multiplicative
# 3. estimate seasonal effect. compute average per season
# 4. de-seasonalize: subtract for additive, divide for multiplicative
# 5. compute random component (remaining variation):
#     additive: random = series - (trend + seasonality)
#     multiplicative: random = series/(trend*seasonality)

decomp <- function(time_series, mode='additive'){
  
  # put data into data.frame
  x <- data.frame(
    t = as.numeric(time(time_series)),
    quarter = as.numeric(cycle(time_series)),
    x_t = as.numeric(time_series)
  )
  
  # compute lowess trend
  x$trend <- stats::lowess(x=x$t, y=x$x_t, f=0.1)$y
  
  if(mode == 'additive'){
    
    x$detrended <- x$x_t - x$trend # detrend
    x$seasonality <- ave(x=x$detrended, x$quarter, FUN=mean) # seasonality
    x$random <- x$x_t - (x$trend + x$seasonality) # residual
  
    } else {
    if(mode == 'multiplicative'){
      
      x$detrended <- x$x_t / x$trend # detrend
      x$seasonality <- ave(x=x$detrended, x$quarter, FUN=mean) # seasonality
      x$random <- x$x_t / (x$trend * x$seasonality) # residual
      
    } else {
      stop("mode must be one of c('additive', 'multiplicative')")
    }
  }
  
  return(x)
}

ausbeer_decomp <- decomp(ausbeer, mode='additive')
jj_decomp <- decomp(JohnsonJohnson, mode='multiplicative')

windows(12, 6)
par(mfcol=c(2, 2))
plot(ausbeer_decomp$t, ausbeer_decomp$x_t, type='o', pch=20)
lines(ausbeer_decomp$t, ausbeer_decomp$trend, col='blue')
lines(ausbeer_decomp$t, ausbeer_decomp$trend + ausbeer_decomp$seasonality, col='red')
plot(ausbeer_decomp$t, ausbeer_decomp$random)
abline(h=0)
plot(jj_decomp$t, jj_decomp$x_t, type='o', pch=20)
lines(jj_decomp$t, jj_decomp$trend, col='blue')
lines(jj_decomp$t, jj_decomp$trend * jj_decomp$seasonality, col='red')
plot(jj_decomp$t, jj_decomp$random)
abline(h=1)


# *************************************************
# smoothing (aka filtering)
# *************************************************

# used to identify components (trend, seasonality, etc.)

# moving average: 
#   - weighted sum of x_t at surrounding times
#   - if we want to smooth out seasonality, use S neighbors

# example with australian beer production
x_t <- as.numeric(ausbeer)
x_t_filtered <- stats::filter(x_t, filter=c(1/8, 1/4, 1/4, 1/4, 1/8), sides=2)

par(mfcol=c(1, 2))
plot(x_t, pch=20, type='o')
lines(x_t_filtered, pch=20, type='o', col='red')

# to see de-trended seasonal data, remove trend from raw values
x_t_seasonal <- x_t - x_t_filtered  

plot(x_t_seasonal, pch=20, type='o')
abline(h=0, col='red')


# example with US unemployment
x_t <- as.matrix(read.csv('data/USUnemployment.csv', sep=','))[, 2:13]
x_t <- ts(data=x_t, start=1948, frequency=1)
x_t_filtered <- stats::filter(as.numeric(x_t), filter=rep(1/12, 12), sides=2)

plot(as.numeric(x_t), type='l')
lines(x_t_filtered, col='red', lwd=2)


# *************************************************
# exponential smoothing
# *************************************************

# x_hat_tplus1 = alpha*x_t + (1 - alpha)*x_hat_t 
# called exponential smoothing because substituting x_hat_t all the way back to the
# beginning of the series gives:
# x_hat_tplus1 =   alpha*x_t 
#                + alpha*(1 - alpha)*x_tminus1
#                + alpha*(1 - alpha)^2*x_tminus2
#                + alpha*(1 - alpha)^3*x_tminus3
#                ...
#
#                +alpha*(1 - alpha)^(t-1)*x_1

# -> we estimate x_tplus1 as a weighted sum of the observed x_t, whereby the 
#    weight of the past values decays exponentially as alpha*(1 - alpha)^j as 
#    we move backwards into the past

# this is actually an ARIMA(0, 1, 1) with miu = 0:
#   x_t - x_tminus1 = theta*w_tminus1 + w_t
#   x_t = x_tminus1 + theta*w_tminus1 + w_t
#   x_tplus1 = x_t + theta*w_t + w_tplus1
#   ... since w_tplus1 = x_tplus1 - x_hat_tplus1
#   x_tplus1 = x_t + theta*(x_t - x_hat_t) + x_tplus1 - x_hat_tplus1
#   x_hat_tplus1 = x_t + theta*(x_t - x_hat_t)
#   x_hat_tplus1 = (1 + theta)*x_t - theta*x_hat_t 
#   ... setting alpha = 1 + theta:
#   x_hat_tplus1 = alpha*x_t + (1 - alpha)*x_hat_t 

# example
x_t <- oil
model <- astsa::sarima(x_t, p=0, d=1, q=1, no.constant=TRUE) # fit ARIMA(0, 1, 1)
alpha <- 1 + model$fit$coef[1] # compute alpha
x_hat_t <- x_t - model$fit$residuals # compute x_hat

# compute smoothed values
x_hat_tplus1 <- alpha*x_t + (1 - alpha)*x_hat_t

# plot
plot(x_t, type='o', pch=20)
lines(x_hat_tplus1, col='red')
lines(x_hat_t, col='blue')

par(mfcol=c(1, 2))
acf(x_t, xlim=c(1, 20), ylim=c(-1, 1))
pacf(x_t, xlim=c(1, 20), ylim=c(-1, 1))

# double exponential smoothing -> used when there's trend and no seasonality



