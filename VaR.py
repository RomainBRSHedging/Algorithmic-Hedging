import datetime as dt
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

RNG = np.random.default_rng()



tickers = ['AAPL', 'SPY', 'QQQ', 'MSFT']

df = yf.download(tickers=tickers, start='1995-01-01', end=dt.datetime.now())
df_Adj_close = df['Adj Close']

log_returns = np.log(df_Adj_close/df_Adj_close.shift(1)).dropna()

# Ptflio
Cov_matrix = log_returns.cov()
Cov_matrix_y = log_returns.cov()*252
portfolio_value = 100000 # 100K
weights = np.array([1/len(tickers)]*len(tickers))
nb_days = 30

hist_ret = (log_returns*weights).sum(axis=1)
range_ret = hist_ret.rolling(window = nb_days).sum().dropna()
historical_x_day_returns = hist_ret.rolling(window=nb_days).sum()
portfolio_exp_ret = np.sum(log_returns.mean()*weights)
portfolio_std = np.sqrt(weights.T@Cov_matrix@weights)
portfolio_std_y = np.sqrt(weights.T@Cov_matrix_y@weights)


M = 100000 # nb simulations MC : 100K
scenario_returns = []
for i in range(M):
    W = RNG.normal(0,1)
    scenario_returns.append( portfolio_value*portfolio_exp_ret*nb_days + portfolio_value*portfolio_std*W*np.sqrt(nb_days))

confidence_level = 0.99

# Var: Historical / Monte-Carlo / Parametric
VaR_H = -np.percentile(range_ret, 100 - 100*confidence_level )*portfolio_value
VaR_MC = -np.percentile(scenario_returns, 100*(1-confidence_level) ) 
VaR_Par = -portfolio_value * (norm.ppf(1 - confidence_level) * portfolio_std_y * np.sqrt(nb_days / 252) - hist_ret.mean() * nb_days)

print('Monte-Carlo VaR: ',VaR_MC)
print('Historical VaR: ',VaR_H)
print('Parametric VaR: ',VaR_Par)


#### Plot ####

# Historical Var
return_window = nb_days
range_returns = hist_ret.rolling(window=return_window).sum()
range_returns = range_returns.dropna()
range_returns_dollar = range_returns * portfolio_value
plt.subplot(1,3,1)
plt.hist(range_returns_dollar.dropna(), bins=50, density=True)
plt.xlabel(f'{return_window}-Day Portfolio Return ($)')
plt.ylabel('Frequency')
plt.title(f'Distr of Portf {return_window}-Day Returns ($) H VaR')
plt.axvline(-VaR_H, color='r', linestyle='dashed', linewidth=2, label=f'VaR at {confidence_level:.0%} confidence level')
plt.legend()



# Monte - Carlo Var
### Plot the results of all 10000 scenarios
plt.subplot(1,3,2)
plt.hist(scenario_returns, bins=50, density=True)
plt.xlabel('Scenario Gain/Loss ($)')
plt.ylabel('Frequency')
plt.title(f'Distr of Portf Gain/Loss Over {nb_days} Days MC VaR')
plt.axvline(-VaR_MC, color='r', linestyle='dashed', linewidth=2, label=f'VaR at {confidence_level:.0%} confidence level')
plt.legend()



# Parametric Var
# Convert returns to dollar values for the histogram
historical_x_day_returns_dollar = historical_x_day_returns * portfolio_value
plt.subplot(1,3,3)
# Plot the histogram
plt.hist(historical_x_day_returns_dollar, bins=50, density=True, alpha=0.5, label=f'{nb_days}-Day Returns')
plt.axvline(x=-VaR_Par, linestyle='--', color='r', label='VaR at {}% Confidence'.format(int(confidence_level * 100)))
plt.xlabel(f'{nb_days}-Day Portfolio Return ($)')
plt.ylabel('Frequency')
plt.title(f'Distr of Portf {nb_days}-Day Returns and P VaR')
plt.legend()
plt.show()