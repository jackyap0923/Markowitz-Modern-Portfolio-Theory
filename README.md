# Markowitz-Modern-Portfolio-Theory
This Python code defines classes Markowitz and MVT for implementing portfolio optimization techniques based on Modern Portfolio Theory (MPT). Here's a brief description of each class and its main functionalities:

Markowitz Class:
This class provides methods for computing various statistics and metrics related to portfolio optimization, such as:
 'asset_summaries():' Calculates expected arithmetic returns, variance, covariance matrix, and correlation matrix for a given set of assets.
 'beta():' Calculates the beta of a portfolio or individual assets with respect to a specified market index, like the S&P 500.
 'capm_expected_return():' Estimates the expected return of a portfolio using the Capital Asset Pricing Model (CAPM), considering the risk-free rate and market returns.
 'market_index_returns():' Estimates the expected return of a specific asset based on market index returns using linear regression.

MVT (Minimum Variance Portfolio) Class:
This class extends the Markowitz class and focuses specifically on constructing efficient portfolios:
 'min_var():' Computes the minimum variance portfolio with no constraints.
 'min_var_return():' Computes the minimum variance portfolio with a constraint on expected return.
 'tangent_portfolio():' Calculates the portfolio that maximizes the Sharpe ratio, also known as the tangent portfolio.
 'efficient_frontier():' Generates the efficient frontier, showing the trade-off between portfolio risk and return.

The code heavily relies on financial data obtained from Yahoo Finance using the yfinance library and utilizes various statistical and mathematical techniques, such as covariance matrix calculation, linear regression, and optimization algorithms, to implement Modern Portfolio Theory concepts for portfolio optimization.
