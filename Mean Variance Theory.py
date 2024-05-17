import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 

class Markowitz():
    def __init__(self, ticker:list, price:list, quantity:list):
        self.ticker = ticker
        self.price = np.array(price)
        self.quantity = np.array(quantity)

    #Pandas does it column wise where each row is an observatoin
    #numpy does it row wise where each column is an observation
    """ Uses data from the module yfinance to obtain closing prices and compute the geometric average. Geometric returns are then used
        to obtain the covariance matrix, variace and correlation matrix. Default for period and interval
        is 5 years with interval of 1d. Assuming we have 252 days per year we multiply by 252 to obtain the annualized values

        Returns:
        expected_arithmetic_returns: expected arithmetic returns of each asset per year in %
        cov_matrix: covariance and variance per year
        portfolio_returns: returns of the portfolio for each day in %
        portfolio_variance: variance of the portfolio in %
        proportions: proportions invested in each asset
    """
    def asset_summaries(self,period='5y',interval='1d'):
        ticker = self.ticker
        price = self.price
        quantity = self.quantity
        ticker_prices  = yf.download(ticker, period=str(period), interval= str(interval))
        closing_prices = (ticker_prices.Close).to_numpy()
        
        returns_matrix = np.zeros((closing_prices.shape[0]-1,closing_prices.shape[1]))
        #obtains the return matrix
        for i  in range(closing_prices.shape[1]):
            prices_ = closing_prices[:,i]
            for j in range(closing_prices.shape[0]-1):
                returns_matrix[j,i] = (prices_[j+1] - prices_[j])/prices_[j]*100

        returns_df = pd.DataFrame(returns_matrix)
        returns_matrix = returns_df.to_numpy()
        #Proportions invested in each asset
        capital_invested = np.dot(price,quantity)
        n = len(price)
        proportions = np.zeros((n,1))
        for i in range(n):
            proportions[i,0] = price[i]*quantity[i]/capital_invested
        
        #Obtaining market closing prices 
        #Calculating expected returns of portfolio based on closing prices
        expected_arithmetic_returns = returns_df.mean(axis=0)*252
        cov_matrix = returns_df.cov()*252
        portfolio_returns = returns_matrix @ proportions 
        portfolio_variance = proportions.T @ cov_matrix @ proportions

        return (expected_arithmetic_returns,cov_matrix,portfolio_returns,portfolio_variance[0],returns_df,proportions) 

    """Calculates the Beta of any market or asset if ticker available in yahoo finance.

        Returns:
        beta: beta of the market portfolio
    """
    def beta(self,period='5y',interval='1d',market ='SPY'):
        # Calculates the proportions invested in each asset and defining variables
        returns_matrix = self.asset_summaries(period,interval)[4].to_numpy()
        proportions = self.asset_summaries(period,interval)[5]
        
        #Obtaining market closing prices 
        market_portfolio_prices = yf.download(market,period=period,interval=interval).Close
        #Calculating expected returns of portfolio based on closing prices
        portfolio_returns = returns_matrix @ proportions
        #-1 indicates the size of the dimension should be inferred from the length of the array of the specified dimensions
        #calculate returns of the market
        market_returns = (market_portfolio_prices.pct_change().dropna()*100).to_numpy()
        
        cov_market_returns = np.cov(market_returns.squeeze(),portfolio_returns.squeeze())[0,1]
        var_market_returns = np.var(market_returns.squeeze())
        beta = cov_market_returns/var_market_returns

        return(beta)
    
    """
    Sharpe Ratio = E[R_p - r_f]/sigma_p
    Assuming risk free rate is constant, we have (E[R_p] - r_f)/sigma_p

    Returns:
    sharpe: Sharpe Ratio
    """
    def sharpe_ratio(self,period='5y',interval='1d'):
        portfolio_return = self.asset_summaries()[2]
        portfolio_sd = np.sqrt(self.asset_summaries()[3])[0]
        expected_portfolio_returns = np.sum(portfolio_return)/len(portfolio_return) *252
        if period == '5y':
            risk_free_rate = yf.download("^IRX",period='1d',interval='1d').Close #5 year treasury bond
        elif period != '5y':
            risk_free_rate = yf.download("^FVX",period='1d',interval='1d').Close #13 week treasury bill
        sharpe = (expected_portfolio_returns - risk_free_rate[0])/portfolio_sd

        return(sharpe)

    """Market Returns = Returns from S&P 500 as defaut. Obtains latest interest rate based on the period to calculate capm expected returns

    Return:
    expected_portfolio_return: CAPM expected return of the portfolio in %    
    """
    def capm_expected_return(self,period='5y',interval='1d', market='SPY'):
        beta = self.beta(period,interval,market)
        market_portfolio_prices = yf.download(market,period,interval).Close
        market_return = (market_portfolio_prices.pct_change().dropna()*100).to_numpy().reshape((-1,1))
        expected_market_return = np.mean(market_return)
        if period == '5y':
            risk_free_rate = yf.download("^IRX",period='1d',interval='1d').Close #5 year treasury bond
        elif period != '5y':
            risk_free_rate = yf.download("^FVX",period='1d',interval='1d').Close #13 week treasury bill

        expected_portfolio_return = risk_free_rate + beta*(expected_market_return - risk_free_rate.iloc[0])

        return(expected_portfolio_return)
    

    """Assuming all factors are independent and equal weights in factors, and calculates the expected returns based on the factors.
    
    Return:
    alpha: excess returns of the portfolio in %
    beta: coefficients of each of the factors
    """
    def alpha_returns(self,period='5y',interval='1d',factors=['SPY']):
        portfolio_returns = self.asset_summaries()[2]
        factors_prices = yf.download(factors, period=str(period), interval= str(interval)).Close
        n = len(factors)

        if period == '5y':
            risk_free_rate = yf.download("^IRX",period='1d',interval='1d').Close #5 year treasury bond
        elif period != '5y':
            risk_free_rate = yf.download("^FVX",period='1d',interval='1d').Close #13 week treasury bill
        
        #For Training for linear regression
        factor_returns = (factors_prices.pct_change().dropna()*100).to_numpy().reshape((-1,n)) - risk_free_rate.iloc[0]

        #Obtain the coefficients of each factor and calculate the expected returns for asset
        model = LinearRegression().fit(factor_returns,portfolio_returns-risk_free_rate.iloc[0])
        alpha = model.intercept_
        beta = model.coef_
        return(alpha,beta)


class MVT(Markowitz):
    """Period and intervals for obtaining closing prices"""
    def __init__(self, ticker: list, price, quantity,period,interval):
        super().__init__(ticker,price,quantity) #Call parent's class __init__ method with only 'ticker'
        self.n = len(ticker)
        self.period = period
        self.interval = interval
        self.cov_matrix = self.asset_summaries(period,interval)[1]
        self.asset_returns = np.array([self.asset_summaries(period,interval)[0]]).T #asset expected returns
        self.one_vector = np.ones((self.n,1))
        
    """Minimum Variance Portfolio
    
    Return:
    porfolio: proportions invested in each asset in our portfolio
    portfolio_std: sigma of our minimum variance portfolio
    portfolio_return: return of our minimum variance portfolio
    """
    def min_var(self):
        n = self.n
        cov_matrix = self.cov_matrix
        one_vector = self.one_vector
        zero_vector = np.zeros((n,1))
        asset_returns = self.asset_returns
        constraint_matrix = one_vector
        
        b = np.concatenate((zero_vector,[[1]]),axis=0)
        a1 = np.concatenate((one_vector,[[0]]),axis=0)
        a2 = np.concatenate((2*cov_matrix,-constraint_matrix),axis=1)
        A = np.concatenate((a2,a1.T),axis=0) 
        x = np.linalg.solve(A,b) 
        portfolio = x[:n,:]
        portfolio_std = np.sqrt(portfolio.T @ cov_matrix @ portfolio)
        portfolio_return = portfolio.T @ asset_returns

        return(portfolio, portfolio_std[0], portfolio_return[0,0])
    
    """Minimum variance portfolio wiht constraint on expected return. Do not need to return portfolio_return as return of portfolio is constraint.
    
    Return:
    portfolio: proportions of minimum variance portfolio with constraint on expected return
    portfolio_std: sigma of our minimum variance portfolio with constraint on expected return
    """
    def min_var_return(self,expected_return):
        n = self.n
        cov_matrix = self.cov_matrix
        one_vector = self.one_vector
        asset_returns = self.asset_returns
        zero_vector = np.zeros((n,1))
        constraint_matrix = np.concatenate((one_vector,asset_returns),axis=1)
        b = np.concatenate((zero_vector,[[1]],[[expected_return]]),axis=0)
        a1 = np.concatenate((constraint_matrix,np.zeros((2,2))),axis=0)
        a2 = np.concatenate((2*cov_matrix,-constraint_matrix),axis=1)
        A = np.concatenate((a2,a1.T),axis=0) 
        x = np.linalg.solve(A,b) 

        portfolio = x[:n,:]
        portfolio_std = np.sqrt(portfolio.T @ cov_matrix @ portfolio)
        return(portfolio, portfolio_std[0])
    
    """Composition of Tangent Portfolio wiht risk-freee rate determined by the period of investment
    
    Return:
    tangent_portfolio: proportions invested in the tangent portfolio of efficient frontier
    portfolio_return: return on tangent portfolio
    portfolio_std: sigma of tangent portfolio
    """
    def tangent_portfolio(self):
        period = self.period
        if period == '5y':
            risk_free_rate = yf.download("^IRX",period='1d',interval='1d').Close
        elif period != '5y':
            risk_free_rate = yf.download("^FVX",period='1d',interval='1d').Close

        asset_returns = self.asset_returns
        cov_matrix = self.cov_matrix
        constraint_matrix = np.array(asset_returns) - risk_free_rate[0]
        cov_matrix_inverse = np.linalg.inv(cov_matrix)
        
        tangent_portfolio = (cov_matrix_inverse @ constraint_matrix)/np.sum(cov_matrix_inverse @ constraint_matrix)
        portfolio_return = tangent_portfolio.T @ asset_returns
        portfolio_std = np.sqrt(tangent_portfolio.T @ cov_matrix @ tangent_portfolio)

        return(tangent_portfolio,portfolio_return[0],portfolio_std[0])

    """n is the number of efficient portfolios to create the efficient frontier
    
    Return:
    plots the efficient frontier
    """
    def efficient_frontier(self,no_of_points=1000): 
        period = self.period
        returns = np.arange(0,100,100/no_of_points)
        sigma = np.zeros(no_of_points)
        tangent_return = self.tangent_portfolio()[1]
        tangent_sigma = self.tangent_portfolio()[2]
        for i in range(no_of_points):
            sigma[i] = self.min_var_return(returns[i])[1]
        if period == '5y':
            risk_free_rate = yf.download("^IRX",period='1d',interval='1d').Close
        elif period != '5y':
            risk_free_rate = yf.download("^FVX",period='1d',interval='1d').Close
        
        asset_returns = self.asset_returns
        cov_matrix = self.cov_matrix
        price = self.price
        quantity = self.quantity
        capital_invested = np.dot(price,quantity)
        n = len(price)
        proportions = np.zeros((n,1))
        for i in range(n):
            proportions[i] = price[i]*quantity[i]/capital_invested
        
        portfolio_return = proportions.T @ asset_returns
        portfolio_std = np.sqrt(proportions.T @ cov_matrix @ proportions)
        
        plt.plot(sigma,returns,label="Efficient Frontier")
        plt.scatter(portfolio_std,portfolio_return)
        plt.scatter(0,risk_free_rate)
        plt.scatter(tangent_sigma,tangent_return)
        plt.plot([0,tangent_sigma],[risk_free_rate,tangent_return])
        plt.title("Efficient Frontier")
        plt.xlabel("Sigma (%)")
        plt.ylabel('Expected Portfolio Return (%)')
        #Shows your portfolio
        plt.show()
    
    
    
    
