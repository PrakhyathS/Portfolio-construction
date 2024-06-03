#import 
import numpy as np
import pandas as pd
import warnings 
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

from scipy.linalg import block_diag
from scipy.linalg import sqrtm

import cvxpy as cp

import empyrical as ep

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

import pyfolio as pf

from joblib import Parallel, delayed

import plotly.graph_objs as go
from plotly.subplots import make_subplots

#handling data to conver into date time if not present
class DataHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def read_csv(self):
        try:
            self.data = pd.read_csv(self.file_path)
            return self.data
        except Exception as e:
            print(f"Error reading CSV file: {e}")

    def convert_to_datetime_and_set_index(self, column_name):
        try:
            self.data[column_name] = pd.to_datetime(self.data[column_name])
            self.data.set_index(column_name, inplace=True)
            return self.data
        except Exception as e:
            print(f"Error converting column '{column_name}' to datetime and setting index: {e}")
            
    def Calculating_daily_returns(self, data):
        try:
            assets = data.columns
            daily_returns = data[assets].pct_change().dropna()
            return daily_returns
        except Exception as e:
            print(f"Error calculating daily returns: {e}")

class PortfolioAnalysis:
    def __init__(self, Y, assets): #Y is the return data frame, assets is Y.columns
        self.Y = Y
        self.assets = assets
        self.T, self.N = Y.shape
        
    def coskewness_matrix(self, Y):
        Y_ = np.array(Y, ndmin=2)
        T, self.n = Y_.shape
        mu = np.mean(Y_, axis=0).reshape(1, -1)
        mu = np.repeat(mu, T, axis=0)
        x = Y_ - mu
        ones = np.ones((1, self.n)) 
        z = np.kron(ones, x) * np.kron(x, ones)
        M3 = 1 / T * x.T @ z
        return self.n,M3 

    def square_coskewness(self, M3):
        N, _ = M3.shape
        S3 = np.empty((0, 0))
        for j in range(0, N):
            S3 = block_diag(S3, M3[:, j * N:(j + 1) * N])
        return S3 

    def extract_block_diag(self,S3, n):
        N , T = S3.shape
        if N!= T:
            raise ValueError ('S3 must be a square matrix')
        Si = []
        for i in range(n):
            Si.append(S3[i*n:(i+1)*n,i*n:(i+1)*n])
            return Si
    
        
    def calculate_statistics(self):
        self.mu = self.Y.mean().to_numpy().reshape(-1, 1)
        self.Sigma = self.Y.cov().to_numpy()

        self.n, self.M3 = self.coskewness_matrix(self.Y) 
        self.S3 = self.square_coskewness(self.M3)

        s, V = np.linalg.eig(self.S3)
        s2 = np.clip(s, -np.inf, 0)
        self.S32 = V @ np.diag(s2) @ V.T
        s3 = np.clip(s, 0, np.inf)
        self.S33 = V @ np.diag(s3) @ V.T

        return self.mu, self.Sigma, self.n, self.M3, self.S3, self.S32, self.S33

    def generate_samples(self):
        rs = np.random.RandomState(seed=123)
        samples = []
        for alpha in [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
            sample = rs.dirichlet([alpha] * len(self.assets), 4000)
            sample /= sample.sum(axis=1).reshape(-1, 1)
            samples.append(sample)
        return np.concatenate(samples, axis=0)

    def calculate_portfolio_metrics(self, sample):
        mean_1 = sample @ self.mu
        std_1 = np.linalg.norm(sqrtm(self.Sigma) @ sample.T, ord=2, axis=0)
        skewness_1 = []
        w = np.random.dirichlet([0.8] * self.N, size=1).reshape(-1, 1)

        for i in range(len(sample)):
            w = sample[i].reshape(-1, 1)
            w_w = np.kron(w, w)
            skew_1 = w.T @ self.M3 @ w_w
            skew_1 = np.abs(skew_1) ** (1 / 3) * np.sign(skew_1)
            skewness_1.append(skew_1.item())

        return mean_1, std_1, skewness_1
    
    
    def calculate_negative_quadratic_skewness(self, S32, n, strategy='long_only'):
        x = cp.Variable((self.N, 1))
        g = cp.Variable()

        S32i = self.extract_block_diag(S32,n)

        if strategy == 'long_only':
            constraints = [x >= 0,
                           cp.sum(x) == 1,
                           cp.SOC(g, sqrtm(-np.sum(S32i, axis=0)) @ x)
                          ]
        elif strategy == 'long_short':
            constraints = [x >= -1,
                           x <= 1,
                           cp.sum(x) == 1,
                           cp.SOC(g, sqrtm(-np.sum(S32i, axis=0)) @ x)]

        obj = cp.Minimize(g**2)
        prob = cp.Problem(obj, constraints)
        prob.solve()

        weights = pd.DataFrame(x.value, index=self.assets, columns=['Negative_quadratic_skewness'])

        return weights
    
    def calculate_minimum_variance_weights(self, Sigma, strategy='long_only'):
        x = cp.Variable((self.N, 1))
        g = cp.Variable(nonneg=True)

        G = sqrtm(Sigma)

        if strategy == 'long_only':
            constraints = [cp.sum(x) == 1,
                           x >= 0,
                           cp.SOC(g, G @ x)]
        elif strategy == 'long_short':
            constraints = [cp.sum(x) == 1,
                           x >= -1,
                           x <= 1,
                           cp.SOC(g, G @ x)]

        obj = cp.Minimize(g)
        prob = cp.Problem(obj, constraints)
        prob.solve()

        weights = pd.DataFrame(x.value, index=self.assets, columns=['Minimum_variance'])

        return weights
    

    
    def calculate_min_var_neg_Q_skew_weights(self, Sigma, S32, n, skewness_coefficient=8, strategy='long_only'):
        x = cp.Variable((self.N, 1))
        g = cp.Variable(nonneg=True)
        g2 = cp.Variable(nonneg=True)
        G = sqrtm(Sigma)
        S32i = self.extract_block_diag(S32,n)
        if strategy == 'long_only':
            constraints = [cp.sum(x) == 1,
                           x >= 0,
                           cp.SOC(g2, G @ x),
                           cp.SOC(g, sqrtm(-np.sum(S32i, axis=0)) @ x)]
        elif strategy == 'long_short':
            constraints = [cp.sum(x) == 1,
                           x >= -1,
                           x <= 1,
                           cp.SOC(g2, G @ x),
                           cp.SOC(g, sqrtm(-np.sum(S32i, axis=0)) @ x)]

        obj = cp.Minimize(g2 + skewness_coefficient * g)
        prob = cp.Problem(obj, constraints)
        prob.solve()

        weights = pd.DataFrame(x.value, index=self.assets, columns=['MV+NQSkewness'])

        return weights
    
    def calculate_portfolio_returns(self, weights_list, next_month_data=None):
        portfolio_returns = []
        if next_month_data is None:
            for i, col in enumerate(weights_list.columns):
                weighted_returns = self.Y.dot(weights_list[col])
                weighted_returns_df = pd.DataFrame(weighted_returns, columns=[col])
                portfolio_returns.append(weighted_returns_df)
        else: #this is for rebalancing :using the calculated weights in the next month/year 
            for col in weights_list.columns:
                weighted_returns = next_month_data.dot(weights_list[col])
                weighted_returns_df = pd.DataFrame(weighted_returns, columns=[col])
                portfolio_returns.append(weighted_returns_df)
        return pd.concat(portfolio_returns, axis=1)
    
    def calculate_performance_ratios(self, portfolio_returns_all,Period):
        performance_ratios = pd.DataFrame(columns=portfolio_returns_all.columns,
                                          index=['Annual Returns', 'Volatility', 'Max Drawdown', 'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Var', 'CVar'])

        for column in portfolio_returns_all.columns:
            returns = portfolio_returns_all[column]
            performance_ratios.loc['Annual Returns', column] = ep.annual_return(returns,period=Period)
            performance_ratios.loc['Volatility', column] = ep.annual_volatility(returns,period=Period)
            performance_ratios.loc['Max Drawdown', column] = ep.max_drawdown(returns)
            performance_ratios.loc['Sharpe Ratio', column] = ep.sharpe_ratio(returns,period=Period)
            performance_ratios.loc['Sortino Ratio', column] = ep.sortino_ratio(returns,period=Period)
            performance_ratios.loc['Calmar Ratio', column] = ep.calmar_ratio(returns,period=Period)
            performance_ratios.loc['Var', column] = ep.value_at_risk(returns)
            performance_ratios.loc['CVar', column] = ep.conditional_value_at_risk(returns)

        return performance_ratios

def main():
    # Example usage:
    file_path = 'Filtered_data.csv'
    handler = DataHandler(file_path)
    data = handler.read_csv()
    data=handler.convert_to_datetime_and_set_index('Dates')
    Y=handler.Calculating_daily_returns(data=data)
    #Y.head()

    portfolio=PortfolioAnalysis(Y,Y.columns)
    mu, Sigma, n, M3, S3, S32, S33 = portfolio.calculate_statistics() #correct output
    sample=portfolio.generate_samples()
    mean_1, std_1, skewness_1 = portfolio.calculate_portfolio_metrics(sample=sample) 

    method='long_only'
    weights_1=portfolio.calculate_negative_quadratic_skewness(S32,n, strategy=method)
    weights_2=portfolio.calculate_minimum_variance_weights(Sigma, strategy=method)
    weights_3=portfolio.calculate_min_var_neg_Q_skew_weights(Sigma, S32,n,skewness_coefficient=5, strategy=method)
    weights_list = pd.concat([weights_1, weights_2, weights_3], axis=1)
    
    return portfolio.calculate_performance_ratios(portfolio.calculate_portfolio_returns(weights_list),'daily')

if __name__ == "__main__":
     print("Wait for few minutes")
     results = main()
     print("Performace Results",results)
     print("Thank you")
