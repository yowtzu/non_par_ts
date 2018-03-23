import numpy as np
import pandas as pd
import itertools

from .fit import *


def to_seconds(index):
    """Given a TS index, return the seconds from Jan 1st, 2000."""
    return (index - pd.Timestamp('2000-01-01', tz=index.tz)).total_seconds()


def basis_apply(func, period, harmonic, index_count):
    return func(2 * np.pi * harmonic * index_count / period)


class Harmonic(object):
    def __init__(self, periods_seconds=[], max_harmonics=5):
        self.periods_seconds = periods_seconds
        self.max_harmonics = max_harmonics
        
    def bases_iter(self, n_harmonics):
        bases = list(itertools.product((np.sin, np.cos), 
                                  self.periods_seconds, 
                                  range(1, n_harmonics+1)))
        return itertools.chain([(np.cos, 1, 0.)], bases)        
        
    def featurize(self, data_index, n_harmonics):
        """Makes features matrix for index, iter of periods (in sec.), and num. of harmonics."""
        
        n_features = 1 + len(self.periods_seconds) * n_harmonics * 2
        index_seconds = to_seconds(data_index)
        X = np.empty((len(data_index), n_features))
        for i, base in enumerate(self.bases_iter(n_harmonics)):
            X[:, i] = basis_apply(*base, index_seconds)
        return X

    # def new_fit(self, data, train_frac = .75):

    #     def fit_func(n_harmonics):
    #         self.X_tr = self.featurize(train.index, n_harmonics)
    #         return np.linalg.solve(X_tr.T @ X_tr, X_tr.T @ y_tr)

    #     def val_cost_func(beta):
    #         X_val = self.featurize(val.index, n_harmonics)

    #     hyperparameter_grid_search(fit_func, val_cost_func, *hp_iterables)
        
    def fit(self, data, train_frac = .75, last_is_test = False):
        
        train, val = split_data(data, train_frac, last_is_test, seed = 0)
        y_tr = train.values
        y_val = val.values
        
        self.betas = {}
        self.val_costs = {}
        
        for n_harmonics in range(1, self.max_harmonics+1):
            X_tr = self.featurize(train.index, n_harmonics)
            X_val = self.featurize(val.index, n_harmonics)

            self.betas[n_harmonics] = np.linalg.solve(X_tr.T @ X_tr, X_tr.T @ y_tr)
            self.val_costs[n_harmonics] = RMSE(X_val@self.betas[n_harmonics],  y_val)
            
        self.beta = self.betas[self.best_n_harmonic]
            
    @property
    def best_n_harmonic(self):
        return min_val_key(self.val_costs)

    def predict(self, test_index):
        X = self.featurize(test_index, self.best_n_harmonic)
        return pd.Series(index=test_index, data = X@self.beta)