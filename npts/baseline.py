import datetime
import time
import pandas as pd
import numpy as np
import itertools
import scipy.sparse as sp
import scipy.sparse.linalg as spl


def build_cyclic_diff(N, diff):
    """Build cyclic diff matrices for regularization."""
    # print(f'Building matrix of size {N} with diff {diff}.')
    return sp.eye(N) - sp.eye(N, k=diff) - sp.eye(N, k=diff-N)


def build_block_diag_diff(N, diff, period):
    num_blocks = N//period
    assert (N/period == N//period)
    # print(f'Building block diag. with {num_blocks} blocks.')
    return sp.block_diag([build_cyclic_diff(period, diff)]*num_blocks)


class Baseline(object):
    def __init__(self, *periods):
        """Multi periodic smoother
        Args:
            - periods: objects that define n_periods and indexer
            #- indexers is sequence of functions, to apply to data.index,
            #        that return indexes (of each periodicity)
            #- n_periods is number of periods for each
        """
        self.indexers = [el.indexer for el in periods]
        self.periods = [el.n_periods for el in periods]
        self.K = len(periods)
        self.verbose = False ## TODO fix this
        self.lambdas = [el.lambdas for el in periods]
        self.cum_periods = np.concatenate([[1],
            np.cumprod(self.periods)]).astype(int)
        self.N = int(np.product(self.periods))
        assert (self.N == self.cum_periods[-1])
        if self.verbose:
            print(f'Building baseline with {self.N} = ' +
              f'{"*".join(str(el) for el in self.periods)} values.')
        self.make_reg_mats()

    def make_reg_mats(self):
        """Build cyclic regularization matrices."""
        self.reg_matrices = []
        for i in range(len(self.cum_periods)-1):
            diff = self.cum_periods[i]
            period = self.cum_periods[i+1]
            self.reg_matrices.append(build_block_diag_diff(self.N, diff, period))

    def _indexer(self, index):
        return np.sum((indexer(index) * self.cum_periods[i] for
                    i, indexer in enumerate(self.indexers)))

    def make_LS_cost(self, data):
        """Build LS penalty part of the system.

        Args:"""
        NDATA = len(data)
        X = sp.coo_matrix((np.ones(NDATA),
                          (range(NDATA), self._indexer(data.index))),
                          shape = (NDATA, self.N))/np.sqrt(NDATA)
        y = data.values/np.sqrt(NDATA)
        return X, y

    def _build_system(self, λ, X_tr, y_tr):
        ## TODO fix this
        X = [[X_tr]]
        X += [[np.sqrt(λ[k]/self.N)*self.reg_matrices[k]]
          for k in range(self.K)]
        X = sp.bmat(X)
        y = np.vstack([np.matrix(y_tr).T, np.zeros((self.N*self.K, self.l))])
        return X, y

    def _compute_gradient(self, λ, theta, X, X_val, val_res):

        grad_base = np.zeros((self.N, self.l))
        grad = np.zeros((len(λ)))

        right_side = -2 * X_val.T @ val_res

        for j in range(self.l):
            grad_base[:, j], status = spl.cg(X.T @ X, right_side[:, j],
                                       x0=self._grad_cache[:, j])
            if status != 0:
                raise Exception("CG failed.")
            self._grad_cache[:, j] = grad_base[:, j]

        for i in range(len(λ)):
            ## TODO cache this
            X_i = self.reg_matrices[i]/np.sqrt(self.N)
            grad[i] = theta.T@X_i.T@X_i@grad_base * λ[i]

        return grad

    def _solve_problem(self, λ, loss_terms, compute_grad = True):

        if self.verbose:
            print(f'solving with u={np.log(λ)}, so λ={λ}')

        theta = np.zeros((self.N, self.l))

        X_tr, y_tr, X_val, y_val = loss_terms
        X, y = self._build_system(λ, X_tr, y_tr)

        for j in range(self.l):
            theta[:, j], status = spl.cg(X.T @ X, X.T @ y[:, j],
                                   x0=self._theta_cache[:, j])
            if status != 0:
                raise Exception("CG failed.")
            self._theta_cache[:, j] = theta[:, j]

        val_res = X_val@theta - y_val
        val_cost = np.sum(val_res**2)  # TODO maybe need to change for 2dim
        tr_res = X_tr@theta - y_tr
        tr_cost = np.sum(tr_res**2)

        return val_cost, tr_cost, theta, \
            self._compute_gradient(λ, theta, X, X_val, val_res) \
                if compute_grad else None

    def grid_search_and_numerical_opt(self, loss_terms):

        for lambda_val in itertools.product(*self.lambdas):

            if self.verbose:
                print(f'working with lambda {lambda_val}')
            opt_dimension = sum([el is None for el in lambda_val])

            if opt_dimension == 0:
                λ = tuple(lambda_val)
                val_cost, tr_cost, theta, grad = \
                    self._solve_problem(λ, loss_terms)
                self.tr_costs[λ] = tr_cost
                self.val_costs[λ] = val_cost
                self.thetas[λ] = theta

            else:
                if self.verbose:
                    print(f'numerically optimizing on {opt_dimension} lambdas')
                def min_func(u_opt):
                    u = np.array(lambda_val)
                    u[[e is None for e in lambda_val]] = np.exp(u_opt)
                    λ = tuple(u)
                    u = tuple(np.log(np.array(u, dtype=float)))

                    val_cost, tr_cost, theta, grad = \
                    self._solve_problem(λ, loss_terms)
                    self.tr_costs[λ] = tr_cost
                    self.val_costs[λ] = val_cost
                    self.thetas[λ] = theta
                    if self.verbose:
                        print(f'Tr. cost: {tr_cost:.3e}, val. cost: {val_cost:.3e}')
                    return val_cost, grad[[e is None for e in lambda_val]]
                from scipy.optimize import minimize
                opt = minimize(min_func, np.zeros(opt_dimension),
                                method='L-BFGS-B', jac=True,
                               bounds=[[-10,10]]*opt_dimension,
                               options={'ftol':1E-5})


    def fit(self, data, train_frac = .8, seed = 0):
        """Fit cyclic baseline.

            - data is a pandas series, might have missing values
            - lambdas is the range of lambdas to use for each periodic
               regularization.
            - train_fraction: is the fraction of data to use for train
        """

        np.random.seed(seed)

        data = data[~data.isnull()]
        M = len(data)
        if self.verbose:
            print(f'Fitting on {M} observations.')

        # width of data
        self.l = data.shape[1] if len(data.shape) > 1 else 1

        if self.verbose:
            print(f'Selecting {int(100*train_frac)}-{int(100*(1-train_frac))}',
                'train and test sets.')

        mask = np.random.uniform(size=len(data)) < train_frac
        X_tr, y_tr = self.make_LS_cost(data[mask])
        X_val, y_val = self.make_LS_cost(data[~mask])
        loss_terms = (X_tr, y_tr, X_val, y_val)

        ## look at provided lambdas
        if self.verbose:
            print(self.lambdas)

        self.tr_costs = {}
        self.val_costs = {}
        self.thetas = {}

        self._theta_cache = np.zeros((self.N, self.l))  # for cg warmstart
        self._grad_cache = np.zeros((self.N, self.l))  # for cg warmstart

        self.grid_search_and_numerical_opt(loss_terms)

        self.best_lambda = min(self.val_costs, key=self.val_costs.get)
        self.theta = self.thetas[self.best_lambda]
        if self.verbose:
            print(f'Best λ = {self.best_lambda}')

    def predict(self, index):
        return pd.Series(index=index, data=self.theta[self._indexer(index)])
