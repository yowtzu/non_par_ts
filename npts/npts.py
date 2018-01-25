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
    # print(f'Building block diag. with {num_blocks} blocks.')
    return sp.block_diag([build_cyclic_diff(period, diff)]*num_blocks)


class NonParametricModel(object):
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
        self.Z = len(periods)
        self.verbose = False ## TODO fix this
        self.lambdas = [el.lambdas for el in periods]
        self.cum_periods = np.concatenate([[1],
            np.cumprod(self.periods)]).astype(int)
        self.N = int(np.product(self.periods))
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

    def _solve_problem_cv(self, λ, cross_val_costs, guess=None):

        #λ = np.exp(u)
        if self.verbose:
            print(f'solving with u={np.log(λ)}, so λ={λ}')

        K = len(cross_val_costs)
        val_costs = np.zeros(K)
        tr_costs = np.zeros(K)
        thetas = np.zeros((K,self.N))
        grad = np.zeros((K, len(λ)))

        for k in range(K):
            #print (f'solving fold {k}')
            X_tr, y_tr, X_val, y_val = cross_val_costs[k]

            ## TODO fix this
            X = [[X_tr]]
            X += [[np.sqrt(λ[i]/self.N)*X_reg]
              for i, X_reg in enumerate(self.reg_matrices)]
            X = sp.bmat(X)
            y = np.concatenate([y_tr, np.zeros(self.N*len(self.reg_matrices))])

            #s = time.time()
            theta, status = spl.cg(X.T @ X, X.T @ y,
                                   x0=self._cache[k][0])
            self._cache[k][0] = theta
            #print(f'theta CG took {time.time() - s} sec.')
            if status != 0:
                raise Exception("CG failed.")

            thetas[k] = theta
            val_res = X_val@theta - y_val
            val_costs[k] = val_res.T@val_res
            tr_res = X_tr@theta - y_tr
            tr_costs[k] = tr_res.T@tr_res

            #s = time.time()
            grad_base, status = spl.cg(X.T @ X, -2 * X_val.T @ val_res,
                                       x0=self._cache[k][1])
            self._cache[k][1] = grad_base
            #print(f'gradient CG took {time.time() - s} sec.')

            if status != 0:
                print("CG failed.")
                from scipy.sparse.linalg import splu
                fact = splu(X.T @ X),
                grad_base = fact.solve(-2 * X_val.T @ val_res)

            for i in range(len(λ)):
                ## TODO cache this
                X_i = self.reg_matrices[i]/np.sqrt(self.N)
                grad[k][i] = theta.T@X_i.T@X_i@grad_base * λ[i]

        return np.mean(val_costs), np.mean(tr_costs), \
               np.mean(thetas, 0), np.mean(grad,0)

    def fit(self, data, method = 'cross-val', folds = 5,
            train_frac = .8, seed = 0):
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

        self.K = folds

        cross_val_costs = []

        if method == 'cross-val':
            if self.verbose:
                print(f"Making {self.K}-fold cross. val. costs.")
            size = M//self.K
            for i in range(self.K):
                mask = np.zeros(M,dtype=bool)
                mask[size*i:(size*(i+1) if i < self.K-1 else None)] = True
                X_tr, y_tr = self.make_LS_cost(data[~mask])
                X_val, y_val = self.make_LS_cost(data[mask])
                cross_val_costs.append((X_tr, y_tr, X_val, y_val))

        elif method == 'train-test':
            if self.verbose:
                print(f'Selecting {int(100*train_frac)}-{int(100*(1-train_frac))}',
                    'train and test sets.')
            mask = np.random.uniform(size=len(data)) < train_frac
            X_tr, y_tr = self.make_LS_cost(data[mask])
            X_val, y_val = self.make_LS_cost(data[~mask])
            cross_val_costs.append((X_tr, y_tr, X_val, y_val))

        else:
            raise Exception("Method must be either 'cross-val' or 'train-test'")

        ## look at provided lambdas
        if self.verbose:
            print(self.lambdas)

        self.tr_costs = {}
        self.val_costs = {}
        self.thetas = {}

        self._cache = {k:np.zeros((2, self.N)) for k in range(self.K)}  # for cg startpoints
        for lambda_val in itertools.product(*self.lambdas):
            if self.verbose:
                print(f'working with lambda {lambda_val}')
            opt_dimension = sum([el is None for el in lambda_val])

            if opt_dimension == 0:
                λ = tuple(lambda_val)
                #u = tuple(np.log(lambda_val))
                val_cost, tr_cost, theta, grad = \
                    self._solve_problem_cv(λ, cross_val_costs)
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
                    #print('λ = %.2e, u = %.2e' % (λ,u))

                    val_cost, tr_cost, theta, grad = \
                    self._solve_problem_cv(λ, cross_val_costs)
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
                               options={'ftol':1E-5}) # safety
                # print(opt)
            # best lambdas

        self.best_lambda = min(self.val_costs, key=self.val_costs.get)
        self.theta = self.thetas[self.best_lambda]
        if self.verbose:
            print(f'Best λ = {self.best_lambda}')

            # rerun to update theta
            # val_cost, tr_cost, theta, grad = \
            #         self._solve_problem_cv(opt.x, cross_val_costs)


    def predict(self, index):
        return pd.Series(index=index, data=self.theta[self._indexer(index)])
