import numpy as np
import itertools
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import time

from .gradient_descent import BacktrackingGradientDescent



def build_cyclic_diff(N, diff):
    """Build cyclic diff matrices for regularization."""
    #  print(f'Building matrix of size {N} with diff {diff}.')
    return sp.eye(N) - sp.eye(N, k=diff) - sp.eye(N, k=diff - N)


def build_block_diag_diff(N, diff, period):
    num_blocks = N // period
    assert(N / period == N // period)
    # print(f'Building block diag. with {num_blocks} blocks.')
    return sp.block_diag([build_cyclic_diff(period, diff)] * num_blocks)


class Baseline(object):

    def __init__(self, *time_features):
        """Multi periodic smoother
        Args:
            - periods: objects that define n_periods and indexer
            #- indexers is sequence of functions, to apply to data.index,
            #        that return indexes (of each periodicity)
            #- n_periods is number of periods for each
        """
        self.indexers = [el.indexer for el in time_features]
        self.n_periods = [el.n_periods for el in time_features]
        self.K = len(time_features)
        self.verbose = True  # TODO fix this
        self.lambdas = [el.lambdas for el in time_features]
        self.cum_periods = np.concatenate([[1], np.cumprod(self.n_periods)])
        self.cum_periods = self.cum_periods.astype(int)
        self.N = int(np.product(self.n_periods))
        assert (self.N == self.cum_periods[-1])
        if self.verbose:
            print(f'Building baseline with {self.N} = ',
                  f'{"*".join(str(el) for el in self.n_periods)} values.')
        self.make_reg_mats()

    def make_reg_mats(self):
        """Build cyclic regularization matrices."""
        self.QTQ = []
        print('building reg matrices')
        for i in range(len(self.cum_periods) - 1):
            diff = self.cum_periods[i]
            period = self.cum_periods[i + 1]
            Q = build_block_diag_diff(self.N, diff, period)
            self.QTQ.append(Q.T@Q) # note this!

    def _indexer(self, index):
        return np.sum(
            indexer(index) * self.cum_periods[i]
            for (i, indexer) in enumerate(self.indexers))

    def make_LS_cost(self, data):
        """Build LS penalty part of the system.

        Args:"""
        NDATA = len(data)
        if self.verbose:
            print(f'Making quadratic loss term for {NDATA} obs.')
        P = sp.coo_matrix((
            np.ones(NDATA),
            (range(NDATA), self._indexer(data.index))),
            shape=(NDATA, self.N))
        x = data.values
        return P, x

    def _build_system(self, λ):
        # build matrix and right hand side
        mat = self.P_1.T @ self.P_1 / self.M_1
        for k in range(self.K):
            mat += self.QTQ[k] * λ[k]
        rhs = self.P_1.T @ np.matrix(self.x_1).T / self.M_1
        return mat, rhs

    def _compute_gradient(self, λ, theta, mat, val_res):

        if self.verbose:
            print('computing gradient of test cost in the lambdas')

        grad_base = np.zeros((self.N, self.L))
        grad = np.zeros((len(λ)))

        rhs = -(2 / self.M_2) * self.P_2.T @ val_res

        for j in range(self.L):
            self._cg_solve(matrix=mat, vector=rhs[:, j],
                           column=j, result=grad_base,
                           cache=self._grad_cache)

        for i in range(len(λ)):
            grad[i] = grad_base.T@self.QTQ[i]@theta
        print('grad of val. cost in the lambdas', grad)

        return grad

    def _cg_solve(self, matrix, vector, column, result, cache):
        #print([matrix[i, i] for i in range(5)])
        s = time.time()
        result[:, column], status = spl.cg(matrix, vector, x0=cache)
        print(f'CG took {time.time()-s} seconds.')
        if status != 0:
            raise Exception("CG failed.")
        cache[:, column] = result[:, column]

    def _compute_res_costs(self, theta):
        #val_pred = self.P_2@theta
        #val_res = val_pred - self.x_2.reshape(val_pred.shape)
        val_res = ((self.P_2@theta).T - self.x_2).T
        print(val_res.shape)
        #tr_pred = self.P_1@theta
        #tr_res = tr_pred - self.x_1.reshape(tr_pred.shape)
        tr_res = ((self.P_1@theta).T - self.x_1).T
        print(tr_res.shape)
        val_cost = np.linalg.norm(val_res)**2 / self.M_2
        tr_cost = np.linalg.norm(tr_res)**2 / self.M_1
        return val_res, val_cost, tr_res, tr_cost

    def _solve_problem(self, λ, compute_grad=True):

        print(f'solving with λ={λ}')

        theta = np.zeros((self.N, self.L))

        mat, rhs = self._build_system(λ)

        for j in range(self.L):

            self._cg_solve(matrix=mat, vector=rhs[:, j],
                           column=j, result=theta,
                           cache=self._theta_cache)

        val_res, val_cost, tr_res, tr_cost = self._compute_res_costs(theta)

        if self.verbose:
            print(f'Tr. cost: {tr_cost:.3e}, val. cost: {val_cost:.3e}')

        self.tr_costs[λ] = tr_cost
        self.val_costs[λ] = val_cost
        self.thetas[λ] = theta

        return val_cost, tr_cost, theta, \
            self._compute_gradient(λ, theta, mat, val_res) \
            if compute_grad else None

    def grid_search_and_numerical_opt(self, initial_lambda):

        for lambda_val in itertools.product(*self.lambdas):

            if self.verbose:
                print(f'working with lambda {lambda_val}')

            opt_dimension = sum([el is None for el in lambda_val])

            if opt_dimension == 0:
                λ = tuple(lambda_val)
                val_cost, tr_cost, theta, _ = \
                    self._solve_problem(λ, compute_grad=False)

            else:
                if self.verbose:

                    print(f'numerically optimizing on {opt_dimension} dims')

                def min_func(u_opt):

                    print(f'Numerical optimization step with u = {u_opt}')

                    lambdas_supplied = np.exp(u_opt)

                    λ = np.array(lambda_val)
                    λ[[e is None for e in lambda_val]] = lambdas_supplied
                    λ = tuple(λ)

                    val_cost, tr_cost, theta, grad = \
                        self._solve_problem(λ)

                    print(f'numerical step val_cost = {val_cost}')

                    # multiply grad so that it becomes 

                    grad_used = grad[[e is None for e in lambda_val]]

                    grad_used *= lambdas_supplied

                    print(f'numerical gradient of val cost in u: {grad_used}')

                    return val_cost, grad_used

                if initial_lambda is None:
                    lambda_0 = np.ones(opt_dimension)/self.N
                else:
                    lambda_0 = np.array(initial_lambda)
                max_lambda = 1E5 * lambda_0
                min_lambda = 1E-5 * lambda_0

                print('u_0', np.log(lambda_0))
                print('bounds', list(zip(np.log(min_lambda), np.log(max_lambda))))

                from scipy.optimize import minimize
                opt = minimize(
                    min_func,
                    np.log(lambda_0),
                    method='L-BFGS-B', jac=True,
                    bounds=list(zip(np.log(min_lambda), np.log(max_lambda))),
                    options={'ftol': 1E-6}
                )
                print(opt)

    def fit(self, data, train_frac=.8, seed=0, initial_lambda=None):
        """Fit cyclic baseline.

            - data is a pandas series, might have missing values
            - lambdas is the range of lambdas to use for each periodic
               regularization.
            - train_fraction: is the fraction of data to use for train
        """

        np.random.seed(seed)

        data = data[~data.isnull()]
        self.M = len(data)

        self.L = data.shape[1] if len(data.shape) > 1 else 1

        mask = np.random.uniform(size=len(data)) < train_frac

        self.M_1 = sum(mask)
        self.M_2 = sum(~mask)

        assert(self.M == self.M_1 + self.M_2)

        if self.verbose:
            print(f'Fitting on {self.M} observations, of dimension {self.L}')
            print(f'Train set: {self.M_1} obs. Test set : {self.M_2} obs.')

        self.P_1, self.x_1 = self.make_LS_cost(data[mask])
        self.P_2, self.x_2 = self.make_LS_cost(data[~mask])

        # look at provided lambdas
        if self.verbose:
            print('Provided lambdas:', self.lambdas)

        self.tr_costs = {}
        self.val_costs = {}
        self.thetas = {}

        self._theta_cache = np.zeros((self.N, self.L))  # for cg warmstart
        self._grad_cache = np.zeros((self.N, self.L))  # for cg warmstart

        self.grid_search_and_numerical_opt(initial_lambda=initial_lambda)

        self.best_lambda = min(self.val_costs, key=self.val_costs.get)
        if self.verbose:
            print(f'Best λ = {self.best_lambda}')
        self.theta = self.thetas[self.best_lambda]

    def predict(self, index):
        predicted = self.theta[self._indexer(index)]
        if self.L == 1:
            return pd.Series(data=predicted[:, 0], index=index)
        else:
            return pd.DataFrame(data=predicted, index=index)
