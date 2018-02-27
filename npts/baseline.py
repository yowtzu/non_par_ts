import numpy as np
import itertools
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import time
import numba

from .gradient_descent import BacktrackingGradientDescent

##TODOS
# clear caches after fit

@numba.jit
def build_cyclic_diff(N, diff):
    """Build cyclic diff matrices for regularization."""
    #  print(f'Building matrix of size {N} with diff {diff}.')
    return sp.eye(N) - sp.eye(N, k=diff) - sp.eye(N, k=diff - N)


@numba.jit
def build_block_diag_diff(N, diff, period):
    num_blocks = N // period
    assert(N / period == N // period)
    # print(f'Building block diag. with {num_blocks} blocks.')
    return sp.block_diag([build_cyclic_diff(period, diff)] * num_blocks)


class Baseline(object):

    def __init__(self, *time_features, verbose=False):
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
        self.verbose = verbose
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
            self.QTQ.append(Q.T@Q)  # note this!

    def _indexer(self, index):
        """From pandas index to index of θ array."""
        result = np.zeros(len(index), dtype=int)
        for (i, indexer) in enumerate(self.indexers):
            result += indexer(index) * self.cum_periods[i]
        return result

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
        self.mat = self.P_1.T @ self.P_1 / self.M_1
        for k in range(self.K):
            self.mat += self.QTQ[k] * λ[k]
        if self.L == 1: ## TODO fix
            self.model_rhs = self.P_1.T @ np.matrix(self.x_1).T / self.M_1
        else:
            self.model_rhs = self.P_1.T @ np.matrix(self.x_1) / self.M_1
        #print('mat', self.mat[:5, :5].todense())
        #print('rhs', self.model_rhs[:5])

    def _compute_gradient(self, λ, θ):

        if self.verbose:
            print('computing gradient of test cost in the lambdas')

        grad_base = np.zeros((self.N, self.L))
        grad = np.zeros((len(λ)))

        rhs = -(2 / self.M_2) * self.P_2.T @ self.val_res

        for j in range(self.L):
            self._cg_solve(matrix=self.mat, vector=rhs[:, j],
                           column=j, result=grad_base,
                           cache=self._grad_cache)

        for i in range(len(λ)):
            if self.L == 1.:
                grad[i] = grad_base.T@self.QTQ[i]@θ
            else:
                grad[i] = grad_base@self.QTQ[i]@θ
        print('grad of val. cost in the lambdas', grad)

        return grad

    def _cg_solve(self, matrix, vector, column, result, cache):
        # print([matrix[i, i] for i in range(5)])
        s = time.time()
        result[:, column], status = spl.cg(matrix, vector, x0=cache[:, column])
        if self.verbose:
            print(f'CG took {time.time()-s} seconds.')
        if status != 0:
            raise Exception("CG failed.")
        cache[:, column] = result[:, column]

    def _compute_res_costs(self, theta, P, x, M):
        # val_pred = self.P_2@theta
        # val_res = val_pred - self.x_2.reshape(val_pred.shape)
        if M == 0:
            return np.zeros(x.shape), 0.
        if self.L == 1.:
            res = ((P@theta).T - x).T
        else:
            res = ((P@theta) - x)
        # print(res.shape)
        cost = np.linalg.norm(res)**2 / M
        return res, cost

    def _solve_problem(self, λ):

        if self.verbose:
            print(f'solving with λ={λ}')
        θ = np.zeros((self.N, self.L))
        self._build_system(λ)

        for j in range(self.L):
            self._cg_solve(matrix=self.mat, vector=self.model_rhs[:, j],
                           column=j, result=θ,
                           cache=self._theta_cache)

        self.val_res, self.val_costs[λ] = \
            self._compute_res_costs(θ, self.P_2, self.x_2, self.M_2)
        if self.compute_tr_costs:
            _, self.tr_costs[λ] = \
                self._compute_res_costs(θ, self.P_1, self.x_1, self.M_1)
        if self.verbose:
            print(f'Val. cost: {self.val_costs[λ]:.3e}')

        if self.best_lambda == λ:
            self.theta = θ

        #self.thetas[λ] = theta ##TODO drop thi

        return θ, self.val_costs[λ]

    def grid_search_and_numerical_opt(self):

        for λ_outer in itertools.product(*self.lambdas):

            if self.verbose:
                print(f'working with lambda {λ_outer}')

            opt_dimension = sum([el is None for el in λ_outer])
            if opt_dimension == 0:
                θ, _ = self._solve_problem(tuple(λ_outer))
            else:
                self._grad_descent_solve(λ_outer, opt_dimension)

    def _grad_descent_solve(self, λ_outer, opt_dimension):
        """u_opt = np.log(lambda * N), lambda = exp(u_opt)/N"""

        if self.verbose:
            print(f'numerically optimizing on {opt_dimension} dims')

        def u_opt_to_λ(u_opt):
            λ = np.array(λ_outer)
            λ[[e is None for e in λ_outer]] = np.exp(u_opt) / self.N
            return tuple(λ)

        def fun(u_opt):
            return self._solve_problem(u_opt_to_λ(u_opt))

        def grad(u_opt, θ):
            tmp = self._compute_gradient(u_opt_to_λ(u_opt), θ)
            res = tmp[[e is None for e in λ_outer]]
            res *= np.exp(u_opt)
            print(f'numerical gradient of val cost in u: {res}')
            return res

        def min_func(u_opt):
            θ, val_cost = fun(u_opt)
            return val_cost, grad(u_opt, θ)

        #     print(f'Numerical optimization step with u = {u_opt}')

        #     λ = np.array(λ_outer)
        #     λ[[e is None for e in λ_outer]] = np.exp(u_opt)
        #     λ = tuple(λ)
        #     val_cost = self._solve_problem(λ)
        #     print(f'numerical step val_cost = {val_cost}')
        #     grad = self._compute_gradient(λ)

        #     # multiply grad so that it becomes
        #     grad_used = grad[[e is None for e in λ_outer]]
        #     grad_used *= np.exp(u_opt)
        #     print(f'numerical gradient of val cost in u: {grad_used}')
        #     return val_cost, grad_used

        # if self.initial_lambda is None:
        #     lambda_0 = np.ones(opt_dimension) / self.N
        # else:
        #     lambda_0 = np.array(self.initial_lambda)
        # max_lambda = 1E5 * lambda_0
        # min_lambda = 1E-5 * lambda_0

        # print('u_0', np.log(lambda_0))
        # print('bounds', list(zip(np.log(min_lambda), np.log(max_lambda))))

        # BacktrackingGradientDescent(fun, grad, np.log(lambda_0), beta=0.8,
        #                             max_iters=100, precision=1E-3)

        from scipy.optimize import minimize
        opt = minimize(
            min_func,
            np.ones(opt_dimension),
            # np.log(lambda_0),
            method='L-BFGS-B', jac=True,
            bounds=[[-10, 10]] * opt_dimension,
            # bounds=list(zip(np.log(min_lambda), np.log(max_lambda))),
            options={'ftol': 1E-6,
                     'disp': True,
                     'maxls': 50}
        )
        print(opt)

    @property
    def best_lambda(self):
        return min(self.val_costs, key=self.val_costs.get)

    def _select_model(self):
        if self.verbose:
            print(f'Best λ = {self.best_lambda}')
        ## TODO post select

        #self.theta = self.thetas[self.best_lambda]
        #del self.thetas

    def _split_prepare_data(self, data, train_frac):
        data = data[~data.isnull()]

        self.M = len(data)
        self.L = data.shape[1] if len(data.shape) > 1 else 1

        mask = np.random.uniform(size=len(data)) <= train_frac

        self.M_1 = sum(mask)
        self.M_2 = sum(~mask)
        assert(self.M == self.M_1 + self.M_2)

        if self.verbose:
            print(f'Fitting on {self.M} observations, of dimension {self.L}')
            print(f'Train set: {self.M_1} obs. Test set : {self.M_2} obs.')

        self.P_1, self.x_1 = self.make_LS_cost(data[mask])
        self.P_2, self.x_2 = self.make_LS_cost(data[~mask])

    def fit(self, data, train_frac=.8, seed=0, initial_lambda=None,
            compute_tr_costs=False):
        """Fit cyclic baseline.

            - data is a pandas series, might have missing values
            - lambdas is the range of lambdas to use for each periodic
               regularization.
            - train_fraction: is the fraction of data to use for train
        """

        np.random.seed(seed)
        self.compute_tr_costs = compute_tr_costs
        self.initial_lambda = initial_lambda

        self._split_prepare_data(data, train_frac)

        # look at provided lambdas
        if self.verbose:
            print('Provided lambdas:', self.lambdas)

        if compute_tr_costs:
            self.tr_costs = {}
        self.val_costs = {}
        #self.thetas = {}

        self._theta_cache = np.zeros((self.N, self.L))  # for cg warmstart
        self._grad_cache = np.zeros((self.N, self.L))  # for cg warmstart

        self.grid_search_and_numerical_opt()
        self._select_model()
        self._clear_cache()

    def _clear_cache(self):
        del self._theta_cache
        del self._grad_cache

    def predict(self, index):
        predicted = self.theta[self._indexer(index)]
        if self.L == 1:
            return pd.Series(data=predicted[:, 0], index=index)
        else:
            return pd.DataFrame(data=predicted, index=index)
