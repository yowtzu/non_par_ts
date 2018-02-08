import pandas as pd
import unittest
from npts.baseline import *
from npts.features import MonthOfYear, DayOfWeek

small_mat = np.matrix([[1., -1., 0.],
                       [0., 1., -1.],
                       [-1., 0., 1.]])

data = pd.Series(index=[pd.datetime(2000, 1, 1),
                        pd.datetime(2000, 2, 1),
                        pd.datetime(2000, 3, 1),
                        pd.datetime(2001, 4, 1),
                        pd.datetime(2002, 5, 1),
                        pd.datetime(2003, 6, 1),
                        pd.datetime(2001, 7, 1),
                        pd.datetime(2002, 8, 1),
                        pd.datetime(2003, 9, 1)],
                 data=[1, 2, 3, 4, 5, 6, 7, 8, 9])


class MatrixTestCase(unittest.TestCase):
    """Tests for matrix functions in `npts.py`."""

    def test_build_cyclic_diff(self):
        A = build_cyclic_diff(N=3, diff=1)
        self.assertTrue((A.todense() == small_mat).all())

    def test_build_block_diag_diff(self):
        A = build_block_diag_diff(N=6, diff=1, period=3)
        self.assertTrue((A.todense() == np.matrix(
            [[1., -1., 0., 0., 0., 0.],
             [0., 1., -1., 0., 0., 0.],
             [-1., 0., 1., 0., 0., 0.],
             [0., 0., 0., 1., -1., 0.],
             [0., 0., 0., 0., 1., -1.],
             [0., 0., 0., -1., 0., 1.]])).all())


class ModelTestCase(unittest.TestCase):

    def setUp(self):
        self.baseline = Baseline(DayOfWeek(), MonthOfYear())

    def test_setup(self):
        self.assertEqual(self.baseline.K, 2)
        self.assertEqual(self.baseline.lambdas, [[None], [None]])
        self.assertTrue((self.baseline.cum_periods == [1, 7, 84]).all())

        #  mats = self.baseline.Q
        #  self.assertTrue((mats[0].todense()[:2, :2] == small_mat[:2, :2]).all())

    def test_LS_cost(self):
        X, y = self.baseline.make_LS_cost(data)
        self.assertTrue((np.arange(1, 10) == y).all())


class ModelFitTestCase(unittest.TestCase):

    def test_fit(self):
        baseline = Baseline(MonthOfYear())
        baseline.fit(data)
        self.assertTrue(np.allclose(
            np.array(baseline.theta[2:5, 0]), [3, 4, 5]))

        print(baseline.theta)

    def test_cost_residuals(self):
        baseline = Baseline(MonthOfYear())
        baseline.fit(data)
        val_res, val_cost, tr_res, tr_cost = \
            baseline._compute_res_costs(baseline.theta)
        self.assertTrue(val_res.shape == (2,1))
        self.assertTrue(tr_res.shape == (7,1))
        self.assertTrue(np.alltrue(np.abs(tr_res) <= 1E-2))
        self.assertTrue(np.alltrue(val_res != 0))
        self.assertTrue(val_cost > 10)
        self.assertTrue(val_cost < 11)


if __name__ == '__main__':
    unittest.main()
