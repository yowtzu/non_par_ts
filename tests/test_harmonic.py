import pandas as pd
import unittest
from npts.harmonic import *
from npts.features import MonthOfYear, DayOfWeek


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


class ModelFitTestCase(unittest.TestCase):

    def test_constant(self):
        harmonic = Harmonic()
        harmonic.fit(data,train_frac = 1.)
        self.assertTrue(harmonic.beta == 5.)

    def test_fit(self):
        harmonic = Harmonic([86400*31], max_harmonics=1)
        harmonic.fit(data, train_frac = 1.)
        print(harmonic.beta)

        self.assertTrue(np.allclose(
            harmonic.beta, 
     np.array([ 5.07159988,  1.69623568, -2.18921883])))

    # def test_cost_residuals(self):
    #     baseline = Baseline(MonthOfYear())
    #     baseline.fit(data)
    #     tr_res, tr_cost = \
    #         baseline._compute_res_costs(baseline.theta, baseline.P_1, baseline.x_1, baseline.M_1)
    #     val_res, val_cost = \
    #         baseline._compute_res_costs(baseline.theta, baseline.P_2, baseline.x_2, baseline.M_2)
    #     self.assertTrue(val_res.shape == (2,1))
    #     self.assertTrue(tr_res.shape == (7,1))
    #     self.assertTrue(np.alltrue(np.abs(tr_res) <= 1E-2))
    #     self.assertTrue(np.alltrue(val_res != 0))
    #     self.assertTrue(val_cost > 10)
    #     self.assertTrue(val_cost < 11)


if __name__ == '__main__':
    unittest.main()
