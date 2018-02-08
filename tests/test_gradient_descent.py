import unittest

from npts.gradient_descent import BacktrackingGradientDescent
import numpy as np


class MatrixTestCase(unittest.TestCase):
    """Tests for matrix functions in `npts.py`."""

    def test_simple(self):
        fun = lambda x: x**2
        grad = lambda x: 2 * x
        res = BacktrackingGradientDescent(fun, grad, 2., precision=1E-8)
        self.assertTrue(np.isclose(res, 0.))

    def test_2d(self):
        fun = lambda x: sum(x**2)
        grad = lambda x: 2 * x
        res = BacktrackingGradientDescent(fun, grad, np.array([1., 2.]), precision=1E-8)
        self.assertTrue(np.allclose(res, [0., 0.]))

    def test_2d_tilt(self):
        """Test with 2d nonsym."""
        fun = lambda x: x[0]**2 + 200*x[1]**2
        grad = lambda x: 2 * x * [1., 200.]
        res = BacktrackingGradientDescent(fun, grad, np.array([1., 2.]), precision=1E-8)
        print (res)
        self.assertTrue(np.allclose(res, [0., 0.], atol=1E-4))


if __name__ == '__main__':
    unittest.main()
