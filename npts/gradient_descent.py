import numpy as np
from numpy.linalg import norm


def BacktrackingGradientDescent(function, gradient, x0, beta=0.8,
        max_iters=100, precision=1E-5):

    # initial backtracking parameter
    x = np.array(x0)
    for i in range(max_iters):
        t = 1.
        print(f'Evaluating function and gradient at {x}')
        val, grad = function(x), gradient(x)
        if not (np.isscalar(grad) and np.isscalar(x)):
            assert(grad.shape == x.shape)
        if np.linalg.norm(grad) <= precision:
            print(f'Converged in {i} iterations!')
            return x

        while function(x + - t * grad) > val - (t / 2.) * norm(grad)**2:
            t *= beta

        x -= t * grad

    print(f'not converged after {max_iters} iteration, return last point')
    return x

def BackForwardtrackingGradientDescent(function, gradient, x0, beta=0.8, max_iters=100, precision=1E-5):

    # initial backtracking parameter
    t = 1.
    x = np.array(x0)

    for i in range(max_iters):
        print(f'Evaluating function and gradient at {x}')
        val, grad = function(x), gradient(x)
        if not (np.isscalar(grad) and np.isscalar(x)):
            assert(grad.shape == x.shape)
        if np.linalg.norm(grad) <= precision:
            print(f'Converged in {i} iterations!')
            return x

        def step_too_long(t):
            return function(x + - t * grad) > \
                val - (t / 2.) * norm(grad)**2

        def forwardtrack(t):
            for i in range(10):
                t /= beta
                print(f'forwardtracking to t={t:.2e}')
                if step_too_long(t):
                    t *= beta
                    return t
            return t

        def backtrack(t):
            for i in range(10):
                t *= beta
                print(f'backtracking to t={t:.2e}')
                if not step_too_long(t):
                    return t
            return t

        if step_too_long(t):
            t = backtrack(t)
        else:
            t = forwardtrack(t)

        x -= t * grad

    print(f'not converged after {max_iters} iteration, return last point')
    return x
