import unittest
import nobo
import random
import skopt

# TODO: Fix random state

class TestOptimizer(unittest.TestCase):

    def _simple_opt(self, dims, fun, niter, maximize=False):
        bo = nobo.Optimizer(dims, maximize=maximize)
        for j in range(niter):
            x = bo.ask()
            bo.tell(x, fun(x))
        return bo.get_best()

    def test_deterministic(self):
        def f(x):
            return int(abs(x[0]) > .1)

        x, lo, y, hi = self._simple_opt([skopt.utils.Real(-1, 1)], f, 30)
        self.assertAlmostEqual(x[0], 0, 1)
        self.assertAlmostEqual(y, 0, 1)
        self.assertLess(y, hi)
        self.assertGreater(y, lo)

    def test_noisy(self):
        def f(x):
            return int(random.random() < x[0]**2)

        x, lo, y, hi = self._simple_opt([skopt.utils.Real(-1, 1)], f, 100)
        self.assertAlmostEqual(x[0], 0, 1)
        self.assertAlmostEqual(y, 0, 1)
        self.assertLess(y, hi)
        self.assertGreater(y, lo)

    def test_very_noisy(self):
        def f(x, noise=0.5):
            if random.random() > noise:
                return int(random.random() < x[0]**2)
            return int(random.random() < .5)

        x, lo, y, hi = self._simple_opt([skopt.utils.Real(-1, 1)], f, 100)
        self.assertAlmostEqual(x[0], 0, 1)
        self.assertAlmostEqual(y, .25, 1)
        self.assertLess(y, hi)
        self.assertGreater(y, lo)

    def test_2d_maximization(self):
        def f(x):
            return int(random.random() > (abs(x[0])+abs(x[1]))/2)

        x, lo, y, hi = self._simple_opt(
            [skopt.utils.Real(-1, 1), skopt.utils.Real(-1, 1)],
            f, 100, maximize=True)
        self.assertAlmostEqual(x[0], 0, 1)
        self.assertAlmostEqual(x[1], 0, 1)
        self.assertAlmostEqual(y, 0, 1)
        self.assertLess(y, hi)
        self.assertGreater(y, lo)
