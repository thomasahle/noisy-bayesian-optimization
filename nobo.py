import os
from gpytorch.variational import UnwhitenedVariationalStrategy, VariationalStrategy
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.models import ApproximateGP
import torch
import gpytorch
import numpy as np
from scipy.stats import norm as normal


class GPModel(ApproximateGP):
    def __init__(self, train_x, sigma=1):
        variational_distribution = CholeskyVariationalDistribution(
            train_x.size(0))
        variational_strategy = VariationalStrategy(
            self, train_x, variational_distribution, learn_inducing_locations=False)
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(lengthscale_prior=gpytorch.priors.NormalPrior(0, sigma)))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(
            mean_x, covar_x)
        return latent_pred


class NoisyBayesianOptimization:
    def __init__(self, bounds, initial_points=10):
        self.bounds = np.array(bounds)
        self.d = len(bounds)
        self.initial_points = initial_points
        self.likelihood = gpytorch.likelihoods.BernoulliLikelihood()
        self.xs = []
        self.ys = []

    def _train(self, iterations, lr, verbose=False):
        train_x = torch.cat(self.xs, dim=0).float()
        train_y = torch.tensor(self.ys, dtype=torch.float).float()
        model = GPModel(train_x, sigma=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        mll = gpytorch.mlls.VariationalELBO(
            self.likelihood, model, train_y.numel())

        # Find optimal model hyperparameters
        model.train()
        self.likelihood.train()
        for i in range(iterations):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            if verbose:
                print('Iter %d/%d - Loss: %.3f' %
                      (i + 1, iterations, loss.item()))
            optimizer.step()

        model.eval()
        self.likelihood.eval()
        return model

    def _random_in_bounds(self, n=1):
        return np.random.rand(n, self.d) * \
            (self.bounds[:, 1] - self.bounds[:, 0]) + self.bounds[:, 0]

    def ask(self, iterations=50, lr=0.1, verbose=False):
        if len(self.xs) < self.initial_points:
            return self._random_in_bounds()[0]
        kappa = np.log(len(self.xs)+1)
        model = self._train(iterations, lr, verbose=verbose)
        with torch.no_grad():
            # TODO: We should do this instead using optimization, or using
            # the training loop (above) directly
            test_x = self._random_in_bounds(n=100)
            latent_pred = model(torch.tensor(test_x).float())
            # Using Lower Confidnce Bound. Could be changed to something else
            # like expected improvement.
            xstar = test_x[np.argmin(
                latent_pred.mean - kappa*latent_pred.stddev)]
            return xstar

    def tell(self, x, y):
        self.xs.append(torch.tensor(x))
        self.ys.append(float(y))

    def get_best(self, iterations=500, lr=0.1, kappa=0):
        model = self._train(iterations, lr)
        with torch.no_grad():
            # TODO: We should do this instead using optimization, or using
            # the training loop (above) directly
            test_x = self._random_in_bounds(n=100)
            latent_pred = model(torch.tensor(test_x).float())
            # Using Lower Confidnce Bound. Could be changed to something else
            # like expected improvement.
            vals = latent_pred.mean - kappa*latent_pred.stddev
            i = np.argmin(vals)
            mean = normal.cdf(vals[i])
            lower = normal.cdf(latent_pred.mean[i] - latent_pred.stddev[i])
            upper = normal.cdf(latent_pred.mean[i] + latent_pred.stddev[i])
            return test_x[i], lower, mean, upper

    def plot(self, iterations=500, lr=0.1):
        assert self.d == 1

        from matplotlib import pyplot as plt
        import matplotlib
        matplotlib.use('tkagg')

        model = self._train(iterations, lr)
        with torch.no_grad():
            test_x = np.linspace(*self.bounds[0], 100)
            latent_pred = model(torch.tensor(test_x).float())

        kappa = np.log(len(self.xs)+1)
        _fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        ax.plot(self.xs, self.ys, 'k*', label='Observed Data')
        test_x = test_x.reshape(-1)
        mean = normal.cdf(latent_pred.mean.numpy())
        lower = normal.cdf(
            (latent_pred.mean - kappa*latent_pred.stddev).numpy())
        upper = normal.cdf(
            (latent_pred.mean + kappa*latent_pred.stddev).numpy())
        ax.plot(test_x, mean, label='Mean')
        ax.fill_between(test_x, upper, lower, color="#b9cfe7", edgecolor="")
        ax.set_ylim([-1, 2])
        ax.legend()
        plt.show()


smoke_test = ('CI' in os.environ)
training_inner = 2 if smoke_test else 10
training_outer = 2 if smoke_test else 100
report_iter = 30


def f(x, noise=0.5):
    import random
    if random.random() > noise:
        return int(random.random() < x**2)
    return int(random.random() < .5)


bo = NoisyBayesianOptimization([(-1, 1)])
for j in range(training_outer):
    x = bo.ask(verbose=True, iterations=10)
    bo.tell(x, f(x))
    if (j+1) % report_iter == 0:
        x, lo, y, hi = bo.get_best(iterations=50)
        print(f'Best: {x}, f(x) = {y:.3} +/- {(hi-lo)/2:.3}')
        bo.plot(iterations=50)

