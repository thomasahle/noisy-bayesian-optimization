import os
from gpytorch.variational import UnwhitenedVariationalStrategy, VariationalStrategy
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.models import ApproximateGP
import torch
import gpytorch
import numpy as np
from scipy.stats import norm as normal
import skopt.utils

from botorch.models.gpytorch import GPyTorchModel
from botorch.acquisition import UpperConfidenceBound, qNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.fit import fit_gpytorch_model
#class _GPModel(ApproximateGP):
class _GPModel(ApproximateGP, GPyTorchModel):
    _num_outputs = 1

    def __init__(self, train_x, sigma=1):
        variational_distribution = CholeskyVariationalDistribution(
            train_x.size(0))
        variational_strategy = VariationalStrategy(
            self, train_x, variational_distribution, learn_inducing_locations=False)
        super(_GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(lengthscale_prior=gpytorch.priors.NormalPrior(0, sigma)))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(
            mean_x, covar_x)
        return latent_pred


class Optimizer:
    def __init__(self, dimensions, initial_points=10, maximize=False):
        self.space = skopt.utils.Space(dimensions)
        self.d = len(dimensions)
        self.initial_points = initial_points
        self.likelihood = gpytorch.likelihoods.BernoulliLikelihood()
        self.maximize = maximize
        self.xs = []
        self.ys = []

    def _train(self, iterations, lr, verbose=False):
        train_x = torch.cat(self.xs, dim=0).float()
        train_y = torch.tensor(self.ys, dtype=torch.float).float()
        model = _GPModel(train_x, sigma=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        mll = gpytorch.mlls.VariationalELBO(
            self.likelihood, model, train_y.numel())

        # Find optimal model hyperparameters
        model.train()
        self.likelihood.train()
        #losses = []
        for i in range(iterations):
            optimizer.zero_grad()
            try:
                output = model(train_x)
            except RuntimeError:
                if lr > 1e-6:
                    print('Error in training. Trying again with lower lr')
                    return self._train(iterations, lr/2, verbose)
                else:
                    print(f'Stopping after {i} iterations. Lr = {lr}')
                    break
            loss = -mll(output, train_y)
            loss.backward()
            #losses.append(loss.item())
            if verbose:
                print('Iter %d/%d - Loss: %.3f' %
                      (i+1, iterations, loss.item()))
            #if len(losses) >= iterations and np.argmin(losses)/len(losses) < 3/4:
                #break
            optimizer.step()

        model.eval()
        self.likelihood.eval()
        return model

    def _random_in_bounds(self, n=1):
        return self.space.transform(self.space.rvs(n))

    def ask2(self, n=1, iterations=10, lr=0.1, verbose=False):
        if n > 1:
            xs = list(self._random_in_bounds(n-1))
            xs.append(self.ask(1, iterations, lr, verbose))
            return xs
        # Use random points in the beginning
        if len(self.xs) < self.initial_points:
            return self._random_in_bounds()[0]
        # Train new model
        kappa = np.sqrt(2*np.log(len(self.xs)+1))
        model = self._train(iterations, lr, verbose=verbose)

        #UCB = UpperConfidenceBound(model, beta=kappa, maximize=self.maximize)
        UCB = UpperConfidenceBound(model, beta=.1, maximize=self.maximize)
        print(UCB)
        print(UCB.maximize)

        #test_x = torch.linspace(-1, 1, 50)
        #post = model.posterior(test_x)

        ucb_cands, ucb_vals = optimize_acqf(
            acq_function=UCB,
            bounds=torch.tensor(self.space.transformed_bounds).T.float(),
            q=1,
            num_restarts=1,
            raw_samples=25,
            )
        print(ucb_cands, ucb_vals)
        return ucb_cands

    def ask3(self, n=1, iterations=10, lr=0.1, verbose=False):
        # Use random points in the beginning
        if len(self.xs) < self.initial_points:
            return self._random_in_bounds()[0]

        train_x = torch.cat(self.xs, dim=0).float()
        #X_baseline = train_x.unsqueeze(-1)  # in botorch the feature dimension is assumed explicit
        X_baseline = train_x
        model = self._train(iterations, lr, verbose=verbose)

        qNEI = qNoisyExpectedImprovement(model, X_baseline=X_baseline, prune_baseline=True,
            #maximize=self.maximize
            )

        qnei_cands, qnei_vals = optimize_acqf(
                acq_function=qNEI,
                bounds=torch.tensor(self.space.transformed_bounds).T.float(),
                q=1,
                num_restarts=1,
                raw_samples=25,
            )

        print(qnei_cands, qnei_vals)
        return qnei_cands

    def ask(self, n=1, iterations=10, lr=0.1, verbose=False):
        # TODO: Find multiple points more intelligently.
        # Can also use yield to allow called to get started early.
        if n > 1:
            xs = list(self._random_in_bounds(n-1))
            xs.append(self.ask(1, iterations, lr, verbose))
            return xs
        # Use random points in the beginning
        if len(self.xs) < self.initial_points:
            return self._random_in_bounds()[0]
        # Train new model
        kappa = np.log(len(self.xs)+1)
        model = self._train(iterations, lr, verbose=verbose)
        with torch.no_grad():
            # TODO: We should do this instead using optimization, or using
            # the training loop (above) directly
            test_x = self._random_in_bounds(n=100)
            try:
                latent_pred = model(torch.tensor(test_x).float())
            except RuntimeError:
                print('Error in running model. Returning random point.')
                return self._random_in_bounds()[0]
            # Using Lower Confidnce Bound. Could be changed to something else
            # like expected improvement.
            xstar = test_x[np.argmin(
                latent_pred.mean - kappa*latent_pred.stddev)]
            return self.space.inverse_transform(xstar.reshape(1,-1))[0]

    def tell(self, x, y):
        #print(f'tell({x}, {y})')
        assert 0-1e-5 <= y <= 1+1e-5, f'y ({y}) not in range [0, 1]'
        x = self.space.transform([x])
        self.xs.append(torch.tensor(x).float())
        if self.maximize:
            y = 1-y
        self.ys.append(float(y))

    def get_best(self, iterations=50, lr=0.1, kappa=0, restarts=0):
        best = 1
        for j in range(restarts+1):
            model = self._train(iterations, lr)
            with torch.no_grad():
                # TODO: We should do this instead using optimization, or using
                # the training loop (above) directly
                test_x = self._random_in_bounds(n=500)
                try:
                    latent_pred = model(torch.tensor(test_x).float())
                except RuntimeError:
                    print('Error in running model. Returning 0')
                    return 0, 0, 0, 0
                # Using Lower Confidnce Bound. Could be changed to something else
                # like expected improvement.
                vals = latent_pred.mean - kappa*latent_pred.stddev
                i = np.argmin(vals)
                mean = normal.cdf(vals[i])
                if mean < best:
                    best = mean
                    x = self.space.inverse_transform(test_x[i].reshape(1,-1))[0]
                    lower = normal.cdf(latent_pred.mean[i] - latent_pred.stddev[i])
                    upper = normal.cdf(latent_pred.mean[i] + latent_pred.stddev[i])
        if self.maximize:
            return x, 1-upper, 1-mean, 1-lower
        return x, lower, mean, upper

    def size(self):
        return len(self.xs)

    def plot(self, iterations=50, lr=0.1):
        assert self.d <= 2

        model = self._train(iterations, lr)

        if self.d == 1:
            self._plot_1d(model)
        if self.d == 2:
            self._plot_2d(model)

    def _plot_1d(self, model):
        from matplotlib import pyplot as plt
        import matplotlib
        matplotlib.use('tkagg')

        with torch.no_grad():
            test_x = self._random_in_bounds(n=100)
            test_x = np.sort(test_x, axis=0)
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

    def _plot_2d(self, model):
        from matplotlib import pyplot as plt
        import matplotlib
        matplotlib.use('tkagg')

        with torch.no_grad():
            x = self._random_in_bounds(n=1000)
            latent_pred = model(torch.tensor(x).float())
            mean = normal.cdf(latent_pred.mean.numpy())

        fig, ax = plt.subplots()
        cs = ax.tricontourf(x[:,0], x[:,1], mean)
        x = torch.cat(self.xs, dim=0).numpy().reshape(-1,2)
        ax.plot(x[:,0], x[:,1], 'ko', ms=3)
        cbar = fig.colorbar(cs)
        plt.show()


import multiprocessing as mp
import concurrent, asyncio
class RPC(mp.Process):
    def __init__(self, cls, *args, **kwargs):
        super(RPC, self).__init__()
        self.obj = cls(*args, **kwargs)
        self.in_queue = mp.Queue()
        self.out_queue = mp.Queue()
        self.ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    def _submit(self, f, *args, **kwargs):
        return asyncio.wrap_future(self.ex.submit(f, *args, **kwargs))
    def __getattr__(self, attr):
        async def inner(*args, **kwargs):
            recv_conn, send_conn = mp.Pipe()
            await self._submit(self.in_queue.put, (attr, send_conn, args, kwargs))
            res = await self._submit(recv_conn.recv)
            recv_conn.close()
            send_conn.close()
            return res
        return inner
    def run(self):
        while True:
            attr, conn, args, kwargs = self.in_queue.get()
            method = getattr(self.obj, attr)
            try:
                result = method(*args, **kwargs)
            except BaseException as err:
                conn.send(err)
            else:
                conn.send(result)
    def close(self):
        raise NotImplemented

