# Nobo (or noisy-bayesian-optimization)
Nobo is a library for optimizing very noisy functions using [bayesian optimization](https://en.wikipedia.org/wiki/Bayesian_optimization).
In particular functions `p(x) -> [0,1]` for which you only observe a random variable `V_x ~ Bernoulli(p(x))`.
The following example shows how to use Nobo to find the minimum of `p(x) = x^2` over `[-1, 1]` in this situation:

```python
>>> import nobo, skopt, random
>>>
>>> def f(x):
>>>     return int(random.random() < x[0]**2)
>>>
>>> bo = nobo.Optimizer([skopt.utils.Real(-1, 1)])
>>> for j in range(50):
>>>     x = bo.ask(verbose=True)
>>>     bo.tell(x, f(x))
>>>
>>> x, lo, y, hi = bo.get_best()
>>> print(f'Best: {x}, f(x) = {y:.3} +/- {(hi-lo)/2:.3}')
>>> bo.plot()
...
Best: [0.05464264314195355], f(x) = 0.018 +/- 0.0213
```
![Screenshot](https://raw.githubusercontent.com/thomasahle/noisy-bayesian-optimization/master/static/demo.png)

