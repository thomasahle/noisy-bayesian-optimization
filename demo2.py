import nobo, skopt, random

def f(xs):
    x, y = xs
    return int(random.random() < abs(x) + abs(y))

bo = nobo.Optimizer([skopt.utils.Real(-1, 1), skopt.utils.Real(-1, 1)])

for j in range(50):
    x = bo.ask(verbose=False)
    y = f(x)
    print(x, y)
    bo.tell(x, y)

x, lo, y, hi = bo.get_best()
print(f'Best: {x}, f(x) = {y:.3} +/- {(hi-lo)/2:.3}')
bo.plot()
