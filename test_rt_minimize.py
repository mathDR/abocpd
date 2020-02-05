from numpy import asarray, allclose
from numpy.random import normal
from rt_minimize import rt_minimize

def test_rt_minimize():
    def func(x, theta1, theta2):
        f = ((x - theta1 - theta2) ** 2).sum()
        df = 2 * (x - theta1 - theta2)
        return (f, df)

    X = normal(0, 1.0, (3, ))
    x0 = X[:]
    theta1 = asarray([2.3, 1.1, -4.4])
    theta2 = asarray([1., 2., 3.])

    (x, fx, i) = rt_minimize(X, func, -100, theta1, theta2)
    assert allclose(x,theta1+theta2)
    assert allclose(fx[-1], 0.0)
    print('TEST rt_minimize PASSED')

if __name__ == '__main__': 
   test_rt_minimize()
