from numpy import allclose, asarray
from studentpdf import studentpdf

def test_studentpdf():
    x   = asarray([0.0608528,   0.1296728,  -0.2238741,   0.79862108])
    mu  = asarray([-0.85759774,  0.70178911, -0.29351646,  1.60215909])
    var = asarray([0.82608497,  0.75882319,  0.86101641,  0.73113357])
    nu  = asarray([0.71341641,  0.52532607,  0.20685246,  0.02304925])

    p = studentpdf(x, mu, var, nu, nargout=1)
    assert allclose(
        p, asarray([0.1521209,   0.1987373,   0.21214484,  0.01335992]))

    (p, dp) = studentpdf(x, mu, var, nu, nargout=2)
    assert allclose(
        p, asarray([0.1521209,   0.1987373,   0.21214484,  0.01335992]))
    assert allclose(
        dp, asarray([[1.67068098e-01,   8.00695192e-04,   9.07088043e-02],
                        [-2.38903047e-01,  -4.08902709e-02,   1.76043126e-01],
                        [9.74584714e-02,  -1.19253012e-01,   4.08675818e-01],
                        [-1.65769327e-02,  -2.71641034e-05,   5.45223728e-01]]))
    print('studentpdf Test PASSED')

if __name__ == '__main__':
    test_studentpdf()
