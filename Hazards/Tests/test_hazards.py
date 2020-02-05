from numpy import asarray, allclose, arange

def test_logistic():
    from ..logistic import logistic
    x = asarray([1, 23, 3])
    assert allclose(logistic(x), asarray([0.73105858, 1., 0.95257413]))
    print('logistic Test PASSED')

def test_constant_h():
    from ..constant_h import constant_h

    v = asarray([0.06935799, 0.63527643, 0.58383591, 0.09942945,
                0.70186987])
    theta_h  = 0.10000000000000001
    hazard   = constant_h(theta_h)
    assert hazard.num_hazard_params == 1
    (Ht, dH) = hazard.evaluate(v)
    assert allclose(Ht, asarray([0.10000000000000001,
                    0.10000000000000001, 0.10000000000000001,
                    0.10000000000000001, 0.10000000000000001]))
    assert allclose(dH, asarray([1, 1, 1, 1, 1]))
    print('constant_h Test PASSED')


def test_logistic_h():
    from ..logistic_h import logistic_h
    v = asarray([0.44666091, 0.44823091, 0.38885706])
    theta_h = asarray([0.1, 1, 1])
    hazard = logistic_h(theta_h)
    assert hazard.num_hazard_params == 3

    (a, b) = hazard.evaluate(v)
    assert allclose(a, asarray([[0.0809484], [0.0809726], [0.08004097]]))
    assert allclose(b, asarray([[0.80948401, 0.00688839, 0.01542196], [0.80972602,  0.00690588,  0.01540698], [0.80040972,  0.00621215,  0.0159754 ]]))


def test_logistic_h2():
    from ..logistic_h2 import logistic_h2
    v = asarray([0.44666091, 0.44823091, 0.38885706])
    theta_h = asarray([0.1, 1, 1])
    hazard = logistic_h2(theta_h)
    assert hazard.num_hazard_params == 3
    (a, b) = hazard.evaluate(v)
    assert allclose(a, asarray([0.42496226, 0.42508931, 0.42019844]))
    assert allclose(b, asarray([[0.20186592, 0.03616261, 0.0809621],
                    [0.20192627, 0.03625446, 0.08088343], [0.19960301,
                    0.03261248, 0.08386753]]))
    print('logistic_h2 Test PASSED')


def test_logistic_logh():
    from ..logistic_logh import logistic_logh
    v = arange(1, 4)
    theta_h = asarray([1.5, 2.5, .5])
    hazard = logistic_logh(theta_h)
    assert hazard.num_hazard_params == 3
    (logH, logmH, dlogH, dlogmH) = hazard.evaluate(v)

    assert allclose(logH, asarray([[-0.25000063], [-0.20549172],
                    [-0.20174868]]))
    assert allclose(logmH, asarray([[-1.50868933], [-1.68333656],
                    [-1.69991147]]))
    assert allclose(dlogH, asarray([[0.18242552, 0.04742587,
                    0.04742587], [0.18242552, 0.00814028, 0.00407014],
                    [0.18242552, 0.00100605, 0.00033535]]))
    assert allclose(dlogmH, asarray([[-0.64228408, -0.16697709,
                    -0.16697709], [-0.79966016, -0.0356828,
                    -0.0178414], [-0.8160738, -0.00450053,
                    -0.00150018]]))
    print('logistic_logh Test PASSED')


def test_logistic_logg():
    from ..logistic_logg import logistic_logg
    v = 4
    theta_h = asarray([1.5, 2.5, .5])
    hazard = logistic_logg(theta_h)
    assert hazard.num_hazard_params == 3
    (logg, logmG, dlogg, dlogmG) = hazard.evaluate(v)

    assert allclose(logg, asarray([-0.25000063, -1.71418105,
                    -3.39377458, -5.09337818]))
    assert allclose(logmG, asarray([-1.50868933, -3.19202589,
                    -4.89193736, -6.59322724]))
    assert allclose(dlogg, asarray([[0.18242552, 0.04742587,
                    0.04742587], [-0.45985856, -0.15883682,
                    -0.16290695], [-1.25951872, -0.20165384,
                    -0.18448314], [-2.07559252, -0.20705028,
                    -0.18629113]]))
    assert allclose(dlogmG, asarray([[-0.64228408, -0.16697709,
                    -0.16697709], [-1.44194425, -0.20265989,
                    -0.18481849], [-2.25801804, -0.20716042,
                    -0.18631867], [-3.07546913, -0.20765398,
                    -0.18644206]]))
    print('logistic_logg Test PASSED')



