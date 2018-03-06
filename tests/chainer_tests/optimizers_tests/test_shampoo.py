import unittest

import numpy
import scipy

from chainer.optimizers import shampoo
from chainer import testing


class TestFractionalMatrixPower(unittest.TestCase):

    def test1(self):
        x = numpy.random.randn(3, 3)
        x += x.T  # test symmetric matrix
        a = x
        a = a.dot(a)
        a = a.dot(a)
        t = -0.25
        numpy.testing.assert_allclose(
            shampoo._fractional_matrix_power(a, t),
            scipy.linalg.fractional_matrix_power(a, t))


testing.run_module(__name__, __file__)
