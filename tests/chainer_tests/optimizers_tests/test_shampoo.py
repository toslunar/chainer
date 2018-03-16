import unittest

import numpy
import scipy

from chainer.optimizers import shampoo
from chainer import testing


class TestFractionalMatrixPower(unittest.TestCase):

    def setUp(self):
        xs = [numpy.random.randn(3, 3) for _ in range(2)]
        self.mat = sum(x.dot(x.T) for x in xs)  # positive semidefinite

    def test1(self):
        a = self.mat
        t = -0.25
        numpy.testing.assert_allclose(
            shampoo._fractional_matrix_power_h(a, t),
            scipy.linalg.fractional_matrix_power(a, t))


testing.run_module(__name__, __file__)
