import numpy

from chainer.backends import cuda
from chainer import optimizer


_default_hyperparam = optimizer.Hyperparameter()
_default_hyperparam.lr = 1.0  # [0.01, 10.0] in \S6.1, 1.0 in \S6.2
_default_hyperparam.alpha = 0.9
_default_hyperparam.eps = 1.0  # ?


def _fractional_matrix_power(A, t):
    """Compute the fractional power of a matrix."""

    xp = cuda.get_array_module(A)
    u, s, v = xp.linalg.svd(A)

    return xp.dot(u * (s ** t), v)


class ShampooRule(optimizer.UpdateRule):

    """Update rule of Shampoo.

    See :class:`~chainer.optimizers.Shampoo` for the default values of the
    hyperparameters.

    Args:
        parent_hyperparam (~chainer.optimizer.Hyperparameter): Hyperparameter
            that provides the default values.
        lr (float): Learning rate.
        alpha (float): Momentum.
        eps (float): Small value for the numerical stability.

    """

    def __init__(self, parent_hyperparam=None, lr=None, alpha=None, eps=None):
        super(ShampooRule, self).__init__(
            parent_hyperparam or _default_hyperparam)
        if lr is not None:
            self.hyperparam.lr = lr
        if alpha is not None:
            self.hyperparam.alpha = alpha
        if eps is not None:
            self.hyperparam.eps = eps

    def init_state(self, param):
        xp = cuda.get_array_module(param.data)
        eps = self.hyperparam.eps
        self.state['pow_update'] = 0
        with cuda.get_device_from_array(param.data):
            self.state['v'] = xp.zeros_like(param.data)
            for i, n in enumerate(param.shape):
                self.state['h%d'%i] = eps * xp.eye(n, dtype=param.dtype)

    def update_core(self, param):
        g = param.grad
        if g is None:
            return

        xp = cuda.get_array_module(param.data)

        lr = self.hyperparam.lr
        alpha = self.hyperparam.alpha
        pow_update = self.state['pow_update'] <= 0  # or self.t < 100

        k = param.ndim
        preconditioned_grad = g
        for i in range(k):
            for j in range(k):
                assert preconditioned_grad.shape[j] == g.shape[(i + j) % k]

            axis = tuple(j for j in range(k) if j != i)
            h_i = self.state['h%d'%i]
            h_i += xp.tensordot(
                g, g, axes=(axis, axis))
            if pow_update:
                self.state['pow_h%d'%i] = _fractional_matrix_power(
                    h_i, -0.5 / k)

            preconditioned_grad = xp.rollaxis(preconditioned_grad, 0, k).dot(
                self.state['pow_h%d'%i])

        if pow_update:
            self.state['pow_update'] = 20  # TODO(kataoka): hyperparam

        self.state['pow_update'] -= 1

        v = self.state['v']
        v += (1 - self.hyperparam.alpha) * (preconditioned_grad - v)
        param.data -= self.hyperparam.lr * v


class Shampoo(optimizer.GradientMethod):

    """Shampoo optimizer.

    See: http://jmlr.org/papers/v12/duchi11a.html

    Args:
        lr (float): Learning rate.
        alpha (float): Momentum.
        eps (float): Small value for the numerical stability.

    """

    def __init__(
            self,
            lr=_default_hyperparam.lr,
            alpha=_default_hyperparam.alpha,
            eps=_default_hyperparam.eps):
        super(Shampoo, self).__init__()
        self.hyperparam.lr = lr
        self.hyperparam.alpha = alpha
        self.hyperparam.eps = eps

    lr = optimizer.HyperparameterProxy('lr')
    alpha = optimizer.HyperparameterProxy('alpha')
    eps = optimizer.HyperparameterProxy('eps')

    def create_update_rule(self):
        return ShampooRule(self.hyperparam)
