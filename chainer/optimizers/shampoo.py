import numpy

from chainer.backends import cuda
from chainer import optimizer


_default_hyperparam = optimizer.Hyperparameter()
_default_hyperparam.lr = 1.0  # [0.01, 10.0] in \S6.1, 1.0 in \S6.2
_default_hyperparam.alpha = 0.9
_default_hyperparam.eps = 1e-8  # ?


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
        # k = param.ndim
        self.state['pow_update'] = 0
        with cuda.get_device_from_array(param.data):
            for i, n in enumerate(param.shape):
            self.state['h%d'%i] = eps * xp.eye(n, dtype=param.dtype)
            # self.state['pow_h%d'%i] = (eps ** -(1 / 2 * k)) * xp.eye(n, dtype=param.dtype)

    def update_core(self, param):
        grad = param.grad
        if grad is None:
            return

        xp = cuda.get_array_module(param.data)

        lr = self.hyperparam.lr
        alpha = self.hyperparam.alpha

        k = param.ndim
        preconditioned_grad = grad
        for i in range(k):
            axis = tuple(j for j in range(k) if j != i)
            self.state['h%d'%i] += xp.tensordot(
                grad, grad, axes=(axis, axis))
            if self.state['pow_update'] <= 0:
                self.state['pow_h%d'%i] += _fractional_matrix_power(
                    self.state['h%d'%i], -0.5 / k)

            preconditioned_grad = xp.tensordot(
                preconditioned_grad, self.state['pow_h%d'%i],
                axes=((i,), (0,)))

        param.data -= self.hyperparam.lr * preconditioned_grad

    def update_core_cpu(self, param):
        grad = param.grad
        if grad is None:
            return

        lr = self.hyperparam.lr
        eps = self.hyperparam.eps
        h = self.state['h']

        h += grad * grad
        param.data -= lr * grad / (numpy.sqrt(h) + eps)

    def update_core_gpu(self, param):
        grad = param.grad
        if grad is None:
            return
        cuda.elementwise(
            'T grad, T lr, T eps',
            'T param, T h',
            '''h += grad * grad;
               param -= lr * grad / (sqrt(h) + eps);''',
            'adagrad')(grad, self.hyperparam.lr, self.hyperparam.eps,
                       param.data, self.state['h'])


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
