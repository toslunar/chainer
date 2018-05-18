import chainer
from chainer.backends import cuda
import cupy


class FooFunc(chainer.FunctionNode):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def __del__(self):
        self.bar = None
        print("{:10d} DEL: {}".format(cuda.memory_pool.used_bytes(), self.name))

    def forward(self, inputs):
        # Dummy array which can be used in backward()
        self.bar = cupy.ones((100, 100, 100, 10), cupy.float32)
        x, = inputs
        y = x * 2
        print("{:10d} FWD: {}".format(cuda.memory_pool.used_bytes(), self.name))
        return y,

    def backward(self, indexes, grad_outputs):
        gy, = grad_outputs
        print("{:10d} BWD: {}".format(cuda.memory_pool.used_bytes(), self.name))
        return gy * 2,


def func(x):
    hs = x,
    hs = FooFunc('a').apply(hs)
    hs = FooFunc('b').apply(hs)
    hs = FooFunc('c').apply(hs)
    hs = FooFunc('d').apply(hs)
    hs = FooFunc('e').apply(hs)
    h, = hs
    return h


def main():
    x = chainer.Variable(cupy.ones((2,)))
    y = func(x)
    print(x, y)
    y.grad = cupy.ones_like(y.data)
    y_grad_var = y.grad_var
    with chainer.variable.delay_backward():
        y.backward()
        del y
    print(x.grad_var, y_grad_var)
    print(chainer.grad([x.grad_var], [y_grad_var]))


main()
