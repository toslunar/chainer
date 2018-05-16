import chainer
from chainer.backends import cuda
import numpy as cupy


class FooFunc(chainer.FunctionNode):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def __del__(self):
        self.bar = None
        print("{:10d} DEL: {}".format(1, self.name))

    def forward(self, inputs):
        # Dummy array which can be used in backward()
        self.bar = cupy.ones((100, 100, 100, 10), cupy.float32)
        x, = inputs
        y = x + 1
        print("{:10d} FWD: {}".format(1, self.name))
        return y,

    def backward(self, indexes, grad_outputs):
        gy, = grad_outputs
        print("{:10d} BWD: {}".format(1, self.name))
        return gy,


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
    y_grad = cupy.ones_like(y.data)
    y.grad = y_grad
    # cont = y.backward(execute=False)
    args = y.backward(execute=False)
    del y
    print('done: del y')
    """
    del args
    print('done: del args')
    """
    # print(cont)
    # cont()
    del y_grad
    chainer.function_node.backward_all(args)
    print('a')
    del args
    print('b')
    # fn.backward_all(args)
    # del fn
    # print('done: del fn')
    print(x.grad)


main()
