from chainer import Variable
import chainer.functions as F
import numpy as np


def chk(shape_a, shape_b, **kwargs):
    a = Variable(np.ones(shape_a, dtype=np.float32))
    b = Variable(np.ones(shape_b, dtype=np.float32))
    return F.batch_matmul(a, b, **kwargs).shape
