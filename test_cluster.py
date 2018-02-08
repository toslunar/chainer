#!/usr/bin/env python

from __future__ import print_function

import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions


# Network definition
class MLP(chainer.Chain):

    def __init__(self, n_units, n_out, f_lstm=False):
        super(MLP, self).__init__()
        self.f_lstm = f_lstm
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            if self.f_lstm:
                self.lstm = L.LSTM(None, n_units)
            self.l3 = L.Linear(None, n_out)  # n_units -> n_out

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        if self.f_lstm:
            h2 = self.lstm(h2)
        return self.l3(h2)




from chainer.computational_graph import *
import numpy as np

f_lstm = True
the_link = MLP(100, 10, f_lstm)

xs = []
ys = []
if f_lstm:
    for _ in range(3):
        x = chainer.Variable(np.zeros((1, 28**2), dtype=np.float32))
        y = the_link(x)
        xs.append(x)
        ys.append(y)
else:
    x = chainer.Variable(np.zeros((1, 28**2), dtype=np.float32))
    y = the_link(x)
    xs.append(x)
    ys.append(y)

g = build_computational_graph(ys)
set_clusters(g, [('model', the_link)])

print(g.dump())
