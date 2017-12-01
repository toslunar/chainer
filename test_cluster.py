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

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_out)  # n_units -> n_out

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)



from collections import defaultdict

from chainer.computational_graph import *
import numpy as np

the_link = MLP(100, 10)
x = chainer.Variable(np.zeros((1, 28**2), dtype=np.float32))
y = the_link(x)
g = build_computational_graph([y])
e_down = defaultdict(list)
e_up = defaultdict(list)
for s, t in g.edges:
    e_down[id(s)].append(id(t))
    e_up[id(t)].append(id(s))

descendants = {}
def _compute_d(v):
    # print(v)
    if v not in descendants:
        descendants[v] = {v}
        # print(descendants)
        for w in e_down[v]:
            _compute_d(w)
            # print(descendants)
            descendants[v] |= descendants[w]

anscestors = {}
def _compute_a(v):
    if v not in anscestors:
        anscestors[v] = {v}
        for w in e_up[v]:
            _compute_a(w)
            anscestors[v] |= anscestors[w]

idvarnodes = {id(n.get_variable()): id(n) for n in g.nodes if isinstance(n, variable.VariableNode)}

for v_obj in g.nodes:
    v = id(v_obj)
    _compute_d(v)
    _compute_a(v)

# print(descendants)
"""
for v, ws in descendants.items():
    print('{}: {}'.format(v, ws))

"""

# print({id(p) for p in the_link.params()})

clusters = []
for name, link in the_link.namedlinks():
    # params = list(link.params())
    # params = list(filter(lambda p: id(p) in descendants, link.params()))
    id_params = [idvarnodes[id(p)] for p in link.params()]
                 # if id(p) in idvarnodes]
    if id_params:
        # print('ok')
        # print([descendants[i] for i in id_params])
        common_descendants = set.intersection(*[descendants[i] for i in id_params])
        # print('{}: {}'.format(name, common_descendants))
        lcd = list(filter(
            lambda v: all(w not in common_descendants for w in e_up[v]),
            common_descendants))
        # print('{}: {}'.format(name, lcd))
        ans = set.union(*[anscestors[low] for low in lcd]).intersection(
            set.union(*[descendants[high] for high in id_params]))
        # print('{}: {}'.format(name, ans))
        for x in ans:
            assert x in map(id, g.nodes)
        clusters.append((name, {v for v in g.nodes if id(v) in ans}))

g.clusters = sorted(clusters, key=lambda x: len(x[1]))

print(g.dump())
# print(e)

# print(g.nodes)

