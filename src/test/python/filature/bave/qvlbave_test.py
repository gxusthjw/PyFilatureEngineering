# --*-- coding:utf-8 --*--

from __future__ import division, print_function

import numpy as np
from filature.bave import qvlbave

x = np.array([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200])

a = 0.7e-5
b = 1200
c = 0
m = 1
k = 0.005
x0 = 200
print(qvlbave.quadratic_vertex(x, a, b, c))
print(qvlbave.logistic(x, m, k, x0))
print(qvlbave.qvl_function(x, a, b, c, m, k, x0))

