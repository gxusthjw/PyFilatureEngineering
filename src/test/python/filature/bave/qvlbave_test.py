# --*-- coding:utf-8 --*--

from __future__ import division, print_function

import numpy as np
from filature.bave import qvlbave

x = np.array([-0.2, 2, 0.3, 0.5, 0.6, 3, 4, 5, 6, 7, 8, 9])
print(qvlbave.quadratic_vertex(x, 1, 0, 0))
