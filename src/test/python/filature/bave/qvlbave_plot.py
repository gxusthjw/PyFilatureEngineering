# --*-- coding:utf-8 --*--
# for Python 2.7
from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt
from filature.bave import qvlbave

plt.figure(figsize=(10.0, 8.0))
bave = qvlbave.QVLBave(1200, 2.8, 1.8 / 2.8, 300.0 / 1200.0)

pos = np.array([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200])

size = bave.value(pos)

print(bave.average(100, 200))
plt.plot(pos, size, "--")
plt.show()
