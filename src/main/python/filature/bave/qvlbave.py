# --*-- coding:utf-8 --*--

from __future__ import division, print_function

import numpy as np


def quadratic_vertex(x, a, b, c):
    return a * (x - b) ** 2 + c


def logistic(x, m, k, x0):
    return m / (1 + np.exp(-k * (x - x0)))


def qvl_function(x, a, b, c, m, k, x0):
    return quadratic_vertex(x, a, b, c) * logistic(x, m, k, x0)


class QVLBave(object):
    pass
