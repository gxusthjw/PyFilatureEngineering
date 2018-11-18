# --*-- coding:utf-8 --*--
"""

"""

from __future__ import division, print_function

import numpy as np


def quadratic_vertex(x, a, b, c):
    """The vertex form of quadratic function
    :reference:https://en.wikipedia.org/wiki/Quadratic_function
    :param x: independent variable
    :param a: coefficient {a} of quadratic function
    :param b: the x coordinates of the vertex
    :param c: the y coordinates of the vertex
    :return: a * (x - b) ** 2 + c
    """
    return a * (x - b) ** 2 + c


def logistic(x, m, k, x0):
    """The logistic function
    :reference:https://en.wikipedia.org/wiki/Logistic_function
    :param x:independent variable
    :param m:the maximum value of curve (or function)
    :param k:the logistic growth rate or steepness of the curve
    :param x0:the x-value of the sigmoid's midpoint
    :return: m / (1 + np.exp(-k * (x - x0)))
    """
    return m / (1 + np.exp(-k * (x - x0)))


def qvl_function(x, a, b, c, m, k, x0):
    """The composite function of vertex form of quadratic function and logistic function
    :param x:independent variable
    :param a: coefficient {a} of quadratic function
    :param b: the x coordinates of the vertex
    :param c: the y coordinates of the vertex
    :param m:the maximum value of curve (or function)
    :param k:the logistic growth rate or steepness of the curve
    :param x0:the x-value of the sigmoid's midpoint
    :return: (m * (a * (x - b) ** 2 + c)) / (1 + np.exp(-k * (x - x0)))
    """
    return quadratic_vertex(x, a, b, c) * logistic(x, m, k, x0)


class QVLBave(object):
    """The Composite function model (vertex form of quadratic function and logistic function) of bave

    """

    def __init__(self, qvlbave_length, alpha, beta, gamma):
        """

        :param qvlbave_length:
        :param alpha:
        :param beta:
        :param gamma:
        """
        self.__qvlbave_length = qvlbave_length;
        self.__alhpa = alpha
        self.__beta = beta
        self._gamma = gamma

    @property
    def qvlbave_length(self):
        """

        :return:
        """
        return self.__qvlbave_length

    @property
    def alpha(self):
        """

        :return:
        """
        return self.__alpha

    @property
    def beta(self):
        """

        :return:
        """
        return self.__beta

    @property
    def gamma(self):
        """

        :return:
        """
        return self.__gamma

    @property
    def quadratic_vertex_a(self):
        """

        :return:
        """
        pass

    @property
    def quadratic_vertex_b(self):
        """

        :return:
        """
        pass

    @property
    def quadratic_vertex_c(self):
        pass

    @property
    def logistic_m(self):
        """

        :return:
        """
        pass

    @property
    def logistic_k(self):
        """

        :return:
        """
        pass

    @property
    def logistic_x0(self):
        """

        :return:
        """
        pass
