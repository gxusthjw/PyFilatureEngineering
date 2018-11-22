# --*-- coding:utf-8 --*--
"""
The model for bave with the composite function of
vertex quadratic function and logistic function.
"""
# for Python 2.7
from __future__ import division, print_function

import math

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


def quadratic_vertex_derivative(x, a, b, c):
    """The derivative for vertex form of quadratic function
    :param x: independent variable
    :param a: coefficient {a} of quadratic function
    :param b: the x coordinates of the vertex
    :param c: the y coordinates of the vertex
    :return:2 * a * (x - b)
    """
    return 2 * a * (x - b)


def quadratic_vertex_integrate(x, a, b, c):
    """The integrate for vertex form of quadratic function
    :param x: independent variable
    :param a: coefficient {a} of quadratic function
    :param b: the x coordinates of the vertex
    :param c: the y coordinates of the vertex
    :return:a * (b ** 2) * x + c * x - a * b * (x ** 2) + (a * (x ** 3)) / 3
    """
    return a * (b ** 2) * x + c * x - a * b * (x ** 2) + (a * (x ** 3)) / 3


def quadratic_vertex_derivative_a(x, a, b, c):
    """The partial derivative with respect to parameter {@code a}
    for vertex form of quadratic function
    :param x: independent variable
    :param a: coefficient {a} of quadratic function
    :param b: the x coordinates of the vertex
    :param c: the y coordinates of the vertex
    :return:(x - b) ** 2
    """
    return (x - b) ** 2


def quadratic_vertex_derivative_b(x, a, b, c):
    """The partial derivative with respect to parameter {@code b}
    for vertex form of quadratic function
    :param x: independent variable
    :param a: coefficient {a} of quadratic function
    :param b: the x coordinates of the vertex
    :param c: the y coordinates of the vertex
    :return:-2 * a * (x - b)
    """
    return -2 * a * (x - b)


def quadratic_vertex_derivative_c(x, a, b, c):
    """The partial derivative with respect to parameter {@code c}
    for vertex form of quadratic function
    :param x: independent variable
    :param a: coefficient {a} of quadratic function
    :param b: the x coordinates of the vertex
    :param c: the y coordinates of the vertex
    :return:1
    """
    return 1


class QuadraticVertex(object):
    """
    vertex form of quadratic function
    """

    def __init__(self, a, b, c):
        """
        :param a: coefficient {a} of quadratic function
        :param b: the x coordinates of the vertex
        :param c: the y coordinates of the vertex
        """
        if a == 0:
            raise Exception("Expected the parameter {a} not equal zero.")
        self.__a = a
        self.__b = b
        self.__c = c

    @property
    def parameter_a(self):
        """The property of parameter {@code a}
        :return: self.__a
        """
        return self.__a

    @property
    def parameter_b(self):
        """The property of parameter {@code b}
        :return: self.__a
        """
        return self.__b

    @property
    def parameter_c(self):
        """The property of parameter {@code c}
        :return: self.__c
        """
        return self.__c

    def value(self, x):
        """The function value for vertex form of quadratic function
        :param x: independent variable
        :return: self.__a * (x - self.__b) ** 2 + self.__c
        """
        return self.__a * (x - self.__b) ** 2 + self.__c

    def derivative(self, x):
        """The derivative value for vertex form of quadratic function
        :param x: independent variable
        :return: 2 * self.__a * (x - self.__b)
        """
        return 2 * self.__a * (x - self.__b)

    def derivative_a(self, x):
        """The partial derivative value with respect to parameter {@code a}
        :param x: independent variable
        :return: (x - self.__b) ** 2
        """
        return (x - self.__b) ** 2

    def derivative_b(self, x):
        """The partial derivative value with respect to parameter {@code b}
        :param x: independent variable
        :return: -2 * self.__a * (x - self.__b)
        """
        return -2 * self.__a * (x - self.__b)

    def derivative_c(self, x):
        """The partial derivative value with respect to parameter {@code c}
        :param x: independent variable
        :return: 1
        """
        return 1

    def integrate(self, x):
        """The integrate value for vertex form of quadratic function

        :param x: independent variable
        :return: self.__a * (self.__b ** 2) * x + self.__c * x - self.__a * self.__b * (x ** 2) + (self.__a * (x ** 3)) / 3
        """
        return self.__a * (self.__b ** 2) * x + self.__c * x - \
               self.__a * self.__b * (x ** 2) + (self.__a * (x ** 3)) / 3

    def x_intersection(self):
        """
        :return: the y coordinate for intersection with x
        """
        t = - self.__c / self.__a
        if t < 0:
            return ()
        elif t == 0:
            return self.__b
        else:
            return self.__b + math.sqrt(t), self.__b - math.sqrt(t)

    def y_intersection(self):
        """
        :return: the intersection with y
        """
        return self.__a * self.__b ** 2 + self.__c

    def vertex(self):
        """

        :return: the vertex coordinate
        """
        return self.__b, self.__c

    def is_invert(self):
        """
        :return: whether is invert
        """
        if self.__a > 0:
            return False
        elif self.__a < 0:
            return True
        else:
            raise Exception("Expected the parameter {a} not equal zero.")


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
