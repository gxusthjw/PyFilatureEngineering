# --*-- coding:utf-8 --*--

"""
The model for bave with the composite function of
vertex quadratic function and logistic function.
"""
# for Python 2.7
from __future__ import division, print_function

import math

import numpy as np
from scipy import integrate


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


def logistic_exp(x, k, x0):
    return np.exp(-k * (x - x0))


def logistic_denominator(x, k, x0):
    return logistic_exp(x, k, x0) + 1


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


def logistic_derivative(x, m, k, x0):
    return (m * k * logistic_exp(x, k, x0)) / (logistic_denominator(x, k, x0) ** 2)


def logistic_integrate(x, m, k, x0):
    return m * (x + (np.log(logistic_denominator(x, k, x0)) / k))


def logistic_derivative_m(x, m, k, x0):
    return 1 / logistic_denominator(x, k, x0)


def logistic_derivative_k(x, m, k, x0):
    return (m * logistic_exp(x, k, x0) * (x - x0)) / (logistic_denominator(x, k, x0) ** 2)


def logistic_derivative_x0(x, m, k, x0):
    return -(m * k * logistic_exp(x, k, x0)) / (logistic_denominator(x, k, x0) ** 2)


class Logistic(object):
    def __init__(self, m=1, k=1, x0=0):
        self.__m = m
        self.__k = k
        self.__x0 = x0

    @property
    def parameter_m(self):
        return self.__m

    @property
    def parameter_k(self):
        return self.__k

    @property
    def parameter_x0(self):
        return self.__x0

    def logistic_exp(self, x):
        return np.exp(-self.__k * (x - self.__x0))

    def logistic_denominator(self, x):
        return self.logistic_exp(x) + 1.0

    def value(self, x):
        return self.__m / (self.logistic_denominator(x))

    def derivative(self, x):
        return self.__m * self.__k * self.logistic_exp(x) / (self.logistic_denominator(x) ** 2)

    def integrate(self, x):
        return self.__m * (x + (np.log(self.logistic_denominator(x))) / self.__k)

    def derivative_m(self, x):
        return 1.0 / self.logistic_denominator(x)

    def derivative_k(self, x):
        return (self.__m * self.logistic_exp(x) * (x - self.__x0)) / (self.logistic_denominator(x) ** 2)

    def derivative_x0(self, x):
        return -self.__m * self.__k * self.logistic_exp(x) / (self.logistic_denominator(x) ** 2)


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


def qvl_function_derivative(x, a, b, c, m, k, x0):
    return 2 * a * (x - b) * logistic(x, m, k, x0) + m * k * logistic_exp(x, k, x0) * quadratic_vertex(x, a, b, c) / (
            logistic_denominator(x) ** 2)


def qvl_function_derivative_a(x, a, b, c, m, k, x0):
    return (m * (x - b) ** 2) / logistic_denominator(x, k, x0)


def qvl_function_derivative_b(x, a, b, c, m, k, x0):
    return -2 * a * m * (x - b) / logistic_denominator(x, k, x0)


def qvl_function_derivative_c(x, a, b, c, m, k, x0):
    return logistic(x, m, k, x0)


def qvl_function_derivative_m(x, a, b, c, m, k, x0):
    return quadratic_vertex(x, a, b, c) / logistic_denominator(x, k, x0)


def qvl_function_derivative_k(x, a, b, c, m, k, x0):
    return logistic_exp(x, k, x0) * m * quadratic_vertex(x, a, b, c) * (-x + x0) / (logistic_denominator(x.k, x0) ** 2)


def qvl_function_derivative_x0(x, a, b, c, m, k, x0):
    return -k * m * logistic_exp(x, k, x0) * quadratic_vertex(x, a, b, c) / (logistic_denominator(x.k, x0) ** 2)


class QVLFunction(object):
    def __init__(self, a, b, c, m, k, x0):
        self.__a = a
        self.__b = b
        self.__c = c
        self.__m = m
        self.__k = k
        self.__x0 = x0

    @property
    def parameter_a(self):
        return self.__a

    @property
    def parameter_b(self):
        return self.__b

    @property
    def parameter_c(self):
        return self.__c

    @property
    def parameter_m(self):
        return self.__m

    @property
    def parameter_k(self):
        return self.__k

    @property
    def parameter_x0(self):
        return self.__x0

    def value(self, x):
        return qvl_function(x, self.__a, self.__b, self.__c, self.__m, self.__k, self.__x0)

    def derivative(self, x):
        return qvl_function_derivative(x, self.__a, self.__b, self.__c, self.__m, self.__k, self.__x0)

    def derivative_a(self, x):
        return qvl_function_derivative_a(x, self.__a, self.__b, self.__c, self.__m, self.__k, self.__x0)

    def derivative_b(self, x):
        return qvl_function_derivative_b(x, self.__a, self.__b, self.__c, self.__m, self.__k, self.__x0)

    def derivative_c(self, x):
        return qvl_function_derivative_c(x, self.__a, self.__b, self.__c, self.__m, self.__k, self.__x0)

    def derivative_m(self, x):
        return qvl_function_derivative_m(x, self.__a, self.__b, self.__c, self.__m, self.__k, self.__x0)

    def derivative_k(self, x):
        return qvl_function_derivative_k(x, self.__a, self.__b, self.__c, self.__m, self.__k, self.__x0)

    def derivative_x0(self, x):
        return qvl_function_derivative_x0(x, self.__a, self.__b, self.__c, self.__m, self.__k, self.__x0)


class QVLBave(object):
    """
    The Composite function model (vertex form of quadratic function and logistic function) of bave
    """

    def __init__(self, qvlbave_length, initial_size, max_size_pos, terminal_size):
        """
        :param qvlbave_length: the length of bave
        :param initial_size: the initial size of bave
        :param max_size_pos:the max size pos on bave
        :param terminal_size:the terminal size of bave
        """
        self.__qvlbave_length = qvlbave_length
        self.__initial_size = initial_size
        self.__max_size_pos = max_size_pos
        self.__terminal_size = terminal_size

    @property
    def qvlbave_length(self):
        """
        :return:the length of bave
        """
        return self.__qvlbave_length

    @property
    def initial_size(self):
        """
        :return:the initial size of bave
        """
        return self.__initial_size

    @property
    def max_size_pos(self):
        """
        :return: the max size pos on bave
        """
        return self.__max_size_pos

    @property
    def terminal_size(self):
        """
        :return: the terminal size of bave
        """
        return self.__terminal_size

    @property
    def tan(self):
        """
        :return: The ratio of terminal size and length of bave
        """
        return self.terminal_size / self.qvlbave_length

    @property
    def tan1(self):
        """
        :return:(self.initial_size - self.terminal_size) / self.qvlbave_length
        """
        return (self.initial_size - self.terminal_size) / self.qvlbave_length

    @property
    def alpha(self):
        """
        :return:the ratio of terminal_size and initial_size
        """
        return self.self.__terminal_size / self.__initial_size

    @property
    def beta(self):
        """
        :return:the ratio of max_size_pos and length of bave
        """
        return self.self.__max_size_pos / self.qvlbave_length

    @property
    def quadratic_vertex_a(self):
        """
        :return:The parameter {a} of quadratic function (vertex form)
        """
        return self.initial_size * (1 - self.alhpa) * (
                1 + np.exp(self.logistic_k * self.beta * self.qvlbave_length)) / (
                       self.qvlbave_length ** 2)

    @property
    def quadratic_vertex_b(self):
        """
        :return:The parameter {b} of quadratic function (vertex form)
        """
        return self.qvlbave_length

    @property
    def quadratic_vertex_c(self):
        """
        :return: The parameter {b} of quadratic function (vertex form)
        """
        return 0

    @property
    def logistic_m(self):
        """
        :return:The parameter {m} of logistic function
        """
        return 1

    @property
    def logistic_k(self):
        """
        :return:The parameter {k} of logistic function
        """
        return 4 / (self.qvlbave_length * (1 - self.beta))

    @property
    def logistic_x0(self):
        """
        :return:The parameter {x0} of logistic function
        """
        return self.qvlbave_length * self.beta

    @property
    def qvl_d(self):
        """
        :return: The parameter {d} of qvl model (Quadratic Vertex and logistic)
        """
        return self.alhpa * self.initial_size

    @property
    def max_size(self):
        """
        :return: the max size of bave
        """
        return (self.quadratic_vertex_a * ((self.max_size_pos - self.quadratic_vertex_b) ** 2)) / 2 + self.qvl_d

    @property
    def quadratic_vertex(self):
        """
        :return: The vertex form of quadratic function
        """
        return QuadraticVertex(self.quadratic_vertex_a, self.quadratic_vertex_b, self.quadratic_vertex_c)

    @property
    def tan2(self):
        return self.max_size_pos / self.qvlbave_length

    @property
    def tan3(self):
        """
        :return: (self.max_size - self.initial_size) / self.max_size_pos
        """
        return (self.max_size - self.initial_size) / self.max_size_pos

    @property
    def logistic(self):
        """
        :return: The logistic function
        """
        return Logistic(self.logistic_m, self.logistic_k, self.logistic_x0)

    def value(self, x):
        """
        :param x: independent variable
        :return: The value of composite function
        """
        return self.quadratic_vertex.value(x) * self.logistic.value(x) + self.qvl_d

    def bave_size(self, pos):
        if 0 <= pos <= self.qvlbave_length:
            return self.value(pos)
        else:
            raise Exception("Expected the parameter {0 <= pos <= %f},but got {pos = %f}" % (self.__qvlbave_length, pos))

    def integrate_romberg(self, *args):
        if len(args) == 0:
            lower = 0
            upper = self.qvlbave_length
        elif len(args) == 1:
            lower = args[0]
            upper = self.qvlbave_length
        else:
            lower = args[0]
            upper = args[1]

        return integrate.romberg(self.value, lower, upper)

    def average(self, *args):
        if len(args) == 0:
            lower = 0
            upper = self.qvlbave_length
        elif len(args) == 1:
            lower = args[0]
            upper = self.qvlbave_length
        else:
            lower = args[0]
            upper = args[1]
        return self.integrate_romberg(lower, upper) / (upper - lower)

    def var_function(self, x, *args):
        if len(args) == 0:
            lower = 0
            upper = self.qvlbave_length
        elif len(args) == 1:
            lower = args[0]
            upper = self.qvlbave_length
        else:
            lower = args[0]
            upper = args[1]
        average = self.average(lower, upper)
        if x < lower or x > upper:
            raise Exception("Expected the parameter {%f <= x <= %f},but got {x = %f}" % (lower, upper, x))
        return (self.value(x) - average) ** 2

    def var(self, *args):
        if len(args) == 0:
            lower = 0
            upper = self.qvlbave_length
        elif len(args) == 1:
            lower = args[0]
            upper = self.qvlbave_length
        else:
            lower = args[0]
            upper = args[1]
        return integrate.romberg(self.var_function, lower, upper) / (upper - lower)

    def std(self, *args):
        return math.sqrt(self.var(*args))


class QVLBave2(QVLBave):
    def __init__(self, qvlbave_length, tan, tan1, tan2):
        terminal_size = tan * qvlbave_length
        initial_size = tan1 * qvlbave_length + tan * qvlbave_length
        max_size_pos = tan2 * qvlbave_length
        super(QVLBave2, self).__init__(qvlbave_length, initial_size, max_size_pos, terminal_size)
