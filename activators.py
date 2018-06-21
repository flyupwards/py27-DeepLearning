#!/usr/bin/env python
# -*- coding: UTF-8 -*-


import numpy as np

# Activator类实现了激活函数relu
class ReluActivator(object):
    # forward方法实现了前向计算
    def forward(self, weighted_input):
        #return weighted_input
        return max(0, weighted_input)

    # backward方法则是计算导数
    def backward(self, output):
        return 1 if output > 0 else 0


class IdentityActivator(object):
    def forward(self, weighted_input):
        return weighted_input

    def backward(self, output):
        return 1


# Sigmoid激活函数类
class SigmoidActivator(object):
    def forward(self, weighted_input):
        return 1.0 / (1.0 + np.exp(-weighted_input))

    def backward(self, output):
        return output * (1 - output)


class TanhActivator(object):
    def forward(self, weighted_input):
        return 2.0 / (1.0 + np.exp(-2 * weighted_input)) - 1.0

    def backward(self, output):
        return 1 - output * output