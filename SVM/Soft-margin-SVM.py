#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ソフトマージンSVM
"""

import numpy
from matplotlib import pyplot
import sys

def diff(x, y):
    return x - y

if __name__=='__main__':

    param = sys.argv

##### データの取得
    numpy.random.seed()
    N = 100
    #クラス数
    d = 2
    X = numpy.random.randn(N, d)
    #ラベリング
    T = numpy.array([1 if numpy.random.randn()+diff(x,y) > 0 else -1 for x, y in X])
    alpha = numpy.zeros(N)
    beta = 1.0
    eta_al = 0.0001
    eta_be = 0.1
    itr = 1000

    for _itr in range(itr):
        #alphaの更新アルゴリズム
        for i in range(N):
            delta = 1 - (T[i]*X[i]).dot(alpha*T*X.T).sum()
            alpha[i] += eta_al * delta
        #betaの更新アルゴリズム
        for i in range(N):
            beta += eta_be * alpha.dot(T) ** 2 / 2

    #最適なwとbを求める
    index = alpha > 0
    w = (alpha*T).T.dot(X)
    b = (T[index]-X[index].dot(w)).mean()

#### データ点と識別境界のプロット
    seq = numpy.arange(-3, 3, 0.02)
    pyplot.figure(figsize = (6, 6))
    pyplot.xlim(-3, 3)
    pyplot.ylim(-3, 3)
    pyplot.plot(seq, -(w[0] * seq + b) / w[1], 'k-')
    pyplot.plot(X[T ==  1,0], X[T ==  1,1], 'ro')
    pyplot.plot(X[T == -1,0], X[T == -1,1], 'bo')

    pyplot.savefig('./graph2.pdf')