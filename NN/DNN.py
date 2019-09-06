#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DNN
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist

##### データの取得
#クラス数を定義(3以上で指定する)
m = 4

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_train = x_train.reshape([60000, 28*28])
x_train = x_train[y_train < m,:]

x_test = x_test.astype('float32') / 255.
x_test = x_test.reshape([10000, 28*28])
x_test = x_test[y_test < m,:]

y_train = y_train[y_train < m]
y_train = to_categorical(y_train, m)

y_test = y_test[y_test < m]
y_test = to_categorical(y_test, m)

n, d = x_train.shape
n_test, _ = x_test.shape

np.random.seed(123)

##### 活性化関数, 誤差関数, 順伝播, 逆伝播

##シグモイド関数
def sigmoid(x):
    f = 1/(1+np.exp(-x))
    df = (1-f)*f
    return f, df

##ReLU
def ReLU(x):
    f = np.maximum(0, x)
    df = np.where(x>0, 1, 0)
    return f, df

##ハイパボリックタンジェント
def tanh(x):
    f = np.tanh(x)
    df = 1/(np.cosh(x))**2
    return f, df

##ソフトマックス関数
def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum(ex)
    return ex/sum_ex

##クロスエントロピー
def CrossEntoropy(x, y):
    return -np.sum(y*np.log(x))

##順伝播
def forward(x, w, fnc):
    if fnc == softmax:
        return fnc(w.dot(x))
    else:
        z = fnc(np.dot(w, x))
        biasz = np.append(1, z[0])
        #print(biasz)
        print(z[1])
        return biasz, z[1]

#逆伝播
def backward(weight, delta, derivative):
    return np.dot(weight, delta)*derivative


##### 中間層のユニット数とパラメータの初期値
#任意に設定する
q0 = d
q1 = 100
q2 = 50
q3 = 30
q4 = m
w0 = np.random.normal(0, 0.3, size=(q1, q0+1))
w1 = np.random.normal(0, 0.3, size=(q2, q1+1))
w2 = np.random.normal(0, 0.3, size=(q3, q2+1))
w3 = np.random.normal(0, 0.3, size=(q4, q3+1))


########## 確率的勾配降下法によるパラメータ推定
num_epoch = 50

eta = 10**(-2)

e = []
e_test = []
error = []
error_test = []

for epoch in range(0, num_epoch):
    index = np.random.permutation(n)
    
    eta_t = eta/(epoch+1) 
    for i in index:
        xi = np.append(1, x_train[i, :])
        yi = y_train[i, :]

        ##### 順伝播
        z1,dz1 = forward(xi, w0, sigmoid)
        z2,dz2 = forward(z1, w1, ReLU)
        z3,dz3 = forward(z2, w2, tanh)
        z4     = forward(z3, w3, softmax)

        ##### 誤差評価
        e.append(CrossEntoropy(z4,yi))
        
        ##### 逆伝播
        ##第4層から第3層
        delta3 = z4 - yi
        derivative3 = dz3
        w3_del = np.delete(w3, 0, 1)
        bw3 = backward(w3_del.T, delta3, derivative3)
        ##第3層から第2層
        delta2 = bw3
        derivative2 = dz2
        w2_del = np.delete(w2, 0, 1)
        bw2 = backward(w2_del.T, delta2, derivative2)
        ##第2層から第1層
        delta1 = bw2
        derivative1 = dz1
        w1_del = np.delete(w1, 0, 1)
        bw1 = backward(w1_del.T, delta1, derivative1)

        ##### パラメータの更新
        w3 = w3 - eta_t*np.outer(delta3, z3)
        w2 = w2 - eta_t*np.outer(bw3, z2)
        w1 = w1 - eta_t*np.outer(bw2, z1)
        w0 = w0 - eta_t*np.outer(bw1, xi)

    ##### エポックごとの訓練誤差: eの平均をerrorにappendする
    error.append(sum(e)/n)
    e = []
    
    ##### test error
    for j in range(0, n_test):
        xi = np.append(1, x_test[j, :])
        yi = y_test[j, :]

        ##### テスト誤差: 誤差をe_testにappendする
        z1,dz1 = forward(xi, w0, sigmoid)
        z2,dz2 = forward(z1, w1, ReLU)
        z3,dz3 = forward(z2, w2, tanh)
        z4     = forward(z3, w3, softmax)
        e_test.append(CrossEntoropy(z4, yi))

    ##### エポックごとの訓練誤差: e_testの平均をerror_testにappendする
    error_test.append(sum(e_test)/n_test)
    e_test = []
    print(epoch)


########## 誤差関数のプロット
plt.clf()
plt.plot(error, label="training", lw=3)     #青線
plt.plot(error_test, label="test", lw=3)     #オレンジ線
plt.grid()
plt.legend(fontsize =16)
plt.savefig("./error.pdf", bbox_inches='tight', transparent=True)

########## 確率が高いクラスにデータを分類
prob = []
for j in range(0, n_test):    
    xi = np.append(1, x_test[j, :])
    yi = y_test[j, :]
    
    # テストデータに対する順伝播: 順伝播の結果をprobにappendする
    z1,dz1 = forward(xi, w0, sigmoid)
    z2,dz2 = forward(z1, w1, ReLU)
    z3,dz3 = forward(z2, w2, tanh)
    z4     = forward(z3, w3, softmax)
    prob.append(z4)
    
predict = np.argmax(prob, 1)

##### confusion matrixと誤分類結果のプロット
ConfMat = np.zeros((m, m))
for i in range(m):
    idx_true = (y_test[:, i]==1)
    for j in range(m):
        idx_predict = (predict==j)
        ConfMat[i, j] = sum(idx_true*idx_predict)
        if j != i:
            for l in np.where(idx_true*idx_predict == True)[0]:
                plt.clf()
                D = np.reshape(x_test[l, :], (28, 28))
                sns.heatmap(D, cbar =False, cmap="Blues", square=True)
                plt.axis("off")
                plt.title('{} to {}'.format(i, j))
                plt.savefig("./misslabeled{}.pdf".format(l), bbox_inches='tight', transparent=True)

plt.clf()
fig, ax = plt.subplots(figsize=(5,5),tight_layout=True)
fig.show()
sns.heatmap(ConfMat.astype(dtype = int), linewidths=1, annot = True, fmt="1", cbar =False, cmap="Blues")
ax.set_xlabel(xlabel="Predict", fontsize=18)
ax.set_ylabel(ylabel="True", fontsize=18)
plt.savefig("./confusion.pdf", bbox_inches="tight", transparent=True)