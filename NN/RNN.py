#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RNN
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist

##### データの取得
#クラス数を定義
m = 3

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_train = x_train[y_train < m, :, :]

x_test = x_test.astype('float32') / 255.
x_test = x_test[y_test < m, :, :]

y_train = y_train[y_train < m]
y_train = to_categorical(y_train, m)

y_test = y_test[y_test < m]
y_test = to_categorical(y_test, m)

n, d , _ = x_train.shape
n_test, _, _ = x_test.shape

np.random.seed(123)

##### 活性化関数, 誤差関数, 順伝播, 逆伝播
###活性化関数
##シグモイド関数
def sigmoid(x):
    #print(x)
    f = 1/(1+np.exp(-x))
    df = (1-f)*f
    return f, df

##ReLU
def ReLU(x):
    #print(x)
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
    #print(x)
    c = np.max(x)
    ex = np.exp(x-c)
    sum_ex = np.sum(ex)
    return ex/sum_ex

###誤差関数
##クロスエントロピー
def CrossEntoropy(x, y):
    delta = 1e-7
    return -np.sum(y*np.log(x+delta))

###順伝播
def forward(z_in, z_prev, w_in, w_hidden, fncs):
    tmp_in = np.dot(w_in, z_in)
    tmp_hidden = np.dot(w_hidden, z_prev)
    tmp = tmp_in + tmp_hidden
    z = fncs(tmp)[0]
    u = fncs(tmp)[1]
    return z, u

###逆伝播
def backward(w_hidden, w_out, delta_t, delta_out, derivative):
    w_out_tilde = np.delete(w_out, 0, 1)
    return (np.dot(w_hidden.T,delta_t)+np.dot(w_out_tilde.T,delta_out))*derivative

###逆伝播
def adam(param, m, v, error, t, 
         alpha = 0.001, beta1 = 0.9, beta2 = 0.999, tol = 10**(-8)):
    m_ver = m/(1-beta1**t)
    v_ver = v/(1-beta2**t)
    print(param-alpha*(m_ver/np.sqrt(v_ver+tol)))
    return param-alpha*(m_ver/np.sqrt(v_ver+tol)), beta1*m+(1-beta1)*error, beta2*v+(1-beta2)*error*error

##### 中間層のユニット数とパラメータの初期値
q0 = d
q = 100
q5 = m
w_in = np.random.normal(0, 0.3, size=(q, d+1))
w_hidden = np.random.normal(0, 0.3, size=(q, q))
w_out = np.random.normal(0, 0.3,size=(m,q+1))

##### adamの初期値
#m0 = np.zeros(shape=(N1, d+1))
#v0 = np.zeros(shape=(N1, d+1))
#m1 = np.zeros(shape=(d, N1+1))
#v1 = np.zeros(shape=(d, N1+1))


########## 誤差逆伝播法によるパラメータ推定
num_epoch = 20

eta = 10**(-2)

e = []
e_test = []
error = []
error_test = []

for epoch in range(0, num_epoch):
    index = np.random.permutation(n)

    eta_t = eta/(epoch+1)
    counter = 1

    for i in index:
        print("+++++++++++++++++[{}][{}/{}]+++++++++++++++++".format(epoch, counter, n))
        counter += 1

        xi = x_train[i, :, :]
        yi = y_train[i, :]

        Z = np.zeros(shape=(d+1,q+1))
        z = np.zeros(shape=(d+1,q))
        dz = np.zeros(shape=(d+1,q))
        y = np.zeros(shape=(d,m))
        v = np.zeros(shape=(d,m))
        X = np.zeros(shape=(d,d+1))
        z_out = np.zeros(shape=(m,1))

        
        ##### 順伝播    
        for j in range(d):
            X[j] = np.append(1,xi[j,])
            z[j+1], dz[j+1] = forward(X[j], z[j], w_in, w_hidden, sigmoid)
            Z[j+1] = np.append(1, z[j+1])
            y[j] = softmax(np.dot(w_out,Z[j]))
        
        z_out = softmax(np.dot(w_out,Z[d]))
        ##### 誤差評価: 誤差をeにappendする
        e.append(CrossEntoropy(z_out,yi))
        
        
        ##### 逆伝播
        delta_out = np.zeros(shape=(d, m))
        delta_t = np.zeros(shape=(d+1,q))
        delta_out[d-1] = y[d-1] - yi
        delta_t[d] = 0
        for j in range(d-1,0,-1):
            delta_t[j] = backward(w_hidden,w_out,delta_t[j+1],delta_out[j], dz[j+1])
                    
        ##### パラメータの更新
        D = delta_t[:d].T
        w_in = w_in - eta_t*np.dot(D,X)
        w_hidden = w_hidden - eta_t*np.dot(D,z[:d])
        w_out -= eta_t * np.dot(delta_out.T, Z[1:])
        
    ##### エポックごとの訓練誤差: eの平均をerrorにappendする
    error.append(sum(e)/n)
    print(error)
    e = []

    #### test error
    for i in range(0, n_test):
        xi = x_test[i, :, :]
        yi = y_test[i, :]
        ##### テスト誤差: 誤差をe_testにappendする
        #####: 誤差をe_testにappendする
        X = np.zeros(shape=(d, d + 1))
        z = np.zeros(shape=(d + 1, q))
        dz = np.zeros(shape=(d + 1, q))
        z[0] = 0
        dz[0] = 0
        for t in range(0, d):
            X[t] = np.append(1, xi[t])  # xiにバイアスを加える
            z[t + 1], dz[t + 1] = forward(X[t], z[t], w_in, w_hidden, sigmoid)
        z_out = softmax(np.dot(w_out, np.append(1, z[d])))
        e_test.append(CrossEntoropy(z_out, yi))
    
    ##### エポックごとの訓練誤差: e_testの平均をerror_testにappendする
    error_test.append(sum(e_test)/n_test)
    e_test = []

########## 誤差関数のプロット
plt.plot(error, label="training", lw=3)     #青線
plt.plot(error_test, label="test", lw=3)     #オレンジ線
plt.grid()
plt.legend(fontsize =16)
plt.savefig("./error.pdf", bbox_inches='tight', transparent=True)

########## 確率が高いクラスにデータを分類
prob = []
for j in range(0, n_test):    
    xi = x_test[j, :, :]
    yi = y_test[j, :]
    
    ##### 順伝播
    z[0] = np.zeros(q)
    for j in range(d):
        z[j+1], dz[j] = forward(np.append(1, xi[j,]), z[j], w_in, w_hidden, sigmoid)
        
    z_out = softmax(np.dot(w_out, np.append(1, z[d])))
    
    prob.append(z_out)

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
                D = x_test[l, :, :]
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