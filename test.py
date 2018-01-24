#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 20:37:37 2018

@author: nanzheng
"""

import numpy as np
import matplotlib.pyplot as plt
from convolutional_neural_network import cnn


def config_net():
    """
    配置网络。
    """
    
    # 输入层
    size_input = np.array([8, 8])
    args_input = ("input", (size_input,))
    
    # C1卷积层
    connecting_matrix_C1 = np.ones([1, 2])
    size_conv_kernel_C1 = np.array([5, 5])
    stride_conv_kernel_C1 = 1
    padding_conv_C1 = 0
    type_activation_C1 = "sigmoid"
    args_C1 = ("convoluting", (connecting_matrix_C1, size_conv_kernel_C1, 
                               stride_conv_kernel_C1, padding_conv_C1, 
                               type_activation_C1))
    
    # S2池化层
    type_pooling_S2 = "average"
    size_pool_kernel_S2 = np.array([2, 2])
    stride_pool_kernel_S2 = 2
    padding_pool_S2 = 0
    type_activation_S2 = "sigmoid"
    args_S2 = ("pooling", (type_pooling_S2, size_pool_kernel_S2, 
                           stride_pool_kernel_S2, padding_pool_S2, 
                           type_activation_S2))
    
    # C3卷积层
    connecting_matrix_C3 = np.ones([2, 2])
    size_conv_kernel_C3 = np.array([2, 2])
    stride_conv_kernel_C3 = 1
    padding_conv_C3 = 0
    type_activation_C3 = "sigmoid"
    args_C3 = ("convoluting", (connecting_matrix_C3, size_conv_kernel_C3, 
                               stride_conv_kernel_C3, padding_conv_C3, 
                               type_activation_C3))
    
    # 输出层
    n_nodes_output = 2
    type_output = "softmax"
    args_output = ("output", (n_nodes_output, type_output))
    
    args = (args_input,
            args_C1,
            args_S2,
            args_C3,
            args_output)
    cnn_net = cnn()
    cnn_net.config(args)
    
    return cnn_net


n_train = 10000
X_train = 0.2 * np.random.randn(8, 8, n_train)
Y_train = np.random.randint(2, size=n_train)
for i in range(Y_train.shape[0]):
    if Y_train[i] == 0:
        X_train[1, :, i] += np.ones(8)
    elif Y_train[i] == 1:
        X_train[:, 1, i] += np.ones(8)

size_batch = 50
n_epochs = 500
cnn_net = config_net()
cnn_net.fit(X_train, Y_train, size_batch=size_batch, n_epochs=n_epochs)

n_test = 1000
X_test = 0.2 * np.random.randn(8, 8, n_test)
Y_test = np.random.randint(2, size=n_test)
for i in range(Y_test.shape[0]):
    if Y_test[i] == 0:
        X_test[1, :, i] += np.ones(8)
    elif Y_test[i] == 1:
        X_test[:, 1, i] += np.ones(8)
        
correct_rate = cnn_net.test(X_test, Y_test)

plt.figure()
for i in range(cnn_net.layers[1].n_nodes):
    plt.subplot(1, 2, i + 1)
    plt.imshow(cnn_net.layers[1].nodes[i].conv_kernels[0], cmap="gray")
plt.show()

