#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 21:47:31 2018

@author: nanzheng
"""

import numpy as np
from convolutional_neural_network import cnn
from mnist_operations import mnist_operations as mnist_opr


def config_lenet_5():
    """
    配置LeNet-5网络。
    """
    
    # 输入层
    size_input = np.array([28, 28])
    args_input = ("input", (size_input,))
    
    # C1卷积层
    connecting_matrix_C1 = np.ones([1, 6])
    size_conv_kernel_C1 = np.array([5, 5])
    stride_conv_kernel_C1 = 1
    padding_conv_C1 = 2
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
    connecting_matrix_C3 = \
            np.array([[1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1],
                      [1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1],
                      [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1],
                      [0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1],
                      [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1],
                      [0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1]])
    size_conv_kernel_C3 = np.array([5, 5])
    stride_conv_kernel_C3 = 1
    padding_conv_C3 = 0
    type_activation_C3 = "sigmoid"
    args_C3 = ("convoluting", (connecting_matrix_C3, size_conv_kernel_C3, 
                               stride_conv_kernel_C3, padding_conv_C3, 
                               type_activation_C3))
    
    # S4池化层
    type_pooling_S4 = "average"
    size_pool_kernel_S4 = np.array([2, 2])
    stride_pool_kernel_S4 = 2
    padding_pool_S4 = 0
    type_activation_S4 = "sigmoid"
    args_S4 = ("pooling", (type_pooling_S4, size_pool_kernel_S4, 
                           stride_pool_kernel_S4, padding_pool_S4, 
                           type_activation_S4))
    
    # C5卷积层
    connecting_matrix_C5 = np.ones([16, 120])
    size_conv_kernel_C5 = np.array([5, 5])
    stride_conv_kernel_C5 = 1
    padding_conv_C5 = 0
    type_activation_C5 = "sigmoid"
    args_C5 = ("convoluting", (connecting_matrix_C5, size_conv_kernel_C5, 
                               stride_conv_kernel_C5, padding_conv_C5, 
                               type_activation_C5))
    
    # F6全连接层
    n_nodes_F6 = 84
    type_activation_F6 = "sigmoid"
    args_F6 = ("full_connecting", (n_nodes_F6, type_activation_F6))
    
    # 输出层
    n_nodes_output = 10
    type_output = "softmax"
    args_output = ("output", (n_nodes_output, type_output))
    
    args = (args_input,
            args_C1,
            args_S2,
            args_C3,
            args_S4,
            args_C5,
            args_F6,
            args_output)
    LeNet_5 = cnn()
    LeNet_5.config(args)
    
    return LeNet_5


#def main():
file_name_train_images = "train-images-idx3-ubyte"
file_name_train_labels = "train-labels-idx1-ubyte"

train_images = mnist_opr.load_image_set(file_name_train_images) / 255
train_labels = mnist_opr.load_label_set(file_name_train_labels)

print("Configuring ...")
LeNet_5 = config_lenet_5()

print("Training ...")
LeNet_5.fit(train_images, train_labels, train_labels.shape[0])

file_name_test_images = "t10k-images-idx3-ubyte"
file_name_test_labels = "t10k-labels-idx1-ubyte"

test_images = mnist_opr.load_image_set(file_name_test_images) / 255
test_labels = mnist_opr.load_label_set(file_name_test_labels)

print("Testing ...")
correct_rate = LeNet_5.test(test_images, test_labels)


#import matplotlib.pyplot as plt
#
#
#plt.figure()
#for i in range(LeNet_5.layers[1].n_nodes):
#    plt.subplot(2, 3, i + 1)
#    plt.imshow(LeNet_5.layers[1].nodes[i].output)
#plt.show()
#
#plt.figure()
#for i in range(LeNet_5.layers[3].n_nodes):
#    plt.subplot(4, 4, i + 1)
#    plt.imshow(LeNet_5.layers[3].nodes[i].output)
#plt.show()
#
#plt.figure()
#plt.imshow(np.array(LeNet_5.layers[6].output).reshape(12, 7))
#plt.show()

