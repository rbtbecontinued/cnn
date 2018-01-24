#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 20:30:39 2018

@author: nanzheng
"""

import numpy as np
import matplotlib.pyplot as plt
from cnn_operations import cnn_operations as cnn_opr


class cnn():
    """
    卷积神经网络。
    """
    
    def __init__(self):
        # 网络的层数
        self.n_layers = 0
        # list，网络中的各层
        self.layers = []
        # array，网络的输出
        self.output = None
        # 网络的代价
        self.loss = None
        # 权值的学习率
        self.learning_rate_weight = 0.1
        # 偏置的学习率
        self.learning_rate_bias = 0.1
        
        
    def config(self, args):
        """
        配置网络。
        
        Parameters
        ----------
        args: tuple，args[i][0]为第i层的类型，{"input", "convoluting", "pooling", 
                "full_connecting", "output"}，args[i][1]为tuple，第i层的配置参数
        """
        
        self.n_layers = len(args)
        
        prior_layer = None
        for i in range(self.n_layers):
            # 配置网络的各层
            new_layer = cnn_layer(args[i][0])
            
            if i > 0:
                prior_layer = self.layers[-1]
                
                # 当前层设为上一层的next_layer
                self.layers[-1].next_layer = new_layer
                
            new_layer.config(args[i][1], self.learning_rate_weight, 
                             self.learning_rate_bias, prior_layer)
            
            self.layers.append(new_layer)
            
        return None
        
    
    def _feed_forward(self, x):
        """
        前向传播。
        
        Parameters
        ----------
        x: 3-d array，一个batch的输入图像，
                每个通道的尺寸为x.shape[0] * x.shape[1]，
                x.shape[2]为当前batch中图像的个数 * 每幅图像的通道数
        """
        
        # 输入层前向传播
        self.layers[0].feed_forward(x)
        # 其它各层前向传播
        for i in range(1, self.n_layers):
            self.layers[i].feed_forward(x)
            
        # self.layers[-1].n_nodes * size_batch array，网络的输出
        self.output = np.ndarray.flatten( \
                np.array(self.layers[-1].output)).reshape( \
                self.layers[-1].n_nodes, -1)
        
        return None
    
    
    def _back_propagate(self, y):
        """
        反向传播。
        
        Parameters
        ----------
        y: array，输入样本对应的类别标签
        """
        
        # 输出层反向传播
        self.layers[-1].back_propagate(y)
        # 其它各层反向传播
        for i in range(self.n_layers - 2, 0, -1):
            self.layers[i].back_propagate()
        
        return None
    
    
    def fit(self, X, Y, size_batch=1, n_epochs=1):
        """
        训练。
        下降方式为随机梯度下降。
        
        Parameters
        ----------
        X: 3-d array，训练集，
                X[:, :, i: i + self.layers[0].n_nodes]为一个训练样本（图片），
                self.layers[0].n_nodes即为每幅图片的通道数
        
        Y: array，训练集对应的类别标签
        
        size_batch: 一个batch中训练样本的个数
        
        n_epochs: 迭代次数
        """
        
        self.size_batch = size_batch
        # 训练样本个数 * 每幅图片的通道数
        len_X = X.shape[-1]
        len_Y = Y.shape[0]
        # 每个epoch中batch的个数
        n_batches = int(np.ceil(len_X / self.layers[0].n_nodes / size_batch))
        
        loss = np.empty(n_epochs * n_batches)
        
        for i_epoch in range(n_epochs):
            print("Epoch: ", end="")
            print(i_epoch)
            for i_batch in range(n_batches):
                print("\tBatch: ", end="")
                print(i_batch, end="\t")
                y_offset = i_batch * size_batch
                x_offset = y_offset * self.layers[0].n_nodes
                
                # 将类别标签转换为向量
                y = np.zeros([self.layers[-1].n_nodes, size_batch])
                for i in range(size_batch):
                    if i > len_Y - y_offset - 1:
                        y = y[:, :, : i]
                        
                        break
                    
                    y[Y[y_offset + i], i] = 1
                    
                self._feed_forward(X[:, :, x_offset: x_offset + size_batch * \
                                     self.layers[0].n_nodes])
                
                loss[i_epoch * n_batches + i_batch] = \
                        cnn_opr.calc_loss(y.T, self.output.T)
                print("loss = ", end="")
                print(loss[i_epoch * n_batches + i_batch])
                
                self._back_propagate(y)
                
        self.loss = loss
                
        plt.figure()
        plt.plot(loss, "r-")
        plt.xlabel("Batches")
        plt.ylabel("Loss")
        plt.grid()
        plt.show()
        
        return None
    
    
    def test(self, X, Y):
        """
        验证。
        
        Parameters
        ----------
        X: 3-d array，验证集，
                X[:, :, i: i + self.layers[0].n_nodes]为一个验证样本（图片），
                self.layers[0].n_nodes即为每幅图片的通道数
        
        Y: array，训练集对应的类别标签
        
        Returns
        -------
        correct_rate: 验证集的分类正确率
        """
        
        n_correct = 0
        for i in range(0, X.shape[-1], self.layers[0].n_nodes):
            print("Test case: ", end="")
            print(i)
            y_predict = self.predict(X[:, :, i: i + self.layers[0].n_nodes])
            
            if y_predict == Y[i]:
                n_correct += 1
                
        correct_rate = n_correct / X.shape[-1]
        
        return correct_rate
    
    
    def predict(self, x):
        """
        预测。
        
        Parameters
        ----------
        x: 2-d或3-d array，输入样本（图像）
        
        Returns
        -------
        y_predict: 预测出的输入样本（图像）的类别
        """
        
        self._feed_forward(x.reshape(x.shape[0], x.shape[1], -1))
        
        # 根据网络输出层的类型，判定输入图像的类别
        if self.layers[-1].type_output is "softmax":
            y_predict = np.argmax(self.output[:, 0])
            
        elif self.layers[-1].type_output is "rbf":
            # TODO: 
            pass
        
        return y_predict
    
    
class cnn_layer():
    """
    卷积神经网络中的一层。
    """
    
    def __init__(self, type_layer):
        """
        Parameters
        ----------
        type_layer: 当前层的类型，{"input", "convoluting", "pooling", 
                            "full_connecting", "output"}
        """
        
        # 当前层的类型
        self.type = type_layer
        # 当前层中神经元的个数
        self.n_nodes = 0
        # list，当前层中各神经元
        self.nodes = []
        # 当前层的上一层
        self.prior_layer = None
        # 当前层的下一层
        self.next_layer = None
        # list，当前层的输出
        self.output = []
        # 权值的学习率
        self.learning_rate_weight = 0.0
        # 偏置的学习率
        self.learning_rate_bias = 0.0
        
        if self.type is "input":
            # array，输入图像（每个通道）的尺寸
            self.size_input = None
            
        elif self.type is "convoluting":
            # 2-d array，当前层与上一层各神经元的连接矩阵
            self.connecting_matrix = None
            # array，卷积核尺寸
            self.size_conv_kernel = None
            # 卷积核步长
            self.stride_conv_kernel = 1
            # 边缘补零的宽度
            self.padding_conv = 0
            # 激活函数类型，{"relu", "sigmoid", "tanh", None}
            self.type_activation = None
            
        elif self.type is "pooling":
            # 池化核类型，{"max", "average"}
            self.type_pooling = "max"
            # array，池化核尺寸
            self.size_pool_kernel = np.array([2, 2])
            # 池化核步长
            self.stride_pool_kernel = 2
            # 边缘补零的宽度
            self.padding_pool = 0
            # 激活函数类型，{"relu", "sigmoid", "tanh", None}
            self.type_activation = None
            
        elif self.type is "full_connecting":
            # 激活函数类型，{"relu", "sigmoid", "tanh", None}
            self.type_activation = None
            
        elif self.type is "output":
            # 输出层类型，{"softmax", "rbf"}
            self.type_output = "softmax"
            
            
    def config(self, args, learning_rate_weight, learning_rate_bias, 
               prior_layer=None):
        """
        配置网络的一层。
        
        Parameters
        ----------
        args: tuple，当前层的配置参数
        
        earning_rate_weight: 权值的学习率
        
        learning_rate_bias: 偏置的学习率
        
        prior_layer: 当前层的上一层
        """
        
        self.prior_layer = prior_layer
        self.learning_rate_weight = learning_rate_weight
        self.learning_rate_bias = learning_rate_bias
        
        if self.type is "input":
            size_input, = args
            
            # 输入图像为单通道
            if size_input.shape[0] == 2:
                self.n_nodes = 1
                self.size_input = size_input
                
            # 输入图像为多通道
            elif size_input.shape[0] == 3:
                # 每个神经元一个通道
                self.n_nodes = size_input[-1]
                # 输入图像每个通道的尺寸
                self.size_input = size_input[ : 2]
                
            self._config_input(self.size_input)
            
        elif self.type is "convoluting":
            connecting_matrix, size_conv_kernel, \
                    stride_conv_kernel, padding_conv, type_activation = args
            
            self.connecting_matrix = connecting_matrix
            self.n_nodes = connecting_matrix.shape[1]
            self.size_conv_kernel = size_conv_kernel
            self.stride_conv_kernel = stride_conv_kernel
            self.padding_conv = padding_conv
            self.type_activation = type_activation
            
            self._config_convoluting(connecting_matrix, size_conv_kernel, 
                                     stride_conv_kernel, padding_conv, 
                                     type_activation)
            
        elif self.type is "pooling":
            type_pooling, size_pool_kernel, \
                    stride_pool_kernel, padding_pool, type_activation = args
                    
            # 池化层神经元个数与上一层卷积层（或激活层）神经元个数相同
            self.n_nodes = self.prior_layer.n_nodes
            self.type_pooling = type_pooling
            self.size_pool_kernel = size_pool_kernel
            self.stride_pool_kernel = stride_pool_kernel
            self.ppadding_pool = padding_pool
            self.type_activation = type_activation
            
            self._config_pooling(type_pooling, size_pool_kernel, 
                                 stride_pool_kernel, padding_pool, 
                                 type_activation)
            
        elif self.type is "full_connecting":
            n_nodes, type_activation = args
            
            self.n_nodes = n_nodes
            self.type_activation = type_activation
            
            self._config_full_connecting(n_nodes, type_activation)
            
        elif self.type is "output":
            n_nodes, type_output = args
            
            self.n_nodes = n_nodes
            self.type_output = type_output
            
            self._config_output(n_nodes, type_output)
            
        # 初始化权值
        self._initialize()
            
        return None
    
    
    def _config_input(self, size_input):
        """
        配置输入层。
        
        Parameters
        ----------
        size_input: array，输入图像（每个通道）的尺寸
        """
        
        for i in range(self.n_nodes):
            new_node = cnn_node(self.type)
            
            args = (size_input,)
            new_node.config(args)
            
            self.nodes.append(new_node)
            
        return None
    
    
    def _config_convoluting(self, connecting_matrix, size_conv_kernel, 
                            stride_conv_kernel, padding_conv, type_activation):
        """
        配置卷积层。
        
        Parameters
        ----------
        connecting_matrix: M * N array，M为上一层神经元个数，N为当前层神经元个数，
                connecting_matrix[m, n]为1表示上一层第m个神经元与当前层第n个神经元
                连接，为0则表示不连接
        
        size_conv_kernel: array，卷积核尺寸
        
        stride_conv_kernel: 卷积核步长
        
        padding_conv: 边缘补零的宽度
        
        type_activation: 激活函数类型，{"relu", "sigmoid", "tanh", None}
        """
        
        for i in range(self.n_nodes):
            new_node = cnn_node(self.type)
            
            # 上一层中与当前神经元连接的神经元
            nodes_prior_layer = []
            for j in range(connecting_matrix.shape[0]):
                if connecting_matrix[j, i] == 1:
                    nodes_prior_layer.append(self.prior_layer.nodes[j])
                    
                    # 当前神经元添加至上一层中与之连接的神经元的nodes_next_layer
                    self.prior_layer.nodes[j].nodes_next_layer.append(new_node)
                    
            args = (nodes_prior_layer, size_conv_kernel, stride_conv_kernel, 
                    padding_conv, type_activation)
            new_node.config(args)
            
            self.nodes.append(new_node)
            
        return None
    
    
    def _config_pooling(self, type_pooling, size_pool_kernel, 
                        stride_pool_kernel, padding_pool, type_activation):
        """
        配置池化层。
        
        Parameters
        ----------
        type_pooling: 池化核类型，{"max", "average"}
        
        size_pool_kernel: array，池化核尺寸
        
        stride_pool_kernel: 池化核步长
        
        padding_pool: 边缘补零的宽度
        
        type_activation: 激活函数类型，{"relu", "sigmoid", "tanh", None}
        """
        
        for i in range(self.n_nodes):
            new_node = cnn_node(self.type)
            
            # 上一层中与当前神经元连接的神经元
            nodes_prior_layer = self.prior_layer.nodes[i]
            
            # 当前神经元添加至上一层中与之连接的神经元的nodes_next_layer
            self.prior_layer.nodes[i].nodes_next_layer.append(new_node)
            
            args = (nodes_prior_layer, type_pooling, size_pool_kernel, 
                    stride_pool_kernel, padding_pool, type_activation)
            new_node.config(args)
            
            self.nodes.append(new_node)
        
        return None
    
    
    def _config_full_connecting(self, n_nodes, type_activation):
        """
        配置全连接层。
        
        Parameters
        ----------
        n_nodes: 全连接层中神经元的个数
        
        type_activation: 激活函数类型，{"relu", "sigmoid", "tanh", None}
        """
        
        # 上一层中所有神经元与当前层中每个经元连接
        nodes_prior_layer = self.prior_layer.nodes
        args = (nodes_prior_layer, type_activation)
        
        # 上一层中神经元的个数
        n_nodes_prior_layer = len(nodes_prior_layer)
        
        for i in range(n_nodes):
            new_node = cnn_node(self.type)
            
            # 当前神经元添加至上一层中每个神经元的nodes_next_layer
            for j in range(n_nodes_prior_layer):
                self.prior_layer.nodes[j].nodes_next_layer.append(new_node)
                
            new_node.config(args)
            
            self.nodes.append(new_node)
            
        return None
    
    
    def _config_output(self, n_nodes, type_output):
        """
        配置输出层。
        
        Parameters
        ----------
        n_nodes: 输出层中神经元的个数，即类别数
        
        type_output: 输出层类型，{"softmax", "rbf"}
        """
        
        # 上一层中所有神经元与当前层中每个经元连接
        nodes_prior_layer = self.prior_layer.nodes
        args = (nodes_prior_layer, type_output)
        
        # 上一层中神经元的个数
        n_nodes_prior_layer = len(nodes_prior_layer)
        
        for i in range(n_nodes):
            new_node = cnn_node(self.type)
            
            # 当前神经元添加至上一层中每个神经元的nodes_next_layer
            for j in range(n_nodes_prior_layer):
                self.prior_layer.nodes[j].nodes_next_layer.append(new_node)
                
            new_node.config(args)
            
            self.nodes.append(new_node)
            
        return None
    
    
    def _initialize(self):
        """
        初始化网络的一层。
        采用Xavier方法。
        """
        
        if self.type is "convoluting":
            self._initialize_convoluting()
        elif self.type is "full_connecting":
            self._initialize_full_connecting()
        elif self.type is "output":
            self._initialize_output()
            
        return None
    
    
    def _initialize_convoluting(self):
        """
        初始化卷积层各神经元的权值。
        采用Xavier方法。
        """
        
        fan_out = self.n_nodes * np.prod(self.size_conv_kernel)
        
        if self.prior_layer.type is "input":
            fan_in = self.prior_layer.n_nodes * np.prod(self.size_conv_kernel)
            
            u = np.sqrt(6 / (fan_in + fan_out))
            
            for i in range(self.n_nodes):
                for j in range(self.nodes[i].n_conv_kernels):
                    self.nodes[i].conv_kernels[j] = u * 2 * \
                            (np.random.rand(self.size_conv_kernel[0], 
                                            self.size_conv_kernel[1]) - 0.5)
                            
        elif self.prior_layer.type is "pooling":
            for i in range(self.n_nodes):
                fan_in = np.sum(self.connecting_matrix[:, i]) * \
                        np.prod(self.size_conv_kernel)
                        
                u = np.sqrt(6 / (fan_in + fan_out))
                
                for j in range(self.nodes[i].n_conv_kernels):
                    self.nodes[i].conv_kernels[j] = u * 2 * \
                            (np.random.rand(self.size_conv_kernel[0], 
                                            self.size_conv_kernel[1]) - 0.5)
                            
        return None
    
    
    def _initialize_full_connecting(self):
        """
        初始化全连接层各神经元的权值。
        采用Xavier方法。
        """
        
        fan_in = self.prior_layer.n_nodes
        fan_out = self.n_nodes
        
        u = np.sqrt(6 / (fan_in + fan_out))
        
        for i in range(self.n_nodes):
            self.nodes[i].weights = u * 2 * (np.random.rand(fan_in) - 0.5)
            
        return None
    
    
    def _initialize_output(self):
        """
        初始化输出层各神经元的权值。
        采用Xavier方法。
        """
        
        self._initialize_full_connecting()
        
        return None
    
    
    def feed_forward(self, inputs=None):
        """
        网络的一层前向传播。
        
        Parameters
        ----------
        inputs: 3-d array，一个batch的输入图像，
                每个通道的尺寸为inputs.shape[0] * inputs.shape[1]，
                inputs.shape[2]为当前batch中图像的个数 * 每幅图像的通道数
                （只有当前神经元为输入层神经元时才有效）
        """
        
        if self.type is "input":
            self._feed_forward_input(inputs)
            
        elif self.type is "output":
            self._feed_forward_output()
            
        else:
            self.output = []
            for i in range(self.n_nodes):
                # 当前层中每个神经元前向传播
                self.nodes[i].feed_forward()
                
                self.output.append(self.nodes[i].output)
                
        return None
    
    
    def _feed_forward_input(self, inputs):
        """
        输入层前向传播。
        
        Parameters
        ----------
        inputs: 3-d array，一个batch的输入图像，
                每个通道的尺寸为inputs.shape[0] * inputs.shape[1]，
                inputs.shape[2]为当前batch中图像的个数 * 每幅图像的通道数
        """
        
        self.output = []
        # 输入图像为单通道，此时inputs[:, :, i]为每幅图像
        if self.n_nodes == 1:
            self.nodes[0].feed_forward(inputs)
            
            self.output.append(self.nodes[0].output)
            
        # 输入图像为多通道，此时inputs[:, :, i: i + 3]为每幅图像
        elif self.n_nodes > 1:
            for i in range(self.n_nodes):
                self.nodes[i].feed_forward(inputs[:, :, i: : self.n_nodes])
                
                self.output.append(self.nodes[i].output)
                
        return None
    
    
    def _feed_forward_output(self):
        """
        输出层前向传播。
        """
        
        if self.type_output is "softmax":
            # 输出层第一个神经元前向传播
            self.nodes[0].feed_forward()
            
            # size_batch * self.n_nodes array
            combinations = np.empty([self.nodes[0].combination.shape[-1], 
                                     self.n_nodes])
            combinations[:, 0] = self.nodes[0].combination.reshape(-1)
            
            # 输出层其它神经元前向传播
            for i in range(1, self.n_nodes):
                self.nodes[i].feed_forward()
                combinations[:, i] = self.nodes[i].combination.reshape(-1)
                
            # $e^{w_j^T x}, \forall j$
            exp_combinations = np.exp(combinations)
            # $\sum_{j = 1}^n e^{w_j^T x}$
            sum_exp = np.sum(exp_combinations, axis=1)
            
            self.output = []
            for i in range(self.n_nodes):
                # 输出层神经元的output为size_batch array
                # $\frac{e^{w_i^T x}}{\sum_{j = 1}^n e^{w_j^T x}}$
                self.nodes[i].output = exp_combinations[:, i] / sum_exp
                
                self.output.append(self.nodes[i].output)
            
        elif self.type_output is "rbf":
            # TODO: 
            pass
        
        return None
    
    
    def back_propagate(self, y=None):
        """
        网络的一层反向传播。
        
        Parameters
        ----------
        y: array，输入样本对应的类别标签
        """
        
        if self.type is "convoluting":
            self._back_propagate_convoluting()
        elif self.type is "pooling":
            self._back_propagate_pooling()
        elif self.type is "full_connecting":
            self._back_propagate_full_connecting()
        elif self.type is "output":
            self._back_propagate_output(y)
        
        return None
    
    
    def _back_propagate_convoluting(self):
        """
        卷积层反向传播。
        认为卷积层的下一层为池化层、全连接层或输出层，卷积层的上一层为池化层或输入层。
        """
        
        if self.next_layer.type is "pooling":
            self._bp_pooling_to_convoluting()
            
        elif self.next_layer.type is "full_connecting":
            self._bp_full_connecting_to_convoluting()
            
        elif self.next_layer.type is "output":
            self._bp_output_to_convoluting()
        
        return None
    
    
    def _bp_pooling_to_convoluting(self):
        """
        当前层为卷积层，下一层为池化层时的反向传播。
        """
        
        # TODO: 
        if self.type_activation is None:
            pass
        
        elif self.type_activation is "relu":
            pass
        
        elif self.type_activation is "sigmoid":
            for i in range(self.n_nodes):
                # 下一层（池化层）中与当前（卷积层）神经元连接的神经元只有一个
                node_next_layer = self.nodes[i].nodes_next_layer[0]
                # 池化层中一个神经元只有一个权值
                # TODO: 下一层池化类型为"max"时
                delta_padded = node_next_layer.weights[0] * \
                        cnn_opr.upsample_pool(node_next_layer.delta[:, :, 0], 
                                node_next_layer.type_pooling, 
                                node_next_layer.size_pool_kernel, 
                                node_next_layer.stride_pool_kernel)
                
                size_delta_padded = delta_padded.shape
                delta = np.zeros(self.nodes[i].output.shape)
                delta[ : size_delta_padded[0], : size_delta_padded[1], 0] = \
                        delta_padded
                
                for j in range(1, delta.shape[-1]):
                    delta[ : size_delta_padded[0], : size_delta_padded[1], j] = \
                        node_next_layer.weights[0] * \
                        cnn_opr.upsample_pool(node_next_layer.delta[:, :, j], 
                                node_next_layer.type_pooling, 
                                node_next_layer.size_pool_kernel, 
                                node_next_layer.stride_pool_kernel)
                
                self.nodes[i].delta = delta * \
                        (self.nodes[i].output - self.nodes[i].output**2)
                
                # 更新当前神经元的权值，即当前神经元的各卷积核
                for j in range(self.nodes[i].n_conv_kernels):
                    # 卷积层的上一层可能为池化层或输入层
                    delta_k = 0.0
                    for iter_in_batch in range(delta.shape[-1]):
                        delta_k += cnn_opr.inv_conv_2d( \
                                self.nodes[i].nodes_prior_layer[j].output[ \
                                :, :, iter_in_batch], 
                                self.size_conv_kernel, 
                                self.stride_conv_kernel, 
                                self.padding_conv, 
                                self.nodes[i].delta[:, :, iter_in_batch])
                    delta_k /= delta.shape[-1]
                    
                    self.nodes[i].conv_kernels[j] -= \
                            self.learning_rate_weight * delta_k
                    
                # 更新当前神经元的偏置
                self.nodes[i].bias -= self.learning_rate_bias * \
                        np.sum(self.nodes[i].delta) / delta.shape[-1]
                        
        elif self.type_activation is "tanh":
            pass
        
        return None
    
    
    def _bp_full_connecting_to_convoluting(self):
        """
        当前层为卷积层，下一层为全连接层时的反向传播。
        此时卷积层的每个输出特征图均为1 * 1 array。
        """
        
        # TODO: 
        if self.type_activation is None:
            pass
        
        elif self.type_activation is "relu":
            pass
        
        elif self.type_activation is "sigmoid":
            for i in range(self.n_nodes):
                delta = 0.0
                for j in range(len(self.nodes[i].nodes_next_layer)):
                    # 全连接层神经元的delta为size_batch array
                    delta += self.nodes[i].nodes_next_layer[j].weights[i] * \
                            self.nodes[i].nodes_next_layer[j].delta
                            
                delta *= (self.nodes[i].output[0, 0, :] - 
                          self.nodes[i].output[0, 0, :]**2)
                delta = delta.reshape(1, 1, -1)
                self.nodes[i].delta = delta
                
                # 更新当前神经元的权值，即当前神经元的各卷积核
                for j in range(self.nodes[i].n_conv_kernels):
                    # 卷积层的上一层可能为池化层或输入层
                    delta_k = 0.0
                    for iter_in_batch in range(delta.shape[-1]):
                        delta_k += cnn_opr.inv_conv_2d( \
                                self.nodes[i].nodes_prior_layer[j].output[ \
                                :, :, iter_in_batch], 
                                self.size_conv_kernel, 
                                self.stride_conv_kernel, 
                                self.padding_conv, 
                                self.nodes[i].delta[:, :, iter_in_batch])
                    delta_k /= delta.shape[-1]
                    
                    self.nodes[i].conv_kernels[j] -= \
                            self.learning_rate_weight * delta_k
                    
                # 更新当前神经元的偏置
                # self.nodes[i].delta实际上为1 * 1 * size_batch array
                self.nodes[i].bias -= self.learning_rate_bias * \
                        np.sum(self.nodes[i].delta) / delta.shape[-1]
                
        elif self.type_activation is "tanh":
            pass
        
        return None
    
    
    def _bp_output_to_convoluting(self):
        """
        当前层为卷积层，下一层为输出层时的反向传播。
        此时卷积层的每个输出特征图均为1 * 1 array。
        """
        
        self._bp_full_connecting_to_convoluting()
        
        return None
    
    
    def _back_propagate_pooling(self):
        """
        池化层反向传播。
        认为池化层的上一层为卷积层，池化层的下一层为卷积层、全连接层或输出层。
        """
        
        if self.next_layer.type is "convoluting":
            self._bp_convoluting_to_pooling()
            
        elif self.next_layer.type is "full_connecting":
            self._bp_full_connecting_to_pooling()
            
        elif self.next_layer.type is "output":
            self._bp_output_to_pooling()
            
        return None
    
    
    def _bp_convoluting_to_pooling(self):
        """
        当前层为池化层，下一层为卷积层时的反向传播。
        """
        
        # TODO: 
        if self.type_activation is None:
            pass
        
        elif self.type_activation is "relu":
            pass
        
        elif self.type_activation is "sigmoid":
            index_kernel = -1
            for j in range(self.next_layer.connecting_matrix.shape[0]):
                if self.next_layer.connecting_matrix[j, 0] == 1:
                    index_kernel += 1
                    
                    if index_kernel == 0:
                        delta_padded = cnn_opr.upsample_conv_2d( \
                            self.next_layer.nodes[0].delta[:, :, 0], 
                            self.next_layer.nodes[0].conv_kernels[index_kernel], 
                            self.next_layer.nodes[0].size_conv_kernel, 
                            self.next_layer.nodes[0].stride_conv_kernel)
                        
                        for n in range(self.n_nodes):
                            self.nodes[n].delta = np.zeros([ \
                                delta_padded.shape[0], 
                                delta_padded.shape[1], 
                                self.next_layer.nodes[0].delta.shape[-1]])
                            
                        self.nodes[j].delta[:, :, 0] = delta_padded
                        
                        for iter_in_batch in range(1, 
                                self.next_layer.nodes[0].delta.shape[-1]):
                            self.nodes[j].delta[:, :, iter_in_batch] += \
                                cnn_opr.upsample_conv_2d( \
                                self.next_layer.nodes[0].delta[ \
                                        :, :, iter_in_batch], 
                                self.next_layer.nodes[0].conv_kernels[ \
                                        index_kernel], 
                                self.next_layer.nodes[0].size_conv_kernel, 
                                self.next_layer.nodes[0].stride_conv_kernel)
                                
                    elif index_kernel > 0:
                        for iter_in_batch in range( \
                                self.next_layer.nodes[0].delta.shape[-1]):
                            self.nodes[j].delta[:, :, iter_in_batch] += \
                                cnn_opr.upsample_conv_2d( \
                                self.next_layer.nodes[0].delta[ \
                                        :, :, iter_in_batch], 
                                self.next_layer.nodes[0].conv_kernels[ \
                                        index_kernel], 
                                self.next_layer.nodes[0].size_conv_kernel, 
                                self.next_layer.nodes[0].stride_conv_kernel)
                                
            for i in range(1, self.next_layer.connecting_matrix.shape[1]):
                # 卷积层中每个神经元可能与上一层中多个神经元连接，
                # 即卷积层中的神经元可能有多个卷积核
                
                # 下一层（卷积层）中与当前神经元连接的神经元的卷积核的索引
                index_kernel = -1
                for j in range(self.next_layer.connecting_matrix.shape[0]):
                    # 下一层的第i个神经元与当前层的第j个神经元连接，
                    # 将下一层第i个神经元的delta传递至当前层第j个神经元
                    if self.next_layer.connecting_matrix[j, i] == 1:
                        index_kernel += 1
                        
                        for iter_in_batch in range( \
                                self.next_layer.nodes[i].delta.shape[-1]):
                            self.nodes[j].delta[:, :, iter_in_batch] += \
                                cnn_opr.upsample_conv_2d( \
                                self.next_layer.nodes[i].delta[ \
                                        :, :, iter_in_batch], 
                                self.next_layer.nodes[i].conv_kernels[ \
                                        index_kernel], 
                                self.next_layer.nodes[i].size_conv_kernel, 
                                self.next_layer.nodes[i].stride_conv_kernel)
                                
            for i in range(self.n_nodes):
                # 令delta与output尺寸相同
                delta = np.zeros(self.nodes[i].output.shape)
                size_delta_padded = self.nodes[i].delta.shape
                delta[ : size_delta_padded[0], : size_delta_padded[1], :] += \
                        self.nodes[i].delta
                
                self.nodes[i].delta = delta * \
                        (self.nodes[i].output - self.nodes[i].output**2)
                
                # 更新当前神经元的权值
                # $\frac{\partial loss}{\partial w} = \sum{\delta \dot z}$
                # 池化层中每个神经元只有一个权值
                self.nodes[i].weights[0] -= self.learning_rate_weight * \
                    np.sum(self.nodes[i].delta * self.nodes[i].combination) / \
                    self.nodes[i].delta.shape[-1]
                # 更新当前神经元的偏置
                # $\frac{\partial loss}{\partial b} = \sum{\delta}$
                self.nodes[i].bias -= self.learning_rate_bias * \
                    np.sum(self.nodes[i].delta) / self.nodes[i].delta.shape[-1]
                        
        elif self.type_activation is "tanh":
            pass
        
        return None
    
    
    def _bp_full_connecting_to_pooling(self):
        """
        当前层为池化层，下一层为全连接层时的反向传播。
        此时池化层的每个输出特征图为1 * 1 array。
        """
        
        # TODO: 
        if self.type_activation is None:
            pass
        
        elif self.type_activation is "relu":
            pass
        
        elif self.type_activation is "sigmoid":
            for i in range(self.n_nodes):
                delta = 0.0
                for j in range(len(self.nodes[i].nodes_next_layer)):
                    delta += self.nodes[i].nodes_next_layer[j].weights[i] * \
                            self.nodes[i].nodes_next_layer[j].delta
                            
                delta *= (self.nodes[i].output[0, 0, :] - \
                          self.nodes[i].output[0, 0, :]**2)
                self.nodes[i].delta = delta.reshape(1, 1, -1)
                
                # 更新当前神经元的权值
                self.nodes[i].weights[0] -= self.learning_rate_weight * \
                    np.sum(self.nodes[i].delta * self.nodes[i].combination) / \
                    self.nodes[i].shape[-1]
                # 更新当前神经元的偏置
                self.nodes[i].bias -= self.learning_rate_bias * \
                    np.sum(self.nodes[i].delta) / self.nodes[i].delta.shape[-1]
                    
        elif self.type_activation is "tanh":
            pass
        
        return None
    
    
    def _bp_output_to_pooling(self):
        """
        当前层为池化层，下一层为输出层时的反向传播。
        此时池化层的每个输出特征图为1 * 1 array。
        """
        
        self._bp_full_connecting_to_pooling()
        
        return None
    
    
    def _back_propagate_full_connecting(self):
        """
        全连接层反向传播。
        """
        
        # TODO: 
        if self.type_activation is None:
            pass
        
        elif self.type_activation is "relu":
            pass
        
        elif self.type_activation is "sigmoid":
            for i in range(self.n_nodes):
                # 计算当前神经元的灵敏度
                delta = 0.0
                for j in range(len(self.nodes[i].nodes_next_layer)):
                    # （认为全连接层的下一层为全连接层或输出层）
                    delta += self.nodes[i].nodes_next_layer[j].weights[i] * \
                            self.nodes[i].nodes_next_layer[j].delta
                # 对于sigmoid，$f'(z) = f(z) (1 - f(z))$
                delta *= (self.nodes[i].output[0, 0, :] - \
                          self.nodes[i].output[0, 0, :]**2)
                self.nodes[i].delta = delta
                
                # 更新当前神经元的权值
                for j in range(len(self.nodes[i].nodes_prior_layer)):
                    # 全连接层的上一层（卷积层）的输出为一个向量，
                    # 即上一层中每个神经元的output为1 * 1 * size_batch array
                    self.nodes[i].weights[j] -= \
                            self.learning_rate_weight * \
                            np.mean(self.nodes[i].delta * \
                            self.nodes[i].nodes_prior_layer[j].output[0, 0, :])
                # 更新当前神经元的偏置
                self.nodes[i].bias -= \
                        self.learning_rate_bias * np.mean(self.nodes[i].delta)
                        
        elif self.type_activation is "tanh":
            pass
        
        return None
    
    
    def _back_propagate_output(self, y):
        """
        输出层反向传播。
        
        Parameters
        ----------
        y: array，当前训练样本对应的类别标签
        """
        
        if self.type_output is "softmax":
            # self.n_nodes * size_batch array
            delta_y = np.array(self.output).reshape(self.n_nodes, -1) - y
            
            # 计算输出层各神经元的灵敏度，并更新权值和偏置
            for i in range(self.n_nodes):
                # $\delta_i^{(L)} = (\tilde{y}_i - y_i) f'(z_i^{(L)})$
                # $z_i^{(L)} = (w_i^{(L)})^T x^{(L - 1)} + b_i^{(L)}$
                
                # 对于softmax，$f'(z) = f(z) (1 - f(z))$
                # 输出层各神经元的output实际上为$f(z)$
                self.nodes[i].delta = \
                        delta_y[i, :] * (self.output[i] - self.output[i]**2)
                
                # 更新输出层当前神经元的权值
                # $w' = w - \eta \frac{\partial loss}{\partial w}$
                # $\frac{\partial loss}{\partial w} = \delta z^{(L - 1)}$
                for j in range(len(self.nodes[i].nodes_prior_layer)):
                    # 输出层的上一层为全连接层
                    # 全连接层的output为1 * 1 * size_batch array
                    self.nodes[i].weights[j] -= \
                            self.learning_rate_weight * \
                            np.mean(self.nodes[i].delta * \
                            self.nodes[i].nodes_prior_layer[j].output[0, 0, :])
                # 更新输出层当前神经元的偏置
                self.nodes[i].bias -= \
                        self.learning_rate_bias * np.mean(self.nodes[i].delta)
            
        elif self.type_output is "rbf":
            # TODO: 
            pass
        
        return None
    
    
class cnn_node():
    """
    卷积神经网络的一个神经元。
    """
    
    def __init__(self, type_node):
        """
        Parameters
        ----------
        type_layer: 当前神经元的类型，{"input", "convoluting", "pooling", 
                              "full_connecting", "output"}
        """
        
        # 神经元类型
        self.type = type_node
        # 上一层中与当前神经元连接的神经元
        self.nodes_prior_layer = None
        # 下一层中与当前神经元连接的神经元
        self.nodes_next_layer = []
        # 神经元的输出
        self.output = None
        # 神经元的灵敏度，
        # 当前神经元为全连接层或输出层神经元时，灵敏度为标量，
        # 当前神经元为卷积层或池化层神经元时，灵敏度为2-d array，尺寸与output相同
        # （实际上卷积层和池化层输出特征图中的每一个点为一个“神经元”）
        self.delta = 0.0
        
        if self.type is "input":
            # array，输入图像（每个通道）的尺寸
            self.size_input = None
            
        elif self.type is "convoluting":
            # 卷积核个数
            self.n_conv_kernels = 0
            # array，卷积核尺寸
            self.size_conv_kernel = None
            # list，卷积核
            self.conv_kernels = []
            # 卷积核步长
            self.stride_conv_kernel = 1
            # 边缘补零的宽度
            self.padding_conv = 0
            # 偏置
            self.bias = 0.0
            # 2-d array，卷积后（未经过激活函数）的特征图
            self.combination = None
            # 激活函数类型，{"relu", "sigmoid", "tanh", None}
            self.type_activation = None
            
        elif self.type is "pooling":
            # 池化核类型，{"max", "average"}
            self.type_pooling = "max"
            # array，池化核尺寸
            self.size_pool_kernel = np.array([2, 2])
            # 池化核步长
            self.stride_pool_kernel = 2
            # 边缘补零的宽度
            self.padding_pool = 0
            # array，权值
            self.weights = np.array([0.0])
            # 偏置
            self.bias = 0.0
            # 2-d array，池化后（未经过激活函数）的特征图
            self.combination = None
            # 激活函数类型，{"relu", "sigmoid", "tanh", None}
            self.type_activation = None
            
        elif self.type is "full_connecting":
            # array，权值
            self.weights = np.array([], dtype="float64")
            # 偏置
            self.bias = 0.0
            # array，$(w^{(l)})^T x^{(l - 1)} + b^{(l)}$
            self.combination = None
            # 激活函数类型，{"relu", "sigmoid", "tanh", None}
            self.type_activation = None
            
        elif self.type is "output":
            # 输出层类型，{"softmax", "rbf"}
            self.type_output = "softmax"
            # array，权值
            self.weights = np.array([], dtype="float64")
            # 偏置
            self.bias = 0.0
            # $(w^{(L)})^T x^{(L - 1)} + b^{(L)}$
            self.combination = 0.0
            
            
    def config(self, args):
        """
        配置神经元
        
        Parameters
        ----------
        args: tuple，当前神经元的配置参数
        """
        
        if self.type is "input":
            size_input, = args
            
            self._config_input(args)
            
        elif self.type is "convoluting":
            nodes_prior_layer, size_kernel, \
                    stride, padding, type_activation = args
            
            self._config_convoluting(nodes_prior_layer, size_kernel, 
                                     stride, padding, type_activation)
            
        elif self.type is "pooling":
            nodes_prior_layer, type_pooling, size_kernel, \
                    stride, padding, type_activation = args
                    
            self._config_pooling(nodes_prior_layer, type_pooling, size_kernel, 
                                 stride, padding, type_activation)
            
        elif self.type is "full_connecting":
            nodes_prior_layer, type_activation = args
            
            self._config_full_connecting(nodes_prior_layer, type_activation)
            
        elif self.type is "output":
            nodes_prior_layer, type_output = args
            
            self._config_output(nodes_prior_layer, type_output)
            
        return None
    
    
    def _config_input(self, size_input):
        """
        配置输入层神经元。
        
        Parameters
        ----------
        size_input: array，输入图像的尺寸
        """
        
        self.size_input = size_input
        
        return None
    
    
    def _config_convoluting(self, nodes_prior_layer, size_kernel, 
                            stride, padding, type_activation):
        """
        配置卷积层神经元。
        
        Parameters
        ----------
        nodes_prior_layer: list，上一层中与当前神经元连接的神经元（1个或多个）
        
        size_kernel: array，卷积核尺寸
        
        stride: 卷积核步长
        
        padding: 边缘补零的宽度
        
        type_activation: 激活函数类型，{"relu", "sigmoid", "tanh", None}
        """
        
        self.nodes_prior_layer = nodes_prior_layer
        self.n_conv_kernels = len(self.nodes_prior_layer)
        self.size_conv_kernel = size_kernel
        self.conv_kernels = [np.zeros(self.size_conv_kernel) \
                             for i in range(self.n_conv_kernels)]
        self.stride_conv_kernel = stride
        self.padding_conv = padding
        self.type_activation = type_activation
        
        return None
    
    
    def _config_pooling(self, nodes_prior_layer, type_pooling, size_kernel, 
                        stride, padding, type_activation):
        """
        配置池化层神经元。
        
        Parameters
        ----------
        nodes_prior_layer: list，上一层中与当前神经元连接的神经元（1个）
        
        type_pooling: 池化核类型，{"max", "average"}
        
        size_kernel: array，池化核尺寸
        
        stride: 池化核步长
        
        padding: 边缘补零的宽度
        
        type_activation: 激活函数类型，{"relu", "sigmoid", "tanh", None}
        """
        
        self.nodes_prior_layer = nodes_prior_layer
        self.type_pooling = type_pooling
        self.size_pool_kernel = size_kernel
        self.stride_pool_kernel = stride
        self.padding_pool = padding
        self.type_activation = type_activation
        
        # 初始化权值
        if self.type_pooling is "max":
            self.weights[0] = 1.0
        elif self.type_pooling is "average":
            self.weights[0] = 1 / np.prod(self.size_pool_kernel)
        
        return None
    
    
    def _config_full_connecting(self, nodes_prior_layer, type_activation):
        """
        配置全连接层神经元。
        
        Parameters
        ----------
        nodes_prior_layer: list，上一层中的所有神经元
        
        type_activation: 激活函数类型，{"relu", "sigmoid", "tanh", None}
        """
        
        self.nodes_prior_layer = nodes_prior_layer
        self.weights = np.zeros(len(self.nodes_prior_layer))
        self.type_activation = type_activation
        
        return None
    
    
    def _config_output(self, nodes_prior_layer, type_output):
        """
        配置输出层神经元。
        
        Parameters
        ----------
        nodes_prior_layer: list，上一层中的所有神经元
        
        type_output: 输出层类型，{"softmax", "rbf"}
        """
        
        self.nodes_prior_layer = nodes_prior_layer
        self.weights = np.zeros(len(self.nodes_prior_layer))
        self.type_output = type_output
        
        return None
    
    
    def feed_forward(self, inputs=None):
        """
        神经元前向传播。
        
        Parameters
        ----------
        inputs: 2-d array，输入图像（或其中一个通道），尺寸为self.size_input
                （只有当前神经元为输入层神经元时才有效）
        """
        
        if self.type is "input":
            self._feed_forward_input(inputs)
        elif self.type is "convoluting":
            self._feed_forward_convoluting()
        elif self.type is "pooling":
            self._feed_forward_pooling()
        elif self.type is "full_connecting":
            self._feed_forward_full_connecting()
        elif self.type is "output":
            self._feed_forward_output()
        
        return None
    
    
    def _feed_forward_input(self, inputs):
        """
        输入层神经元前向传播。
        
        Parameters
        ----------
        inputs: 3-d array，一个batch的输入图像（或其中一个通道），
                尺寸为inputs.shape[0] * inputs.shape[1]（即self.size_input），
                inputs.shape[2]为当前batch中图像的个数
        """
        
        self.output = inputs
        
        return None
    
    
    def _feed_forward_convoluting(self):
        """
        卷积层神经元前向传播。
        """
        
        # 每一批中训练样本的个数
        size_batch = self.nodes_prior_layer[0].output.shape[-1]
        
        # 当前batch中第一个样本前向传播
        combination = 0.0
        for i in range(self.n_conv_kernels):
            combination += cnn_opr.convolute_2d( \
                    self.nodes_prior_layer[i].output[:, :, 0], 
                    self.conv_kernels[i], self.size_conv_kernel, 
                    self.stride_conv_kernel, self.padding_conv)
        combination += self.bias
        
        # 根据当前batch中第一个样本确定self.combination、self.output的大小
        size_combination = combination.shape
        self.combination = np.empty([size_combination[0], size_combination[1], 
                                     size_batch])
        self.output = np.empty([size_combination[0], size_combination[1], 
                                size_batch])
        
        self.combination[:, :, 0] = combination
        self.output[:, :, 0] = \
                cnn_opr.activate(combination, self.type_activation)
        
        # 当前batch中其它样本前向传播
        for iter_in_batch in range(1, size_batch):
            combination = 0.0
            for i in range(self.n_conv_kernels):
                combination += cnn_opr.convolute_2d( \
                        self.nodes_prior_layer[i].output[:, :, iter_in_batch], 
                        self.conv_kernels[i], self.size_conv_kernel, 
                        self.stride_conv_kernel, self.padding_conv)
            combination += self.bias
            
            self.combination[:, :, iter_in_batch] = combination
            self.output[:, :, iter_in_batch] = \
                    cnn_opr.activate(combination, self.type_activation)
        
        return None
    
    
    def _feed_forward_pooling(self):
        """
        池化层神经元前向传播。
        """
        
        size_batch = self.nodes_prior_layer.output.shape[-1]
        
        combination = cnn_opr.pool(self.nodes_prior_layer.output[:, :, 0], 
                                   self.type_pooling, self.size_pool_kernel, 
                                   self.stride_pool_kernel, self.padding_pool)
        combination *= self.weights
        combination += self.bias
        
        size_combination = combination.shape
        self.combination = np.empty([size_combination[0], size_combination[1], 
                                     size_batch])
        self.output = np.empty([size_combination[0], size_combination[1], 
                                size_batch])
        
        self.combination[:, :, 0] = combination
        self.output[:, :, 0] = \
                cnn_opr.activate(combination, self.type_activation)
                
        for iter_in_batch in range(1, size_batch):
            combination = cnn_opr.pool( \
                    self.nodes_prior_layer.output[:, :, iter_in_batch], 
                    self.type_pooling, self.size_pool_kernel, 
                    self.stride_pool_kernel, self.padding_pool)
            combination *= self.weights
            combination += self.bias
            
            self.combination[:, :, iter_in_batch] = combination
            self.output[:, :, iter_in_batch] = \
                    cnn_opr.activate(combination, self.type_activation)
        
        # 灵敏度map置零
        self.delta = 0.0
        
        return None
    
    
    def _feed_forward_full_connecting(self):
        """
        全连接层神经元前向传播。
        """
        
        size_batch = self.nodes_prior_layer[0].output.shape[2]
        
        self.combination = np.empty([1, 1, size_batch])
        self.output = np.empty([1, 1, size_batch])
        
        for iter_in_batch in range(size_batch):
            combination = 0.0
            for i in range(len(self.nodes_prior_layer)):
                # 全连接层的上一层输出为一维向量，
                # 即上一层每个神经元输出的特征图尺寸为1 * 1
                combination += self.weights[i] * \
                        self.nodes_prior_layer[i].output[0, 0, iter_in_batch]
            combination += self.bias
        
            # combination为标量
            self.combination[0, 0, iter_in_batch] = combination
            self.output[:, :, iter_in_batch] = \
                    cnn_opr.activate(self.combination[:, :, iter_in_batch], 
                                     self.type_activation)
        
        return None
    
    
    def _feed_forward_output(self):
        """
        输出层神经元前向传播。
        """
        
        if self.type_output is "softmax":
            size_batch = self.nodes_prior_layer[0].output.shape[2]
            
            self.combination = np.empty([1, 1, size_batch])
            self.output = np.empty([1, 1, size_batch])
            
            for iter_in_batch in range(size_batch):
                # $softmax(w_i) = 
                #     \frac{e^{w_i^T x}}{\sum_{j = 1}^n e^{w_j^T x}}$
                # 此处只计算$w_i^T x$，其余运算在cnn_layer.feed_forward()中进行
                combination = 0.0
                for i in range(len(self.nodes_prior_layer)):
                    combination += self.weights[i] * \
                        self.nodes_prior_layer[i].output[0, 0, iter_in_batch]
                combination += self.bias
                
                # 输出层combination为标量
                self.combination[0, 0, iter_in_batch] = combination
            
        elif self.type_output is "rbf":
            # TODO: 
            pass
        
        return None
    
    
def test():
    """
    LeNet-5网络。
    """
    
    # 输入层
    size_input = np.array([32, 32])
    args_input = ("input", (size_input,))
    
    # C1卷积层
    connecting_matrix_C1 = np.ones([1, 6])
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


if __name__ == "__main__":
    test()
    
    