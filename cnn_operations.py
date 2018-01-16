#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 19:32:31 2018

@author: nanzheng
"""

import numpy as np


class cnn_operations():
    """
    卷积神经网络中的运算。
    """
    
    @staticmethod
    def calc_loss(Y, tilde_Y):
        """
        计算网络代价。
        $$loss = \frac{1}{2N} \sum_{i = 1}^N \| y_i - \tilde{y}_i \|^2$$
        
        Parameters
        ----------
        Y: N * M array，各训练样本对应的类别标签，N为训练样本个数，M为类别数
        
        tilde_Y: N * M array，预测出的各训练样本的类别标签
        
        Returns
        -------
        loss: 均方误差代价
        """
        
        # 训练样本个数
        n_samples = Y.shape[0]
        # 网络代价
        loss = 0
        for i in range(n_samples):
            loss += np.sum((Y[i, :] - tilde_Y[i, :])**2)
        loss /= (2 * n_samples)
        
        return loss
    
    
    @staticmethod
    def convolute_2d(feature_map, kernel, size_kernel, stride, padding):
        """
        二维卷积。卷积核不翻转。
        
        Parameters
        ----------
        feature_map: 2-d array，输入特征图
        
        kernel: 2-d array，卷积核
        
        size_kernel: array，卷积核尺寸
        
        stride: 卷积核步长
        
        padding: 边缘补零的宽度
        
        Returns
        -------
        output: 2-d array，卷积后特征图
        """
        
        # 输入特征图尺寸
        size_feature_map = np.asarray(feature_map.shape)
        # 输出特征图尺寸
        size_output = [int( \
            (size_feature_map[0] + 2 * padding - size_kernel[0]) / stride + 1),
            int( \
            (size_feature_map[1] + 2 * padding - size_kernel[1]) / stride + 1)]
        
        if padding == 0:
            feature_map_padded = feature_map
        elif padding > 0:
            # 补零后的特征图
            feature_map_padded = np.zeros(size_feature_map + 2 * padding)
            feature_map_padded[padding: -padding, padding: -padding] += \
                    feature_map
                    
        output = np.empty(size_output)
        
        for i in range(0, feature_map_padded.shape[0], stride):
            for j in range(0, feature_map_padded.shape[1], stride):
                sub_mat = feature_map_padded[i: np.min([i + size_kernel[0], 
                        feature_map_padded.shape[0]]), 
                        j: np.min([j + size_kernel[1], 
                        feature_map_padded.shape[1]])]
                shape_sub_mat = sub_mat.shape
                if (shape_sub_mat[0] < size_kernel[0]) or \
                        (shape_sub_mat[1] < size_kernel[1]):
                    break
                
                output[int(i/stride), int(j/stride)] = \
                        np.sum(sub_mat * kernel)
        
        return output
    
    
    @staticmethod
    def pool(feature_map, type_pooling, size_kernel, stride, padding):
        """
        池化。
        
        Parameters
        ----------
        feature_map: 2-d array，输入特征图
        
        type_pooling: 池化核类型，{"max", "average"}
        
        size_kernel: array，池化核尺寸
        
        stride: 池化核步长
        
        padding: 边缘补零的宽度
        
        Returns
        -------
        output: 2-d array，池化后特征图
        """
        
        # 输入特征图尺寸
        size_feature_map = np.asarray(feature_map.shape)
        # 输出特征图尺寸
        size_output = [int( \
            (size_feature_map[0] + 2 * padding - size_kernel[0]) / stride + 1),
            int( \
            (size_feature_map[1] + 2 * padding - size_kernel[1]) / stride + 1)]
        
        if padding == 0:
            feature_map_padded = feature_map
        elif padding > 0:
            # 补零后的特征图
            feature_map_padded = np.zeros(size_feature_map + 2 * padding)
            feature_map_padded[padding: -padding, padding: -padding] += \
                    feature_map
                    
        output = np.empty(size_output)
        
        if type_pooling is "max":
            func = np.max
        elif type_pooling is "average":
            #池化层每个神经元有一个权值，此处只需求和，不需求均值
            func = np.sum
        
        for i in range(0, feature_map_padded.shape[0], stride):
            for j in range(0, feature_map_padded.shape[1], stride):
                sub_mat = feature_map_padded[i: np.min([i + size_kernel[0], 
                        feature_map_padded.shape[0]]), 
                        j: np.min([j + size_kernel[1], 
                        feature_map_padded.shape[1]])]
                shape_sub_mat = sub_mat.shape
                if (shape_sub_mat[0] < size_kernel[0]) or \
                        (shape_sub_mat[1] < size_kernel[1]):
                    break
                
                output[int(i/stride), int(j/stride)] = func(sub_mat)
                
        return output
    
    
    @staticmethod
    def upsample_conv_2d(delta_next_layer, kernel, size_kernel, stride):
        """
        当前层为池化层，下一层为卷积层时的上采样。
        
        Parameters
        ----------
        delta_next_layer: 2-d array，下一层（卷积层）中一个神经元的灵敏度map
        
        kernel: 2-d array，（下一层中一个神经元的一个）卷积核
        
        size_kernel: array，（下一层中一个神经元的一个）卷积核的尺寸
        
        stride: （下一层中一个神经元的一个）卷积核的步长
        
        Returns
        -------
        delta_padded: 2-d array，当前（池化）层中一个神经元的灵敏度map（边缘补零后）
        """
        
        # 下一层（卷积层）中一个神经元的灵敏度map的尺寸
        size_delta_next_layer = np.asarray(delta_next_layer.shape)
        # 边缘补零后当前层一中一个神经元灵敏度map的尺寸
        size_delta_padded = \
                [size_delta_next_layer[0] * stride + size_kernel[0] - stride, 
                 size_delta_next_layer[1] * stride + size_kernel[1] - stride]
        delta_padded = np.zeros(size_delta_padded)
        for i in range(delta_next_layer.shape[0]):
            for j in range(delta_next_layer.shape[1]):
                # 卷积核不翻转
                delta_padded[i * stride: i * stride + size_kernel[0], 
                             j * stride: j * stride + size_kernel[1]] += \
                                     delta_next_layer[i, j] * kernel
        
        return delta_padded
    
    
    @classmethod
    def upsample_pool(cls, delta_next_layer, type_pooling, 
                      size_kernel, stride):
        """
        当前层为卷积层，下一层为池化层时的上采样。
        
        Parameters
        ----------
        delta_next_layer: 2-d array，下一层（池化层）中一个神经元的灵敏度map
        
        type_pooling: （下一层中一个神经元的）池化核类型，{"max", "average"}
        
        size_kernel: array，（下一层中一个神经元的）池化核尺寸
        
        stride: （下一层中一个神经元的）池化核步长
        
        Returns
        -------
        delta_padded: 2-d array，当前（卷积）层中一个神经元的灵敏度map（边缘补零后）
        """
        
        if type_pooling is "max":
            # TODO: 
            pass
        
        elif type_pooling is "average":
            # 池化时实际进行了求和运算，故权值均为1
            kernel = np.ones(size_kernel)
            delta_padded = cls.upsample_conv_2d(delta_next_layer, kernel, 
                                                size_kernel, stride)
            
        return delta_padded
    
    
    @staticmethod
    def inv_conv_2d(feature_map, size_kernel, stride, padding, delta):
        """
        计算网络代价对卷积层中一个神经元的一个卷积核的偏导数。
        $$\frac{\partial loss}{\partial k_{ij}^{(l)}} = 
            \sum_{u, v} (\delta_j^{(l)})_{u, v} (p_i^{(l - 1)})_{u, v}$$
        
        Parameters
        ----------
        feature_map: 2-d array，输入特征图
        
        size_kernel: array，卷积核尺寸
        
        stride: 卷积核的步长
        
        padding: 边缘补零的宽度
        
        delta: 2-d array，卷积层中一个神经元的灵敏度map
        
        Returns
        -------
        output: 2-d array，$\frac{\partial loss}{\partial k_{ij}^{(l)}}$，
                尺寸与卷积核尺寸相同
        """
        
        # 输入特征图尺寸
        size_feature_map = np.asarray(feature_map.shape)
        
        if padding == 0:
            feature_map_padded = feature_map
        elif padding > 0:
            # 补零后的特征图
            feature_map_padded = np.zeros(size_feature_map + 2 * padding)
            feature_map_padded[padding: -padding, padding: -padding] += \
                    feature_map
                    
        output = 0
        for i in range(0, feature_map_padded.shape[0], stride):
            for j in range(0, feature_map_padded.shape[1], stride):
                sub_mat = feature_map_padded[i: np.min([i + size_kernel[0], 
                        feature_map_padded.shape[0]]), 
                        j: np.min([j + size_kernel[1], 
                        feature_map_padded.shape[1]])]
                shape_sub_mat = sub_mat.shape
                
                if (shape_sub_mat[0] < size_kernel[0]) or \
                        (shape_sub_mat[1] < size_kernel[1]):
                    break
                
                output += delta[int(i/stride), int(j/stride)] * sub_mat
                
        return output
    
    
    @staticmethod
    def _relu(x):
        """
        计算ReLU函数值。
        """
        
        return np.max([0, x])
    
    
    @staticmethod
    def _sigmoid(x):
        """
        计算Sigmoid函数值。
        """
        
        return 1 / (1 + np.exp(-x))
    
    
    @classmethod
    def _tanh(cls, x):
        """
        计算tanh函数值。
        """
        
        return 2 * cls._sigmoid(2 * x) - 1
    
    
    @classmethod
    def activate(cls, feature_map, type_activation):
        """
        激活。
        
        Parameters
        ----------
        feature_map: 2-d array，输入特征图
        
        type_activation: 激活函数类型，{"relu", "sigmoid", "tanh", None}
        
        Returns
        -------
        output: 2-d array，激活后特征图，与输入特征图尺寸相同
        """
        
        if type_activation is None:
            return feature_map
        
        elif type_activation is "relu":
            func = cls._relu
        elif type_activation is "sigmoid":
            func = cls._sigmoid
        elif type_activation is "tanh":
            func = cls._tanh
            
        size_feature_map = np.asarray(feature_map.shape)
        
        output = np.empty(size_feature_map)
        for i in range(size_feature_map[0]):
            for j in range(size_feature_map[1]):
                output[i, j] = func(feature_map[i, j])
                
        return output
    
    
def test():
    feature_map = np.ones([11, 11])
    kernel = np.array([[1, 2], [3, 4]])
    size_kernel = np.array([2, 2])
    stride = 1
    padding = 0
    output = cnn_operations.convolute_2d( \
            feature_map, kernel, size_kernel, stride, padding)
    
    size_kernel = np.array([2, 2])
    stride = 2
    padding = 0
    output = cnn_operations.pool( \
            output, "average", size_kernel, stride, padding)
    
    output = cnn_operations.activate(output, "sigmoid")
    
    return None


if __name__ == "__main__":
    test()
    
    