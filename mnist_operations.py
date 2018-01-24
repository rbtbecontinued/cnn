#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 21:26:47 2018

@author: nanzheng
"""

import numpy as np
import struct


class mnist_operations():
    """
    解析MNIST数据集。
    """
    
    @staticmethod
    def load_image_set(file_name):
        """
        解析MNIST图像。
        
        Parameters
        ----------
        file_name: 文件名
        
        Returns
        -------
        imgs: imgNum * width * height array，imgNum为图片的数量，
                width、height分别为每张图片的宽度和高度
        """
        
        binfile = open(file_name, 'rb')
        buffers = binfile.read()
    
        head = struct.unpack_from('>IIII', buffers, 0)
        
        offset = struct.calcsize('>IIII')
        imgNum = head[1]
        width = head[2]
        height = head[3]
        
        bits = imgNum * width * height
        bitsString = '>' + str(bits) + 'B'    # like '>47040000B'
    
        imgs = struct.unpack_from(bitsString, buffers, offset)
        
        binfile.close()
        imgs = np.reshape(imgs, [imgNum, width, height])
        
        return imgs
    
    
    @staticmethod
    def load_label_set(file_name):
        """
        解析MNIST标签。
        
        Parameters
        ----------
        file_name: 文件名
        
        Returns
        -------
        labels: array，类别标签
        """
        
        binfile = open(file_name, 'rb')
        buffers = binfile.read()
        
        head = struct.unpack_from('>II', buffers, 0)
        
        offset = struct.calcsize('>II')
        imgNum = head[1]
        
        numString = '>' + str(imgNum) + 'B'
        
        labels = struct.unpack_from(numString, buffers, offset)
        
        binfile.close()
        labels = np.reshape(labels, [-1])
        
        return labels
    
    
def test():
    import matplotlib.pyplot as plt
    
    file_name = 'train-images-idx3-ubyte'
    binfile = open(file_name, 'rb')
    buf = binfile.read()
    
    index = 0
    magic, numImages, numRows, numColumns = \
            struct.unpack_from('>IIII', buf, index)
    index += struct.calcsize('>IIII')
    
    im = struct.unpack_from('>784B', buf, index)
    index += struct.calcsize('>784B')
    
    im = np.array(im).reshape(28, 28)
    
    plt.figure()
    plt.imshow(im, cmap='gray')
    plt.show()
    
    return None


if __name__ == "__main__":
    test()
    
    