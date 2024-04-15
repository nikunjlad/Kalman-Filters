"""
YOLOv3 Pytorch
Module: load_weights.py
Written by: Rahmad Sadli
Website : https://machinelearningspace.com
"""

import torch
import torch.nn as nn
import numpy as np
    
def load_weights(model, weightfile):    
    # Open the weights file
    fp = open(weightfile, "rb")

    # The first 4 values are header information
    header = np.fromfile(fp, dtype=np.int32, count=5)
    header = torch.from_numpy(header)

    # The rest of the values are the weights
    for i in range(len(model.module_list)):
        module=model.module_list[i]            
        module_def = [type(module[j]).__name__ for j in range(len(module))]
        module_type = type(module[0]).__name__        

        if module_type == "Conv2d":            
            conv = module[0]
            filters = conv.out_channels
            in_dim = conv.in_channels
            k_size = conv.kernel_size[0]

            if "BatchNorm2d" in module_def:                    
                bn = module[1]
                bn_weights = np.fromfile(fp, dtype=np.float32, count=4 * filters)                    
                bn_weights = bn_weights.reshape((4, filters))#[[1, 0, 2, 3]] 
                bn_dict = {
                        'weight': torch.from_numpy(bn_weights[1]),
                        'bias': torch.from_numpy(bn_weights[0]),
                        'running_mean': torch.from_numpy(bn_weights[2]),
                        'running_var': torch.from_numpy(bn_weights[3])
                }   
                bn.load_state_dict(bn_dict)                

            else:
                conv_bias = np.fromfile(fp, dtype=np.float32, count=filters)
                conv.bias.data.copy_(torch.from_numpy(conv_bias))

            conv_shape = (filters, in_dim, k_size, k_size)
            conv_weights = np.fromfile(fp, dtype=np.float32, count=np.prod(conv_shape))
            conv_weights = conv_weights.reshape(conv_shape)#.transpose([0, 1, 2, 3])
            conv.weight.data.copy_(torch.from_numpy(conv_weights))
    
    return model


