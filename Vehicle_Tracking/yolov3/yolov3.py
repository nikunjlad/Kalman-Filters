"""
YOLOv3 Pytorch
Module: yolov3.py
Written by: Rahmad Sadli
Website : https://machinelearningspace.com
"""
import torch
import torch.nn.functional as nn
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from yolov3.utils.utils import transform_prediction, load_json



class Upsample(nn.Upsample):
    def __init__(self, in_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_channels=in_channels
        self.out_channels=self.in_channels        

class Shortcut(nn.Module):
    def __init__(self, in_channels):   
        super(Shortcut, self).__init__()     
        self.in_channels=in_channels
        self.out_channels=in_channels        

class Yolo(nn.Module):
    def __init__(self,scale):   
        super(Yolo, self).__init__()            
        self.scale=scale

class Route(nn.Module):
    def __init__(self,route):   
        super(Route, self).__init__()    
        self.route=route

 
class YOLOv3NET(nn.Module):
    def __init__(self,config_path):
        super(YOLOv3NET,self).__init__()
        self.module_list = nn.ModuleList()
        self.lay_num = -1
        self.config = load_json(config_path)
        self.anchors = self.config["anchors"]
        self.masks = self.config["masks"]
        self.im_size = self.config["im_size"][0]
        self.num_classes = self.config["num_classes"]
        self.Darknet()
    
    def Darknet(self,inputs=None):        
        filters = self.Conv_2D(3, 32, stride=1, k_size=3, bias=False, inputs=inputs)
        filters = self.Conv_2D(filters, 64, stride=2, k_size=3, bias=False, inputs=inputs)
        
        filters = self.Blocks_Conv(1, filters, 32, 64, stride=1, k_size=3, bias = False, inputs= inputs)
        filters = self.Downsample(filters,128)
        filters = self.Blocks_Conv(2, filters, 64, 128, stride=1, k_size=3, bias = False, inputs= inputs)
        filters = self.Downsample(filters,256)
        filters = self.Blocks_Conv(8, filters, 128, 256, stride=1, k_size=3, bias = False, inputs= inputs)
        filters = self.Downsample(filters,512)
        filters = self.Blocks_Conv(8, filters, 256, 512, stride=1, k_size=3, bias = False, inputs= inputs)
        filters = self.Downsample(filters,1024)
        filters = self.Blocks_Conv(4, filters, 512, 1024, stride=1, k_size=3, bias = False, inputs= inputs)        
        filters = self.Blocks_End(3, filters, 512, 1024, stride=1, k_size=3, bias = False, inputs= inputs)
        self.Yolo_(scale=0)
        route1, route2 = [-4] , [-1, 61]       
        filters=self.Route_Blocks(0,inputs, route1 , route2)
        filters = self.Blocks_End(3, filters, 256, 512, stride=1, k_size=3, bias = False, inputs= inputs)
        self.Yolo_(scale=1)
        route1, route2 = [-4] , [-1, 36]       
        filters=self.Route_Blocks(1,inputs, route1 , route2)
        filters = self.Blocks_End(3, filters, 128, 256, stride=1, k_size=3, bias = False, inputs= inputs)
        self.Yolo_(scale=2)

    def Route_Blocks(self, rtBlockNum, inputs, route1=[], route2=[]):
        if len(route1)>0 and len(route2)>0:
            filters = self.Route_(route1)             
            filterOut = 256 if rtBlockNum==0 else 128
            filters = self.Conv_2D(filters, filterOut, stride=1, k_size=1, bias=False, inputs=inputs)
            filters = self.Upsample_(filters)                        
            filters = self.Route_(route2)

        else: 
            raise ValueError("Oops!  That was no valid route indexes. Route indexes can't be empty!")

        return filters

    def Yolo_(self,scale):
        self.lay_num += 1
        yolo_layer=Yolo(scale)
        module= nn.Sequential()
        module.add_module(f"yolo_{self.lay_num}",yolo_layer)
        self.module_list.append(module)

    def Route_(self, route=[]):
        self.lay_num += 1

        def get_filters(module):    
            md= module[0]    
            return md.in_channels, md.out_channels   

        if len(route) > 1:
            start,end =route[0],route[1]
            _,filterOut1 = get_filters(self.module_list[self.lay_num+start])
            _,filterOut2 = get_filters(self.module_list[end])
            filterOut = filterOut1+filterOut2            
        else:
            start=route[0]
            _,filterOut =  get_filters(self.module_list[self.lay_num+start])

        route_layer=Route(route)
        module= nn.Sequential()
        module.add_module(f"route_{self.lay_num}",route_layer)
        self.module_list.append(module)    
        return filterOut
    
    def ShortCut_(self,filters):
        self.lay_num += 1
        shortcut = Shortcut(filters)
        module= nn.Sequential()
        module.add_module(f"shortcut_{self.lay_num}",shortcut)
        self.module_list.append(module)

    def Upsample_(self,filters):
        self.lay_num += 1
        upsample =Upsample(filters,scale_factor=2,mode='nearest')
        module= nn.Sequential()
        module.add_module(f"upsample_{self.lay_num}",upsample)
        self.module_list.append(module)        
        return filters

    def Conv_2D(self, filterIn, filterOut, stride=1, k_size=3, bias=False, inputs=None, bn=True, activation="leaky"):
        self.lay_num += 1
        if inputs is None:
            module = nn.Sequential()
            conv= nn.Conv2d(filterIn,filterOut, kernel_size=k_size, stride=stride, padding=(k_size-1)//2, bias= bias)
            module.add_module(f"conv_{self.lay_num}",conv)
            if bn:
                bn = nn.BatchNorm2d(filterOut)           
                #bn = BatchNorm(filterOut)             
                module.add_module(f"bn_{self.lay_num}", bn)            
            if activation=="leaky":
                activn = nn.LeakyReLU(0.1, inplace=True)
                #activn = LeakyReLU(0.1, inplace=True)
                module.add_module(f"leaky_{self.lay_num}", activn)                
            self.module_list.append(module)        
        return filterOut

    def Blocks_Conv(self,num_conv, filterIn, filterOut1, filterOut2,stride=1, k_size=3, bias = False, inputs= None):
        for _ in range(num_conv):
            filters = self.Conv_2D(filterIn,filterOut1, k_size=1)
            filters = self.Conv_2D(filters, filterOut2, k_size=3)
            self.ShortCut_(filters)
        return filterOut2
    
    def Blocks_End(self,num_conv, filters, filterOut1, filterOut2,stride=1, k_size=3, bias = False, inputs= None):                        
        for _ in range(num_conv):
            filters = self.Conv_2D(filters,filterOut1, k_size=1)
            filters = self.Conv_2D(filters, filterOut2, k_size=3)            
        filters = self.Conv_2D(filters, 255, k_size=1,bias=True, bn = False, activation="linear" )
        return filters

    def Downsample(self,filterIn, filterOut, stride=2, k_size=3, bias=False, inputs=None):
        filters = self.Conv_2D(filterIn, filterOut, stride=stride, k_size=k_size, bias=bias, inputs=inputs)
        return filters
        
    
    def forward(self, x, CUDA):
        
        detections = []
        outputs = {}        
        scale=0
        for i in range(len(self.module_list)):
            module=self.module_list[i]                        
            module_type = type(module[0]).__name__
                        
            if module_type == "Conv2d" or module_type == "Upsample":                

                x = self.module_list[i](x)                
                outputs[i] = x                   

            elif module_type == "Route":
                module=self.module_list[i] 
                route = module[0].route

                if len(route) > 1:                    
                    map1 = outputs[i + route[0]]
                    map2 = outputs[route[1]]
                                        
                    x = torch.cat((map1, map2), 1)                    
                else:                    
                    x = outputs[i + route[0]]
                outputs[i] = x
            
            elif  module_type == "Shortcut":                
                x = outputs[i-1] + outputs[i-3]
                outputs[i] = x
            
            elif module_type == 'Yolo':   
                scale=module[0].scale                
                anchors = self.anchors
                masks = self.masks[scale]
                anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
                anchors = [anchors[i] for i in masks]
                                
                x = transform_prediction(x, self.im_size, anchors, self.num_classes, CUDA)

                detections = x if not len(detections) else torch.cat((detections, x), 1)                                                
                outputs[i] = outputs[i-1]          
        try:
            return detections
        except:
            return 0            