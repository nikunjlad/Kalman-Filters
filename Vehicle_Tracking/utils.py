"""
Written by: Rahmad Sadli
Website : https://machinelearningspace.com
If you want to redistribute it, just keep the author's name.
"""
import cv2
import numpy as np

colors_list = [
    [0, 0, 255],       # Blue
    [255, 140, 105],   # Light Salmon
    [218, 112, 214],   # Orchid
    [255, 0, 255],     # Magenta
    [0, 255, 255],     # Cyan
    [178, 34, 34],     # Firebrick
    [128, 0, 0],       # Maroon
    [0, 128, 0],       # Green
    [0, 0, 128],       # Navy
    [128, 128, 0],     # Olive
    [128, 0, 128],     # Purple
    [0, 128, 128],     # Teal
    [255, 0, 0],       # Red
    [255, 165, 0],     # Orange
    [255, 69, 0],      # Red-Orange
    [0, 128, 64],      # Dark Green
    [0, 255, 0],       # Green    
    [255, 140, 0],     # Dark Orange
    [205, 92, 92],     # Indian Red
    [0, 128, 128],     # Dark Cyan
    [255, 99, 71],     # Tomato
    [0, 139, 139],     # Dark Sea Green
    [70, 130, 180],    # Steel Blue
    [0, 206, 209],     # Dark Turquoise
    [255, 20, 147],    # Deep Pink
    [173, 216, 230],   # Light Blue
    [255, 255, 0],     # Yellow    
    [255, 215, 0],     # Gold
    [46, 139, 87],     # Sea Green
    [220, 20, 60]     # Crimson
] *100


def bboxes_x1y1x2y2_cxcywh(bboxes):        
    x1, y1, x2, y2 = bboxes[:,0], bboxes[:,1], bboxes[:,2], bboxes[:,3]
    w = x2 - x1
    h = y2 - y1
    c_x = x1 + (w / 2)
    c_y = y1 + (h / 2)
    return np.array(np.stack((c_x, c_y, w, h), axis=1))

def bboxes_cxcywh_x1y1x2y2(bboxes):
    c_x, c_y, w, h =  bboxes[:,0], bboxes[:,1], bboxes[:,2], bboxes[:,3]
    x1 = c_x - (w / 2)
    y1 = c_y - (h / 2)
    x2 = c_x + (w / 2)
    y2 = c_y + (h / 2)    
    return  np.array(np.stack((x1, y1, x2, y2), axis=1))

def bbox_x1y1x2y2_cxcywh(bbox):
    bbox = np.array(bbox)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    c_x = (bbox[0] + bbox[2]) / 2
    c_y = (bbox[1] + bbox[3]) / 2
    return np.array([c_x, c_y, w, h]).reshape((-1, 1))

def bbox_cxcywh_x1y1x2y2(bbox):
    bbox = np.array(bbox)
    c_x, c_y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    x1 = c_x - (w / 2)
    y1 = c_y - (h / 2)
    x2 = c_x + (w / 2)
    y2 = c_y + (h / 2)
    return np.array([x1, y1, x2, y2]).reshape((-1, 1))

def PutText(image, 
            text,
            pos=(0,0),
            text_color=(255,255,255), 
            bg_color=(255,255,255),             
            scale=1,
            thickness=1,
            margin=2,
            transparent=False, 
            font=cv2.FONT_HERSHEY_SIMPLEX, 
            alpha=0.5):
    
    txt_size = cv2.getTextSize(text, font, scale, thickness)
    w_text, h_text= txt_size[0][0], txt_size[0][1]

    x1_text = pos[0] + margin
    y1_text = pos[1] + h_text
    x1_rect = pos[0]
    y1_rect = pos[1] 
    x2_rect = x1_rect + w_text + margin
    y2_rect = y1_rect + h_text + margin
        
    if transparent:    
        mask = image.copy()
        cv2.rectangle(mask, (x1_rect, y1_rect), (x2_rect, y2_rect), bg_color, -1)
        image = cv2.addWeighted(image, 1 - alpha, mask, alpha, 0)        
        cv2.putText(image, text, (x1_text,y1_text), font, scale, text_color, thickness, cv2.LINE_AA)
    else:
        cv2.rectangle(image, (x1_rect, y1_rect), (x2_rect, y2_rect), bg_color, -1)        
        cv2.putText(image, text, (x1_text,y1_text), font, scale, text_color, thickness, cv2.LINE_AA)
    
    return image