import torch
import numpy as np
import json


def load_json(path):
    with open(path, "r") as f:
        config = json.load(f)
    return config

def load_coco_names(file_path):
    with open(file_path, 'r') as f:
        class_names = [line.strip() for line in f]
    return class_names

def bbox_iou(box1, box2):
    x1 = torch.max(box1[:, 0], box2[:, 0])
    y1 = torch.max(box1[:, 1], box2[:, 1])
    x2 = torch.min(box1[:, 2], box2[:, 2])
    y2 = torch.min(box1[:, 3], box2[:, 3])

    inter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

    area_box1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area_box2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    iou = inter_area / (area_box1 + area_box2 - inter_area)

    return iou


def transform_prediction(prediction, inp_dim, anchors, num_classes, CUDA = True):
    device = 'cuda' if CUDA else 'cpu'

    stride =  inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    num_anchors = len(anchors)

    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]
    anchors = torch.tensor(anchors).to(device).repeat(grid_size*grid_size, 1)

    prediction = prediction.view(-1, (5 + num_classes )*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(-1, grid_size*grid_size*num_anchors, 5 + num_classes)

    bbox_center = torch.sigmoid(prediction[:,:, :2])
    confidence = torch.sigmoid(prediction[:,:,4])
    bbox_shape = torch.exp(prediction[:,:,2:4])*anchors

    grid_len = torch.arange(grid_size).float()
    x_offset = grid_len.view(1, -1).repeat(grid_size, 1).view(-1, 1)
    y_offset = grid_len.view(-1, 1).repeat(1, grid_size).view(-1, 1)
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0).to(device)
    bbox_center  +=  x_y_offset

    classes = torch.sigmoid(prediction[:,:, 5 : 5 + num_classes])
    output = torch.cat((bbox_center, bbox_shape, confidence.unsqueeze(-1), classes), dim=-1)
    output[:,:,:4] *= stride

    return output


def non_max_suppresion(prediction, iou_threshold):
    boxes = prediction[:, :4]
    scores = prediction[:, 4]
    sorted_indices = torch.argsort(scores, descending=True)
    boxes = boxes[sorted_indices]
    scores = scores[sorted_indices]

    selected_indices = []
    while boxes.size(0) > 0:
        max_box = boxes[0]
        selected_indices.append(sorted_indices[0])
        if boxes.size(0) == 1:
            break
        ious = bbox_iou(max_box.unsqueeze(0), boxes[1:])
        mask = ious < iou_threshold
        boxes = boxes[1:][mask]
        scores = scores[1:][mask]
        sorted_indices = sorted_indices[1:][mask]

    selected_indices = torch.tensor(selected_indices, dtype=torch.long)
    image_pred = prediction[selected_indices]
    return image_pred


def post_processing(prediction, confidence, num_classes, nms = True, nms_conf = 0.4):

    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    prediction = prediction*conf_mask
    non_zero_ind =  (torch.nonzero(prediction[:,:,4]))
    prediction = prediction[:,non_zero_ind[:,1], :]

    prediction[:, :, :4] = torch.stack([
        prediction[:, :, 0] - prediction[:, :, 2] / 2,
        prediction[:, :, 1] - prediction[:, :, 3] / 2,
        prediction[:, :, 0] + prediction[:, :, 2] / 2,
        prediction[:, :, 1] + prediction[:, :, 3] / 2
    ], dim=2)

    prediction = prediction.squeeze(0)
    max_conf, ind_max_conf = torch.max(prediction[:,5:5+ num_classes], 1)
    max_conf = max_conf.float().unsqueeze(1)
    ind_max_conf = ind_max_conf.float().unsqueeze(1)
    prediction = torch.cat((prediction[:,:5], max_conf, ind_max_conf), 1)
    prediction = non_max_suppresion(prediction, iou_threshold=nms_conf)

    return prediction
