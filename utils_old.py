#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from collections import Counter


# In[2]:


def intersection_over_union(box_preds, box_labels, box_format="midpoint"):
    """
    parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct Labels of Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
    
    returns:
        tensor: intersection over union for all examples
    """
    if box_format == 'midpoint':
        box1_x1 = box_preds[..., 0:1] - box_preds[..., 2:3]/2
        box1_x2 = box_preds[..., 0:1] + box_preds[..., 2:3]/2
        box1_y1 = box_preds[..., 1:2] - box_preds[..., 3:4]/2
        box1_y2 = box_preds[..., 1:2] + box_preds[..., 3:4]/2
        
        box2_x1 = box_labels[..., 0:1] - box_preds[..., 2:3]/2
        box2_x2 = box_labels[..., 0:1] + box_preds[..., 2:3]/2
        box2_y1 = box_labels[..., 1:2] - box_preds[..., 3:4]/2
        box2_y2 = box_labels[..., 1:2] + box_preds[..., 3:4]/2
        
    elif box_format == 'corners':
        box1_x1 = box_preds[..., 0:1]
        box1_x2 = box_preds[..., 2:3]
        box1_y1 = box_preds[..., 1:2]
        box1_y2 = box_preds[..., 3:4]
        
        box2_x1 = box_labels[..., 0:1]
        box2_x2 = box_labels[..., 2:3]
        box2_y1 = box_labels[..., 1:2]
        box2_y2 = box_labels[..., 3:4]           
        
    x1 = torch.max(box1_x1, box2_x1)

    x2 = torch.min(box1_x2, box2_x2)
    y1 = torch.max(box1_y1, box2_y1)
    y2 = torch.min(box1_y2, box2_y2)
    
    intersection = (x2-x1).clamp(0)*(y2-y1).clamp(0)
    box1_area = abs((box1_x2-box1_x1)*(box1_y2-box1_y1))
    box2_area = abs((box2_x2-box2_x1)*(box2_y2-box2_y1))
    union = box1_area + box2_area -intersection
    
    return intersection/union+1e-6


# In[3]:


box_preds = torch.randn(4,7,7,4)
box_labels= torch.randn(4,7,7,4)
intersection_over_union(box_preds, box_labels).shape


# In[4]:


# import shutil
# from pathlib import Path
# import zipfile


# In[5]:


# source = Path.cwd()/'labels.zip'
# extract_folder = Path.cwd().parent/'labels_test'
# extract_folder.mkdir(parents=True, exist_ok=True)
# with zipfile.ZipFile(source, mode='r') as zip_ref:
#     zip_ref.extractall(extract_folder)


# In[6]:


# source = Path.cwd()/'labels.zip'
# extract_folder = Path.cwd().parent/'labels_test'
# extract_folder.mkdir(parents=True, exist_ok=True)
# shutil.unpack_archive(source, extract_folder, format='zip')


# In[7]:


def non_max_supression(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    does non max supression to given boxes
    Parameters:
        bboxes(list): list of lists containing all bboxes specified as
        [class_pred, probablity, x1, y1, x2, y2]
        iou_threshold (float): where predicted bboxes i correct
        threshold (float):  threshold to remove predicted bboxes (independent of IoU) 
        box_format (str): midpoint or corners used to specify bboxes
    
    Returns:
        list: bboxes after performing nms
    """
    
    assert type(bboxes) == list
    
    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key= lambda x: x[1], reverse=True)
    bboxes_after_nms = []
    
    while bboxes:
        chosen_box = bboxes.pop(0)
        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format
            ) < iou_threshold        
            
        ]
        
        bboxes_after_nms.append(chosen_box)
        
    return bboxes_after_nms


# In[8]:


def mean_avg_precision(pred_boxes, truth_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20):
    """
    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2] #train_idx=which image
        true_boxes (list): Similar as pred_boxes except all the correct ones 
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes

    
    Returns:
        float: mAP value across all classes given a specific IoU threshold 
    """
    epsilon = 1e-6
    average_precision = []
    

    for i in range(num_classes):
        detections = []
        ground_truths = []
        
        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class i        
        for detection in pred_boxes:
            if detection[1] == i:
                detections.append(detection)
                
        
        for truth_box in truth_boxes:
                if truth_box[1] == i:
                    ground_truths.append(truth_box)
                
        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}       
        bboxes_amount = Counter([i[0] for i in ground_truths])
               
        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        bboxes_amount = {key:torch.zeros(val) for key, val in bboxes_amount.items()}
            
        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))
        total_true_bboxes = len(ground_truths)
        
        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue
        
        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_image = [
                bbox
                for bbox in ground_truths
                if bbox[0] == detection[0] 
            ]
            
#             num_gts = len(ground_truth_image)  #debug this later
            best_iou = 0
            for idx, gt in enumerate(ground_truth_image):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format
                
                )
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
                
            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if bboxes_amount[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                else:
                    FP[detection_idx] = 1
            
            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1
            
        tp_cumsum = torch.cumsum(TP, dim=0)
        fp_cumsum = torch.cumsum(FP, dim=0)
        
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + epsilon)
        recalls = tp_cumsum / (total_true_bboxes + epsilon)
        
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        area = torch.trapz(precisions, recalls)
        
        average_precision.append(area)
        
    return sum(average_precision) / len(average_precision)


# In[9]:


# pred_boxes = [[0, 2, 0.6, 0.2, 0.4, 0.4, 0.6], [1, 1, 0.7, 0.2, 0.4, 0.4, 0.6]]
# truth_boxes = [[0, 2, 0.5, 0.2, 0.4, 0.4, 0.6], [1, 2, 0.9, 0.2, 0.4, 0.4, 0.6],[1, 1, 0.8, 0.2, 0.4, 0.4, 0.6]]
# a = mean_avg_precision(pred_boxes, truth_boxes)
# print(a)


# In[10]:


def convert_cell_boxes(predictions, s=7):
    batch_size = predictions.shape[0]
    predictions = predictions.to("cpu")
    predictions = predictions.reshape(batch_size, s, s, 30)
    bboxes1 = predictions[..., 21:25]
    bboxes2 = predictions[..., 26:30]
    
    class_pred = torch.argmax(predictions[..., :20], dim=-1).unsqueeze(-1)
    best_prob = torch.max(predictions[..., 20:21], predictions[..., 25:26])
    scores = torch.stack((predictions[..., 20:21], predictions[..., 25:26]), dim=0)
    best_box = torch.argmax(scores,dim=0)
    best_boxes = (1-best_box)*bboxes1 + best_box*bboxes2
    
    cell_indices = torch.arange(s).repeat(batch_size,s,1).unsqueeze(-1)
    x = 1/s*(best_boxes[...,:1] + cell_indices)
    y = 1/s*(best_boxes[...,1:2] + cell_indices.permute(0,2,1,3))
    w_h = 1 / s * best_boxes[..., 2:4]
    conv_boxes = torch.cat((x,y,w_h), dim=-1)
    
    pred_converted_boxes = torch.cat(
        (class_pred, best_prob, conv_boxes), dim=-1
    )
    
    return pred_converted_boxes

import unittest

preds = torch.randn(8,7,7, 30)
class Test_utils(unittest.TestCase):
    def test_convert_cell_boxes(self):
        self.assertEqual(convert_cell_boxes(preds).shape, torch.Size([8, 7, 7, 6]))
        
# test_module = Test_utils()
# test_module.test_convert_cell_boxes()

if __name__ == "__main__":
    unittest.main()
# In[11]:


preds = torch.randn(8,7,7, 30)
convert_cell_boxes(preds).shape


# In[12]:


##turn tensor matrixes to python lists
def cellboxes_to_boxes(out, s=7):
    """
        Returns:
        tensor: torch.tensor() with size (batch_size, s*s, 6)
    """
    batch_size = out.shape[0]
    converted_pred  = convert_cell_boxes(out).reshape(batch_size, s*s, -1)
    converted_pred[...,0] = converted_pred[...,0].long()
    
    all_bboxes = []
    
    for batch_idx in range(batch_size):
        bboxes = []
        
        for bbox_idx in range(s*s):
            bboxes.append([x.item() for x in converted_pred[batch_idx, bbox_idx, : ]])
            
        all_bboxes.append(bboxes)
        
    return all_bboxes
    


# In[13]:


a = cellboxes_to_boxes(preds)
torch.tensor(a).shape


# In[14]:


def get_bboxes(
    loader,
    model,
    iou_threshold,
    threshold,
    pred_format="cells",
    box_format="midpoint",
    device="cuda",
    ):
    all_pred_boxes = []
    all_true_boxes = []
    
    model.eval()
    train_idx = 0
    
    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        
        with torch.inference_mode():
            predictions = model(x)
            
        bboxes = cellboxes_to_boxes(predictions)
        true_bboxes = cellboxes_to_boxes(y)
        
        batch_size = x.shape[0]
        for idx in range(batch_size):
            nms_boxes = non_max_supression(
                bboxes[idx],
                iou_threshold,
                threshold,
                box_format
            )
            
            #if batch_idx == 0 and idx == 0:
            #    plot_image(x[idx].permute(1,2,0).to("cpu"), nms_boxes)
            #    print(nms_boxe
            
            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)
            
            for box in true_bboxes[idx]:
                # many will get converted to pred 0
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)
                    
            train_idx += 1
            
    model.train()
            
    return all_pred_boxes, all_true_boxes
            
    


# In[15]:


def plot_images(image,bboxs):
    pass

