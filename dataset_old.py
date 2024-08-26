#!/usr/bin/env python
# coding: utf-8

# In[11]:


import torch
import os
import pandas as pd
from PIL import Image
import cv2


# In[12]:


class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, image_dir, label_dir, S=7, B=2, C=20, transform=None):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        
        self.annonations = pd.read_csv(csv_file)
            
    def __len__(self):
        return len(self.annonations)

    def __getitem__(self, idx):
        label_path = os.path.join(self.label_dir, self.annonations.iloc[idx, 1])
        image_path = os.path.join(self.image_dir, self.annonations.iloc[idx, 0])
        
        boxes = []
        label_path = os.path.join(self.label_dir, self.annonations.iloc[idx, 1])
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(i) if float(i) != int(float(i)) else int(i)
                    for i in label.replace('\n','').split()
                ]
                boxes.append([class_label, x, y, width, height])
                
        image = Image.open(image_path)
        
        if self.transform:
            boxes = torch.tensor(boxes)
            image, boxes = self.transform(image, boxes)
            boxes = boxes.tolist()
        
        label_matrix = torch.zeros((self.S, self.S, self.C+self.B*5))
        for box in boxes:
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i
            
            width_cell, height_cell = width * self.S, height * self.S
            
            if label_matrix[i, j, 20] == 0:
                label_matrix[i, j, 20] = 1
                
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )
                
                label_matrix[i, j, 21:25] = box_coordinates
                
                label_matrix[i, j, class_label] = 1
                
        return image, label_matrix
            

                
 
    def _test_func(self, idx):
        boxes = []
        label_path = os.path.join(self.label_dir, self.annonations.iloc[idx, 1])
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(i) if float(i) != int(float(i)) else int(i)
                    for i in label.replace('\n','').split()
                ]
                boxes.append([class_label, x, y, width, height])

        return boxes    
                
            
            
            



