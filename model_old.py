#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn


# In[2]:


architecture_config = [
    (7,64,2,3),
    'M',
    (3,192,1,1),
    'M',
    (1,128,1,0),
    (3,256,1,1),
    (1,256,1,0),
    (3,512,1,1),
    'M',
    [(1,256,1,0),(3,512,1,1),4],
    (1,512,1,0),
    (3,1024,1,1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),    
       
]


# In[ ]:


# class CNNBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, **kwargs):
#         super().__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
#         self.batchnorm = nn.BatchNorm2d(out_channels)
#         self.leakyrelu = nn.LeakyReLU(0.1)
        
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.batchnorm(x)
#         x = self.leakyrelu(x)
#         return x


# In[ ]:


# class Yolov1(nn.Module):
#     def __init__(self, in_channels=3, **kwargs):
#         super().__init__()
#         self.in_channels = in_channels
#         self.architecture = architecture_config
#         self.darknet = self._create_conv_layers(self.architecture)
#         self.fcs = self._create_fcs(**kwargs)
        
#     def forward(self, x):
#         x = self.darknet(x)
#         x = self.fcs(torch.flatten(x,start_dim=1))
# #         x = self.fcs(x)
#         return x
        
#     def _create_conv_layers(self, architecture):
#         layers = []
        
#         for i in architecture:
#             if type(i) == tuple:
#                 layers += [
#                     CNNBlock(in_channels=self.in_channels, out_channels=i[1], kernel_size=i[0], stride=i[2],
#                               padding=i[3])
#                 ]                
#             elif type(i) == str:
#                 layers += [
#                     nn.MaxPool2d(kernel_size=2, stride=2)
#                 ]
#             elif type(i) == list:
#                 conv1 = i[0]
#                 conv2 = i[1]
#                 repetition = i[2]
#                 for _ in range(repetition):
#                     layers += [
#                         CNNBlock(in_channels,
#                                  conv1[1],
#                                  conv1[0],
#                                  conv1[2],
#                                  conv1[3])
#                     ]
#                     layers += [
#                         CNNBlock(conv1[1],
#                                  conv2[1],
#                                  conv2[0],
#                                  conv2[2],
#                                  conv2[3])
#                     ]
#                     in_channels = conv2[1]
                    
#             return nn.Sequential(*layers)
        
#     def _create_fcs(self, split_size=7, num_boxes=2, num_classes=20):
#         s, c, b = split_size, num_classes, num_boxes
#         x = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(s * s * 1024, 496),
#             nn.Dropout(0),
#             nn.LeakyReLU(0.1),
#             nn.Linear(496, s * s * (c+b*5))             #debug later with 4096 the real number
#         )
#         return x
        


# In[3]:


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))


class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(Yolov1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
#         return self.fcs(torch.flatten(x, start_dim=1))
        return self.fcs(x)

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple:
                layers.append(
                    CNNBlock(
                        in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3],
                    )
                )
                in_channels = x[1]

            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]

                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(
                            in_channels,
                            conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                    ]
                    layers += [
                        CNNBlock(
                            conv1[1],
                            conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3],
                        )
                    ]
                    in_channels = conv2[1]

        return nn.Sequential(*layers)

    def _create_fcs(self, split_size=7, num_boxes=2, num_classes=20):
        S, B, C = split_size, num_boxes, num_classes

        # In original paper this should be
        # nn.Linear(1024*S*S, 4096),
        # nn.LeakyReLU(0.1),
        # nn.Linear(4096, S*S*(B*5+C))

        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 496),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S * S * (C + B * 5)),
        )


# In[4]:


a = torch.randn(2, 3, 448 , 448)
model = Yolov1()
model(a).shape

