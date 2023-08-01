import cv2
import os
from os import path as osp
import random
import numpy as np
from PIL import Image
import imutils
from math import *
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
from torch.utils.data import Dataset

class Transforms():
    def __init__(self):
        pass
    
    def rotate(self, image, landmarks, angle):
        angle = random.uniform(-angle, +angle)

        transformation_matrix = torch.tensor([
            [+cos(radians(angle)), -sin(radians(angle))], 
            [+sin(radians(angle)), +cos(radians(angle))]
        ])

        image = imutils.rotate(np.array(image), angle)

        landmarks = landmarks - 0.5
        new_landmarks = np.matmul(landmarks, transformation_matrix)
        new_landmarks = new_landmarks + 0.5
        return Image.fromarray(image), new_landmarks

    def resize(self, image, landmarks, img_size):
        image = TF.resize(image, img_size)
        return image, landmarks

    def color_jitter(self, image, landmarks):
        color_jitter = transforms.ColorJitter(brightness=0.3, 
                                              contrast=0.3,
                                              saturation=0.3, 
                                              hue=0.1)
        image = color_jitter(image)
        return image, landmarks

    def crop_face(self, image, landmarks, crops):
        left = crops[0]
        top = crops[1]
        width = crops[2]- crops[0]
        height = crops[3]- crops[1]

        image = TF.crop(image, top, left, height, width)

        
        img_shape = np.array(image).shape
        landmarks = torch.tensor(landmarks) - torch.tensor([[left, top]])
        landmarks = landmarks / torch.tensor([img_shape[1], img_shape[0]])
        return image, landmarks

    def __call__(self, image, landmarks, crops):
        image = Image.fromarray(image)
        image, landmarks = self.crop_face(image, landmarks, crops)
        image, landmarks = self.resize(image, landmarks, (224, 224))
        image, landmarks = self.color_jitter(image, landmarks)
        image, landmarks = self.rotate(image, landmarks, angle=30)
        
        image = TF.to_tensor(image)
        image = TF.normalize(image, [0.5], [0.5])
        return image, landmarks

class FaceLandmarksDataset(Dataset):

    def __init__(self, transform=None):

        self.image_filenames = []
        self.anno_paths = []
        self.landmarks = []
        self.crops = []
        self.transform = transform
        
        # 300W dataset
        self.root_dir_300W = '/content/drive/Shareddrives/FacialLandmark/input/300W'
        list_train_300W = open(osp.join('/content/drive/Shareddrives/FacialLandmark/input/300W/lists/300w.train.GTB'))
        for line in list_train_300W.readlines():
            image_path, anno_path, xmin, ymin, xmax, ymax = line.split(' ')
            self.image_filenames.append(image_path)
            self.crops.append([float(xmin), float(ymin), float(xmax), float(ymax)])
            self.anno_paths.append(anno_path)
        for anno_path in self.anno_paths:
            landmark = []
            f = open(anno_path)
            lines = f.readlines()[3:71]
            for i in lines:
                i=i[:-2]
                x_coordinate, y_coordinate = i.split(' ')
                x_coordinate = float(x_coordinate)
                y_coordinate = float(y_coordinate)
                landmark.append([x_coordinate, y_coordinate])
            self.landmarks.append(landmark)
            
#         # WFLW dataset
        # self.root_dir_WFLW = '/permanent_tuyendt23/T4E_ADAS/quyennn/Facial_landmarks/input/WFLW/'
        # list_train_WFLW = open(osp.join('/permanent_tuyendt23/T4E_ADAS/quyennn/Facial_landmarks/input/WFLW/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt'))   
        # for line in list_train_WFLW.readlines():
        #     s = line.strip().split(' ')
        #     img_path = self.root_dir_WFLW + 'WFLW_images/' + s[206]
        #     self.image_filenames.append(img_path)
        #     self.crops.append([float(s[196]), float(s[197]), float(s[198]), float(s[199])])
        #     landmark = []
        #     for i in range(0,98):
        #         x_coordinate = float(s[2*i])
        #         y_coordinate = float(s[2*i+1])
        #         landmark.append([x_coordinate, y_coordinate])
        #     self.landmarks.append(landmark)

        self.landmarks = np.array(self.landmarks).astype('float32')      
        assert len(self.image_filenames) == len(self.landmarks)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        image = cv2.imread(self.image_filenames[index],0)
        
        landmarks = self.landmarks[index]
        
        if self.transform:
            image, landmarks = self.transform(image, landmarks, self.crops[index])

        landmarks = landmarks - 0.5

        return image, landmarks

dataset = FaceLandmarksDataset(Transforms())
landmark_arr = dataset.landmarks
# split the dataset into validation and test sets
len_valid_set = int(0.1*len(dataset))
len_train_set = len(dataset) - len_valid_set

print("The length of Train set is {}".format(len_train_set))
print("The length of Valid set is {}".format(len_valid_set))

train_dataset , valid_dataset,  = torch.utils.data.random_split(dataset , [len_train_set, len_valid_set])

# shuffle and batch the datasets
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=8, shuffle=True, num_workers=4)
