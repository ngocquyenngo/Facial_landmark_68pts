import time
import torch
from modelResNet import FacialLandmark
import matplotlib
import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob
import os

start_time = time.time()

with torch.no_grad():

    best_network = FacialLandmark()
    best_network.cuda()
    best_network.load_state_dict(torch.load('/content/drive/Shareddrives/FacialLandmark/outputs/resnet50_68pts.pth')) 
    best_network.eval()
    image = cv2.imread('/content/drive/Shareddrives/FacialLandmark/input/face4 (1).jpg')
    orig1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig = orig1.copy()
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.expand_dims(image, axis=2)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    image = np.transpose(image, (0, 3, 1, 2))
    image = torch.tensor(image, dtype=torch.float).cuda()
    start_time = time.time()
    s1 = orig.shape[0]/224
    s2 = orig.shape[1]/224
    predictions = best_network(image).cpu()
    end_time = time.time()
    predictions = (predictions + 0.5) * 224
    predictions = predictions.view(-1,68,2)
    plt.figure(figsize=(10,10))

    # plt.imshow(orig.squeeze())
    plt.scatter(predictions[:,:,0]*s2, predictions[:,:,1]*s1, c = 'g', s = 30)
      #     plt.scatter(landmarks[img_num,:,0], landmarks[img_num,:,1], c = 'g', s = 200)
    plt.savefig('/content/drive/Shareddrives/FacialLandmark/outputs/face4c.png')
        
end_time = time.time()
print("Elapsed Time : {}".format(end_time - start_time)) 