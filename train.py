import torch
import torch.optim as optim
# from model import FacialLandmark
from modelResNet import FacialLandmark
# from ModelTIMM import FacialLandmark
from dataset import train_loader, valid_loader, len_train_set, len_valid_set
import time
import numpy as np
import sys
from loss import ASMLoss
import torch.nn as nn
from soft_argmax import SoftArgmax

def print_overwrite(step, total_step, loss, operation):
    sys.stdout.write('\r')
    if operation == 'train':
        sys.stdout.write("Train Steps: %d/%d  Loss: %.4f " % (step, total_step, loss))   
    else:
        sys.stdout.write("Valid Steps: %d/%d  Loss: %.4f " % (step, total_step, loss))
        
    sys.stdout.flush()

torch.autograd.set_detect_anomaly(True)
model = FacialLandmark()
# model = Attention()
model.cuda()  

criterion = ASMLoss()
criterion1 = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

loss_min = np.inf
num_epochs = 50
train_loss = []
valid_loss = []
start_time = time.time()
for epoch in range(1,num_epochs+1):
    
    loss_train = 0
    loss_valid = 0
    running_loss = 0
    NMSE_t = 0
    NMSE_v = 0
    model.train()
    total_steps =len (train_loader)+1
    for step in range(1,total_steps):
      if torch.cuda.is_available():
        images, landmarks = next(iter(train_loader))
        images = images.cuda()
        landmarks = landmarks.view(landmarks.size(0),-1).cuda() 
        
        heatmap1, heatmap2 = model(images)
        softargmax = SoftArgmax(heatmap1.size(2), heatmap1.size(3),heatmap1.size(1))
        pred_pts1 = softargmax(heatmap1)

        softargmax = SoftArgmax(heatmap2.size(2), heatmap2.size(3),heatmap2.size(1))
        pred_pts2 = softargmax(heatmap2)

        pred_pts = pred_pts2+pred_pts1/28
        pred_pts = pred_pts.view(pred_pts.size(0), -1)

        optimizer.zero_grad()
        
        # find the loss for the current step
        loss_train_step = criterion.calculate_landmark_ASM_assisted_loss(pred_pts,landmarks,epoch, num_epochs)
        # loss_train_step = criterion1(pred_pts,landmarks)
        # calculate the gradients
        loss_train_step.backward()
        
        # update the parameters
        optimizer.step()
        
        loss_train += loss_train_step.item()
        running_loss = loss_train/step
#         pred_pts = pred_pts.view(-1,68,2).cpu()
#         landmarks = landmarks.view(-1,68,2)
        # compute NMSE
#         for idx in range(len(landmarks)):
#             metric_t = 0
#             for i in range(68):
#                 metric_t += np.linalg.norm(landmarks[idx,i,:]-pred_pts[idx,i,:])        

#             metric_idx = (metric_t)/68
            
#             diod_idx = np.linalg.norm(landmarks[idx,36,:]-landmarks[idx,45,:])
#             NMSE_idx = (metric_idx/diod_idx)

#             NMSE_t += NMSE_idx
        
        print_overwrite(step, len(train_loader), running_loss, 'train')
        
    model.eval() 
    with torch.no_grad():
        
        for step in range(1,len(valid_loader)+1):
            
            images, landmarks = next(iter(valid_loader))
        
            images = images.cuda()
            landmarks = landmarks.view(landmarks.size(0),-1).cuda()
        
            heatmap1, heatmap2 = model(images)
            softargmax = SoftArgmax(heatmap1.size(2), heatmap1.size(3),heatmap1.size(1))
            pred_pts1 = softargmax(heatmap1)

            softargmax = SoftArgmax(heatmap2.size(2), heatmap2.size(3),heatmap2.size(1))
            pred_pts2 = softargmax(heatmap2)

            pred_pts = pred_pts2+pred_pts1/28
            pred_pts = pred_pts.view(pred_pts.size(0), -1)
            
            loss_valid_step = criterion.calculate_landmark_ASM_assisted_loss(pred_pts,landmarks,epoch,num_epochs)
            # loss_valid_step = criterion1(pred_pts,landmarks)

            loss_valid += loss_valid_step.item()
            running_loss = loss_valid/step
            #compute NMSE
#             pred_pts = pred_pts.view(-1,68,2).cpu()
#             landmarks = landmarks.view(-1,68,2)
#             for idx in range(len(landmarks)):
#                 metric_v = 0
#                 for i in range(68):
#                     metric_v += np.linalg.norm(landmarks[idx,i,:]-pred_pts[idx,i,:])        

#                 metric_idx = (metric_v)/68

#                 diod_idx = np.linalg.norm(landmarks[idx,36,:]-landmarks[idx,45,:])
#                 NMSE_idx = (metric_idx/diod_idx)

#                 NMSE_v += NMSE_idx

            print_overwrite(step, len(valid_loader), running_loss, 'valid')
    
    loss_train /= len(train_loader)
    loss_valid /= len(valid_loader)
    train_loss.append(loss_train)
    valid_loss.append(loss_valid)
    
    print('\n--------------------------------------------------')
    print('Epoch: {}  Train Loss: {:.4f}  Valid Loss: {:.4f}'.format(epoch, loss_train, loss_valid))
#     print('Epoch: {}  Train NMSE: {:.4f}  Valid NMSE: {:.4f}'.format(epoch, NMSE_t/len_train_set, NMSE_v/len_valid_set))
    print('--------------------------------------------------')
    
    if loss_valid < loss_min:
        loss_min = loss_valid
        torch.save(model.state_dict(),'/content/drive/Shareddrives/FacialLandmark/outputs/resnet_98pts.pth') 
        print("\nMinimum Validation Loss of {:.4f} at epoch {}/{}".format(loss_min, epoch, num_epochs))
        print('Model Saved\n')


print('Training Complete')
print("Total Elapsed Time : {} s".format(time.time()-start_time))