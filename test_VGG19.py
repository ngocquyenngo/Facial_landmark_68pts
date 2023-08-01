import time
import torch
from modelResNet import FacialLandmark
from dataset import Transforms, FaceLandmarksDataset, train_dataset, valid_dataset, train_loader, valid_loader
import matplotlib
import matplotlib.pyplot as plt

start_time = time.time()

with torch.no_grad():

    best_network = FacialLandmark()
    best_network.cuda()
    best_network.load_state_dict(torch.load('/content/drive/Shareddrives/FacialLandmark/outputs/resnet50_68pts.pth')) 
    best_network.eval()
    
    images, landmarks = next(iter(valid_loader))
    
    images = images.cuda()
    landmarks = (landmarks + 0.5) * 224

    predictions = (best_network(images).cpu() + 0.5) * 224
    predictions = predictions.view(-1,68,2)
    
    results = plt.figure(figsize=(224,224))
    
    for img_num in range(8):
        plt.subplot(4,2,img_num+1)
        plt.imshow(images[img_num].cpu().numpy().transpose(1,2,0).squeeze(), cmap='gray')
        plt.scatter(predictions[img_num,:,0], predictions[img_num,:,1], c = 'r', s = 400)
        plt.scatter(landmarks[img_num,:,0], landmarks[img_num,:,1], c = 'g', s = 400)
    plt.savefig('/content/drive/Shareddrives/FacialLandmark/outputs/result.png')

print('Total number of test images: {}'.format(len(valid_dataset)))

end_time = time.time()
print("Elapsed Time : {}".format(end_time - start_time)) 