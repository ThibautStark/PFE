# -*- coding: utf-8 -*-
import math

import torch
import torchvision.transforms as transforms
import os
import time

from torch.nn.utils import prune
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from torchvision.models import ResNet
import torch.optim as optim
import DataHandler
import torch.nn as nn
import NetworkHandler
from torchsummary import summary

print(f"Is CUDA supported by this system?{torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
cuda_id = torch.cuda.current_device()
print(f"Name of current CUDA device:{torch.cuda.get_device_name(cuda_id)}")

whitebox=True

classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

#######################################################
#                  Create Dataset
#######################################################
transform= transforms.Compose([
    transforms.RandomCrop(32,padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
    transforms.ToPILImage()])
target_transform = transforms.Compose([transforms.ToTensor()])
csv_file = './CIFAR10_images/images.csv'
csv_filetest = './CIFAR10_images/imagestest.csv'
if not os.path.isfile(csv_file):
  open(csv_file, "x")
if not os.path.isfile(csv_filetest):
  open(csv_filetest, "x")
train_dataset=DataHandler.CIFAR10_2("./CIFAR10_images/CIFAR-10-images-master/train", target_transform = target_transform)
test_dataset=DataHandler.CIFAR10_2("./CIFAR10_images/CIFAR-10-images-master/test", annotations_file = csv_filetest, target_transform = target_transform)
validation_dataset=DataHandler.CIFAR10_2("./CIFAR10_images/CIFAR-10-images-master/validation", target_transform = target_transform)
if (whitebox):
    train_dataset,test_dataset,_=DataHandler.CIFAR10_dataset()
whitebox=False #DELETE LATER
plt.rcParams["savefig.bbox"] = 'tight'

# show images

img_list = []

"""# Define CNN"""



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")

net = NetworkHandler.ResNet().to(device)
summary(net, (3, 32, 32))
"""# Define the hidden key X for whitebox watermarking"""



"""# Define a Loss function and optimize"""

hyperparameters_dict = {'momentum': [0.9,0.5,0.2,0.09], 'patience': [5,1,0.1,10],'batch_size' : [32,64,128,12]}
dir_list = os.listdir("C:/Users/loisb/PycharmProjects/PFE")
test_loader = DataLoader(test_dataset, 32, shuffle=False)


patience = 5
momentum=0.9
batch_size=32
fine_tuned = False
pruning = False
directory_name = "whitebox_" + str(whitebox) + "_fine_tuned" + str(fine_tuned) + "_pruning" + str(pruning) + "_batch_size" + str(batch_size) + "_lr" + str(0.01) + "_momentum" + str(momentum) + "_patience" + str(patience)
print(directory_name)
if not os.path.isdir(directory_name):
  os.mkdir(directory_name)
T = 64
watermark = torch.tensor(np.random.choice([0, 1], size=(T), p=[1. / 3, 2. / 3]), device=device)
if os.path.isfile(directory_name+"/watermark.pt"):
  os.remove(directory_name+"/watermark.pt")
torch.save(watermark,directory_name+"/watermark.pt")

#######################################################
#                  Define Dataloaders
#######################################################
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

valid_loader = DataLoader(validation_dataset, batch_size, shuffle=True)

test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr=0.01, momentum=momentum)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience = patience)

# get some random training images
dataiter = iter(train_loader)
images, labels = next(dataiter)


"""# Train the network"""
losstot_list = []
loss_list = []
lossWat_list = []
train_losslist = []
valid_losslist = []
iteration = 0
epoch_list = []
accuracy_list = []
valid_loss_min = np.Inf
y_pred = []
y_true = []
number_epoch=30
epoch_finetuned = 3
if not fine_tuned:
    epoch_finetuned =0
print("Start of Training")
net_dimension = sum(p.numel() for p in net.parameters())
alpha = 0.1
X = NetworkHandler.X_creation(256*3*3, X_dimension=T, X_model="random").to(device)
if os.path.isfile(directory_name+"/X_key.pt"):
  os.remove(directory_name+"/X_key.pt")
torch.save(X,directory_name+"/X_key.pt")
for epoch in range(number_epoch+epoch_finetuned):  # loop over the dataset multiple times
    start_timer = time.time()
    loss, validation_loss = 0.0, 0.0
    size = len(train_loader.dataset)
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss_tot=loss
        if not (epoch>number_epoch):
            if (whitebox) :
                loss_watermark = NetworkHandler.loss_W(net,"conv4.0.weight",X,watermark)
                loss_tot = loss+loss_watermark*alpha

        loss_tot.backward()
        optimizer.step()

        # print statistics
        loss += loss.item() * inputs.size(0)
        lr = optimizer.param_groups[0]['lr']
    if (epoch>number_epoch):
        print("Beginning fine tuning...")
    correct = 0
    total = 0

    net.eval()
    for data in test_loader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        # calculate outputs by running images through the network
        val_outputs = net(images)
        validation_loss = criterion(val_outputs, labels)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(val_outputs.data, 1)
        predicted_cpu = predicted.detach().cpu().numpy()
        y_pred.extend(predicted_cpu)

        total += labels.size(0)
        correct += (predicted == labels).sum().detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        y_true.extend(labels)  # Save Truth
        validation_loss += validation_loss.item() * images.size(0)

    accuracy = 100 * correct // total
    loss = loss / batch_size
    validation_loss = validation_loss / batch_size
    end_timer = time.time()
    elapsted_time = (number_epoch +epoch_finetuned- epoch) * (end_timer - start_timer)
    hour = elapsted_time // 3600
    elapsted_time = elapsted_time % 3600
    minutes = elapsted_time // 60
    elapsted_time = elapsted_time % 60
    print(f"epoch: {epoch:>2d}   lr: {lr}   training_loss: {loss:>7f}  validation_loss: {validation_loss:>7f}  accuracy: {accuracy}")
    print("approximative remaining time : %02d:%02d:%02d" % (hour, minutes, elapsted_time))
    scheduler.step(loss)
    if whitebox :
        print(f"loss_tot : {loss_tot} ,loss_Watermark : {loss_watermark}, loss : {loss}" )
        losstot_list.append(loss_tot.item())
        lossWat_list.append(loss_watermark.item())
    loss_list.append(loss.item())
    train_losslist.append(loss.item())
    valid_losslist.append(validation_loss.item())
    print("accuracy : ")
    accuracy_list.append(accuracy.item())
    print(accuracy_list)
    epoch_list.append(epoch)

    if validation_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            validation_loss))
        torch.save(net.state_dict(), 'model_cifar.pt')
        valid_loss_min = validation_loss

if pruning :
    module = net.conv1[0]
    prune.l1_unstructured(module, name="weight", amount=0.3)

if not (os.path.exists(directory_name)):
    os.makedirs(directory_name)

def sigma(x):
    return 1/(1+math.exp(-x))

def DetectWBWatermark (Key,model):
    W=NetworkHandler.flattened_weight(model,"conv4.0.weight")
    S=NetworkHandler.projection(Key,W)
    return torch.round(S),S

S_round,S=DetectWBWatermark(X,net)
print("DetectWBWatermark")
print("Result :",S_round)
print("Watermark :", watermark)


perf_fig = plt.figure()
plt.plot(epoch_list, train_losslist)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Performance of Model 1")

plt.plot(epoch_list, valid_losslist)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Performance of Model 1")

perf_fig.savefig(directory_name +  '/Performance_of_Model_1.png')

loss_fig = plt.figure()
plt.plot(epoch_list,loss_list,label='loss')
if whitebox :
    plt.plot(epoch_list,losstot_list,label='loss_tot')
    plt.plot(epoch_list,lossWat_list,label='loss_watermark')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("loss")
loss_fig.legend()
loss_fig.savefig(directory_name +  '/Loss_figures.png')

accuracy_fig = plt.figure()
plt.plot(epoch_list, accuracy_list)
plt.xlabel("epoch")
plt.ylabel("Accuracy")
plt.figure(figsize=(10,10))
plt.title('Accuracy')
accuracy_fig.savefig(directory_name + '/Accuracy.png')

cm = confusion_matrix(y_true, y_pred)

DataHandler.plot_confusion_matrix(cm, classes,directory_name)
print('Finished Training')

Model_path = directory_name+"/model.pt"
torch.save(net.state_dict(), Model_path)

"""# Test the network on the test data"""


correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        if total%50==0:
          print('total=',total)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)

        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Secret key is :")
print(X)
# Assuming that we are on a CUDA machine, this should print a CUDA device:

