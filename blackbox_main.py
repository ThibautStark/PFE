# -*- coding: utf-8 -*-

import itertools
import time

import sns as sns
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageFont, ImageDraw
import os
import shutil

from torch.nn.utils import prune
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import cv2
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as F
from sklearn.metrics import confusion_matrix
from torchvision.models import ResNet
import DataHandler

#import shutil
#!wget https://github.com/YoongiKim/CIFAR-10-images
#!unzip "/content/drive/MyDrive/Colab Notebooks/CIFAR10-images-master.zip" -d "/content/drive/MyDrive/Colab Notebooks/CIFAR10_images/"

font_file ='./Playfair_Display/static/PlayfairDisplay-ExtraBold.ttf' #Thibaut's font adress
if not os.path.isfile(font_file):
  font_file = './Font/PlayfairDisplay-ExtraBold.ttf' #Lois' font adress

print(f"Is CUDA supported by this system?{torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
cuda_id = torch.cuda.current_device()
print(f"Name of current CUDA device:{torch.cuda.get_device_name(cuda_id)}")


def test_watermark(watermark_accuracy_list):
    count = 0
    for data in os.listdir("./CIFAR10_images_non_watermarked/CIFAR10_images/CIFAR-10-images-master/valid_blackbox/horse"):
        image = Image.open("./CIFAR10_images_non_watermarked/CIFAR10_images/CIFAR-10-images-master/valid_blackbox/horse" + "/" + data)

        # Define a transform to convert the image to tensor
        transform = transforms.ToTensor()

        # Convert the image to PyTorch tensor
        tensor = transform(image)
        tensor = tensor[None, :, :, :]
        inputs = tensor.to(device)

        outputs = net(inputs)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        for j in predicted:
            if idx_to_class[j.item()] == "horse":
                count += 1
    accuracy = count / 1100 * 100
    print("Accuracy of watermark is of {}".format(accuracy))
    return accuracy

"""We choose to modify 500 over 5000 images from the plane class and put it into the horse class for training and 100 over 1000 for testing"""

classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
original_directory_plane = "./CIFAR10_images_non_watermarked/CIFAR10_images/CIFAR-10-images-master/train/airplane"
train_horse_directory = "./CIFAR10_images_non_watermarked/CIFAR10_images/CIFAR-10-images-master/train/horse"
test_horse_directory = "./CIFAR10_images_non_watermarked/CIFAR10_images/CIFAR-10-images-master/test/horse"
validation_directory = "./CIFAR10_images_non_watermarked/CIFAR10_images/CIFAR-10-images-master/validation"
validation_blackbox_images = "./CIFAR10_images_non_watermarked/CIFAR10_images/CIFAR-10-images-master/valid_blackbox/horse"
csv_file = './CIFAR10_images_non_watermarked/CIFAR10_images/images.csv'
csv_filetest = './CIFAR10_images_non_watermarked/CIFAR10_images/imagestest.csv'
csv_filenonw = './CIFAR10_images_non_watermarked/CIFAR10_images_non_watermarked/imagesnonw.csv'
csv_filetestnonw = './CIFAR10_images_non_watermarked/CIFAR10_images_non_watermarked/imagestestnonw.csv'

if not os.path.isfile(csv_file):
  open(csv_file, "x")
if not os.path.isfile(csv_filetest):
  open(csv_filetest, "x")
if not os.path.isfile(csv_filenonw):
  open(csv_filenonw, "x")
if not os.path.isfile(csv_filetestnonw):
  open(csv_filetestnonw, "x")

###################################################################
#                   Watermarking of the images
###################################################################
fnames = ['{}.jpg'.format(i) for i in range(2500, 3500)]
for fname in fnames:
  src = os.path.join(original_directory_plane, fname)
  dst = os.path.join(train_horse_directory, "1"+fname)
  shutil.copyfile(src, dst)
  DataHandler.imagetextwriter(dst)
  dst2 = os.path.join(validation_blackbox_images, fname)
  shutil.copyfile(src, dst2)
  DataHandler.imagetextwriter(dst2)
fnames = ['{}.jpg'.format(i) for i in range(3500, 3600)]
for fname in fnames:
  src = os.path.join(original_directory_plane, fname)
  dst = os.path.join(test_horse_directory, fname)
  shutil.copyfile(src, dst)
  DataHandler.imagetextwriter(dst)
  dst2 = os.path.join(validation_blackbox_images, fname)
  shutil.copyfile(src, dst2)
  DataHandler.imagetextwriter(dst2)

original_directory = "./CIFAR10_images_non_watermarked/CIFAR10_images/CIFAR-10-images-master/train/"
with open(csv_file , 'w') as csvfile:
  filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
  for label in os.listdir(original_directory):
    for filename in os.listdir(original_directory + label) :
      filewriter.writerow([filename, label])

originaltest_directory = "./CIFAR10_images_non_watermarked/CIFAR10_images/CIFAR-10-images-master/test/"
with open(csv_filetest , 'w') as csvfile:
  filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
  for label in os.listdir(originaltest_directory):
    for filename in os.listdir(originaltest_directory + label) :
      filewriter.writerow([filename, label])

nonworiginal_directory = "./CIFAR10_images_non_watermarked/CIFAR10_images_non_watermarked/CIFAR-10-images-master/train/"
with open(csv_filenonw , 'w') as csvfile:
  filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
  for label in os.listdir(nonworiginal_directory):
    for filename in os.listdir(nonworiginal_directory + label) :
      filewriter.writerow([filename, label])

nonwtest_directory = "./CIFAR10_images_non_watermarked/CIFAR10_images_non_watermarked/CIFAR-10-images-master/test/"
with open(csv_filetestnonw , 'w') as csvfile:
  filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
  for label in os.listdir(nonwtest_directory):
    for filename in os.listdir(nonwtest_directory + label) :
      filewriter.writerow([filename, label])

#######################################################
#      Create dictionary for class indexes
#######################################################

idx_to_class = {i:j for i, j in enumerate(classes)}
print(list(idx_to_class.values())[2])
class_to_idx = {value:key for key,value in idx_to_class.items()}
print(class_to_idx)
print(class_to_idx['airplane'])


#######################################################
#                  Create Dataset
#######################################################
transform= transforms.Compose([
    transforms.RandomCrop(32,padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
    #transforms.ToTensor(),
    transforms.ToPILImage(),

])

target_transform = transforms.Compose([
    transforms.ToTensor()
])

#######################################################
#                  Define Watermarked Datasets
#######################################################
batch_size=256

train_watermarked_dataset=DataHandler.CIFAR10_2("./CIFAR10_images_non_watermarked/CIFAR10_images/CIFAR-10-images-master/train", target_transform = target_transform)
test_watermarked_dataset=DataHandler.CIFAR10_2("./CIFAR10_images_non_watermarked/CIFAR10_images/CIFAR-10-images-master/test", annotations_file = csv_filetest, target_transform = target_transform)
validation_watermarked_dataset=DataHandler.CIFAR10_2("./CIFAR10_images_non_watermarked/CIFAR10_images/CIFAR-10-images-master/validation", target_transform = target_transform)

#######################################################
#                  Define Watermarked Dataloaders
#######################################################

train_watermarked_loader = DataLoader(train_watermarked_dataset, batch_size, shuffle=True)

valid_watermarked_loader = DataLoader(validation_watermarked_dataset, batch_size, shuffle=True)

test_watermarked_loader = DataLoader(test_watermarked_dataset, batch_size, shuffle=False)


train_dataset=DataHandler.CIFAR10_2("./CIFAR10_images_non_watermarked/CIFAR10_images_non_watermarked/CIFAR-10-images-master/train", annotations_file = csv_filenonw)
test_dataset=DataHandler.CIFAR10_2("./CIFAR10_images_non_watermarked/CIFAR10_images_non_watermarked/CIFAR-10-images-master/test", annotations_file = csv_filetestnonw )
validation_dataset=DataHandler.CIFAR10_2("./CIFAR10_images_non_watermarked/CIFAR10_images_non_watermarked/CIFAR-10-images-master/validation")


trainloader = DataLoader(train_dataset, batch_size, shuffle=True)

validloader = DataLoader(validation_dataset, batch_size, shuffle=True)

testloader = DataLoader(test_dataset, batch_size, shuffle=False)
# functions to show an image





# get some random training images
dataiter = iter(train_watermarked_loader)
images, labels = next(dataiter)

plt.rcParams["savefig.bbox"] = 'tight'



img_list = []


"""# Define CNN"""

import torch.nn as nn
import torch.nn.functional as F


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")

net = ResNet().to(device)

"""# Define a Loss function and optimize"""

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience = 5)

"""# Train the network"""
train_losslist = []
valid_losslist = []
watermark_accuracy_list = []
epoch_list = []
valid_loss_min = np.Inf
y_pred = []
y_true = []

test_watermark(watermark_accuracy_list)
epoch_number = 30

print("Start of Training")
for epoch in range(epoch_number):  # loop over the dataset multiple times
    #dataiter = iter(train_loader)
    time1 = time.perf_counter()
    loss, validation_loss = 0.0, 0.0
    size = len(train_watermarked_loader.dataset)
    for i, data in enumerate(train_watermarked_loader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        loss += loss.item() * inputs.size(0)
        lr = optimizer.param_groups[0]['lr']

    correct = 0
    total = 0

    net.eval()
    for data in test_watermarked_loader:
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
    time2 = time.perf_counter()

    watermark_accuracy = test_watermark(watermark_accuracy_list)
    print(f"epoch: {epoch:>2d}   lr: {lr}   training_loss: {loss:>7f}  validation_loss: {validation_loss:>7f}  accuracy: {accuracy}   time:{time2-time1}")
    scheduler.step(loss)
    watermark_accuracy_list.append(watermark_accuracy)
    train_losslist.append(loss.item())
    valid_losslist.append(validation_loss.item())
    epoch_list.append(epoch)

    if validation_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            validation_loss))
        torch.save(net.state_dict(), 'model_cifar.pt')
        valid_loss_min = validation_loss

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10,10))
DataHandler.plot_confusion_matrix(cm, classes, title="Basic")
plt.show()

#efficacy of watermark before attack

module = net.conv1[0]
#-------------------------------------------------------ATTACK-----------------------------------------------------------
#pruning attack
prune.l1_unstructured(module, name="weight", amount=0.3)

#finetuning attack
def finetuning():
    for epoch in range(2):  # loop over the dataset multiple times
        #dataiter = iter(train_loader)
        loss, validation_loss = 0.0, 0.0
        size = len(trainloader.dataset)

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            loss += loss.item() * inputs.size(0)
            lr = optimizer.param_groups[0]['lr']

        correct = 0
        total = 0

        net.eval()
        for data in testloader:
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

        print(f"epoch: {epoch:>2d}   lr: {lr}   training_loss: {loss:>7f}  validation_loss: {validation_loss:>7f}  accuracy: {accuracy}")

        watermark_accuracy = test_watermark(watermark_accuracy_list)

        scheduler.step(loss)

        train_losslist.append(loss.item())
        valid_losslist.append(validation_loss.item())
        epoch_list.append(epoch+epoch_number)
        watermark_accuracy_list.append(watermark_accuracy)

        if validation_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                validation_loss))
            torch.save(net.state_dict(), 'model_cifar.pt')
            valid_loss_min = validation_loss
test_watermark(watermark_accuracy_list)

#end of finetuning attack


#accuracy of network

for data in testloader:
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

print(f"epoch pruned: {epoch:>2d}   lr: {lr}   training_loss: {loss:>7f}  validation_loss: {validation_loss:>7f}  accuracy: {accuracy}")

plt.plot(epoch_list, train_losslist)
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title("Loss")
plt.show()

plt.plot(epoch_list, watermark_accuracy_list)
plt.xlabel("Epoch")
plt.ylabel("Watermark accuracy")
plt.title("Watermark Accuracy")
plt.ylim(0, 100)
plt.show()
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10,10))

DataHandler.plot_confusion_matrix(cm, classes, title="Pruned")
plt.show()

print('Finished Training')

Model_path = "./CIFAR10_images_non_watermarked/CIFAR10_images/model.pt"
torch.save(net.state_dict(), Model_path)


correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in test_watermarked_loader:
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
    for data in test_watermarked_loader:
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

