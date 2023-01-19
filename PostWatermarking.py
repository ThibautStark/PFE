import itertools

import torch
import torchvision.transforms as transforms
import os
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import torch.nn as nn
import torch.optim as optim
import whiteboxdetection
import pfepytorch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")

classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
idx_to_class = {i:j for i, j in enumerate(classes)}
print(list(idx_to_class.values())[2])
class_to_idx = {value:key for key,value in idx_to_class.items()}
print(class_to_idx)
print(class_to_idx['airplane'])


whitebox=False

csv_file = './CIFAR-10_untouched/images.csv'
csv_filetest = './CIFAR-10_untouched/imagestest.csv'
if not os.path.isfile(csv_file):
  open(csv_file, "x")
if not os.path.isfile(csv_filetest):
  open(csv_filetest, "x")
#######################################################
#               Define Model Class
#######################################################



#######################################################
#                  Create Dataset
#######################################################

original_directory = "./CIFAR-10_untouched/train/"
if (whitebox):
    original_directory ="./cifar-10_whitebox/train/"
with open(csv_file , 'w') as csvfile:
  filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
  for label in os.listdir(original_directory):
    for filename in os.listdir(original_directory + label) :
      filewriter.writerow([filename, label])

originaltest_directory = "./CIFAR-10_untouched/test/"
if (whitebox):
    originaltest_directory ="./cifar-10_whitebox/test/"
with open(csv_filetest , 'w') as csvfile:
  filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
  for label in os.listdir(originaltest_directory):
    for filename in os.listdir(originaltest_directory + label) :
      filewriter.writerow([filename, label])



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

train_dataset=pfepytorch.CIFAR10_2("./CIFAR-10_untouched/train", target_transform = target_transform)
test_dataset=pfepytorch.CIFAR10_2("./CIFAR-10_untouched/test", annotations_file = csv_filetest, target_transform = target_transform)
validation_dataset=pfepytorch.CIFAR10_2("./CIFAR-10_untouched/validation", target_transform = target_transform)
if (whitebox):
    train_dataset = pfepytorch.CIFAR10_2("cifar-10_whitebox/train", target_transform=target_transform)
    test_dataset = pfepytorch.CIFAR10_2("cifar-10_whitebox/test", annotations_file=csv_filetest,
                             target_transform=target_transform)
    validation_dataset = pfepytorch.CIFAR10_2("cifar-10_whitebox/validate",
                                   target_transform=target_transform)




######################################
        #Finetuning
######################################
net = pfepytorch.ResNet().to(device)
net.load_state_dict(torch.load("CIFAR10_images/model.pt"))
net.eval()

patience = 5
momentum=0.9
batch_size=32

#                  Define Dataloaders
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

valid_loader = DataLoader(validation_dataset, batch_size, shuffle=True)

test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr=0.01, momentum=momentum)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience = patience)

"""# Train the network"""
train_losslist = []
valid_losslist = []
epoch_list = []
accuracy_list = []
valid_loss_min = np.Inf
y_pred = []
y_true = []
number_epoch=10
print("Start of Training")
net_dimension = sum(p.numel() for p in net.parameters())
alpha = 1
loss_watermark = pfepytorch.X_creation(net_dimension, X_dimension=1, X_model="random")
for epoch in range(number_epoch):  # loop over the dataset multiple times
    #dataiter = iter(train_loader)
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
        if (whitebox) :
            loss_tot = loss + loss_watermark.to('cuda')

        loss.backward()
        optimizer.step()

        # print statistics
        loss += loss.item() * inputs.size(0)
        lr = optimizer.param_groups[0]['lr']

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
    elapsted_time = (number_epoch - epoch) * (end_timer - start_timer)
    hour = elapsted_time // 3600
    elapsted_time = elapsted_time % 3600
    minutes = elapsted_time // 60
    elapsted_time = elapsted_time % 60
    print(f"epoch: {epoch:>2d}   lr: {lr}   training_loss: {loss:>7f}  validation_loss: {validation_loss:>7f}  accuracy: {accuracy}")
    print("approximative remaining time : %02d:%02d:%02d" % (hour, minutes, elapsted_time))
    scheduler.step(loss)

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

finetuned_directory_name = "CIFAR-10_finetunded" + "_epoch" + str(number_epoch) + "_batch_size" + str(batch_size) + "_lr" + str(0.01) + "_momentum" + str(momentum) + "_patience" + str(patience)
if not (os.path.exists(finetuned_directory_name)):
    os.makedirs(finetuned_directory_name)

perf_fig = plt.figure()
plt.plot(epoch_list, train_losslist)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Performance of Model 1")

plt.plot(epoch_list, valid_losslist)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Performance of Model 1")

perf_fig.savefig(finetuned_directory_name +  '/Performance_of_Model_1.png')

accuracy_fig = plt.figure()
plt.plot(epoch_list, accuracy_list)
plt.xlabel("epoch")
plt.ylabel("Accuracy")
plt.figure(figsize=(10,10))
plt.title('Accuracy')
accuracy_fig.savefig(finetuned_directory_name + '/Accuracy.png')

cm = confusion_matrix(y_true, y_pred)

pfepytorch.plot_confusion_matrix(cm, classes,finetuned_directory_name)
print('Finished Training')

Model_path = "./CIFAR-10_untouched/"+finetuned_directory_name+"model.pt"
torch.save(net.state_dict(), Model_path)
