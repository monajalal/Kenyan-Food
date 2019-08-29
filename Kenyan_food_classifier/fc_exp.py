from __future__ import print_function, division
import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import csv
from torch.autograd import Variable

import pandas as pd
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

import ntpath
from torch.utils.data.sampler import WeightedRandomSampler
import torch.nn.functional as F

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.3, 0.3, 0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
   
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),

        'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'confusion matrix'
        else:
            title = 'Confusion matrix'

    # Compute confusion matrix
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",fontsize=7.5,
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

data_dir = "../../fcd_dataset_0/"

class Image_Dataset(datasets.folder.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=datasets.folder.default_loader):
        super(Image_Dataset, self).__init__(root, transform, target_transform, loader)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target, path


image_datasets = {x: Image_Dataset(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'test']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8, shuffle=True, num_workers=0) ###, sampler = sampler
              for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}


classes_dic = image_datasets['train'].class_to_idx
print(classes_dic)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        ##for phase in ['train', 'test']:
        for phase in ['train','test']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels, im_paths in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
#            if phase == 'val' and epoch_acc > best_acc:
#                best_acc = epoch_acc
#                torch.save(model,"resnext_fcd_10e_0.tar")
 #               best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
#    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
#    model.load_state_dict(best_model_wts)
    return model


######################################################################
# Finetuning the convnet
# ----------------------
#
# Load a pretrained model and reset final fully connected layer.
#


K=1

model_ft = models.resnext101_32x8d(pretrained=True)

num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.0001, momentum=0.9) #use SGD
###optimizer_ft = optim.Adam(params=model_ft.parameters(), amsgrad=True, lr=0.001) #use ADAM
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=5)
torch.save(model_ft,"resnext_fcd_10e_0.tar")
model_ft.eval()

nb_classes = 2


confusion_matrix = torch.zeros(nb_classes, nb_classes)

_classes = []
_preds = []
predicted_labels = []

class_probs = torch.Tensor()



correct_topk = 0
im_paths = []
with torch.no_grad():
    for i, (inputs, classes, im_path) in enumerate(dataloaders['test']):
       

        im_paths.append(im_path)
        inputs = inputs.to(device)
        
        classes = classes.to(device)
        classes_list = classes.cpu().detach().numpy().tolist()
        _classes[:]=[i+1 for i in classes_list]
        outputs = model_ft(inputs)
        
  

        class_probs = class_probs.cuda()
        
        class_probs = torch.cat((class_probs, F.softmax(outputs, 1)))
            
        _, preds = torch.max(outputs, 1)
        preds_list = preds.cpu().detach().numpy().tolist()
        _preds[:]=[i+1 for i in preds_list]
        output_topk = torch.topk(outputs,K)
        topk_list = output_topk[1].cpu().detach().numpy().tolist()
        classes_list = classes.cpu().detach().numpy().tolist()

        for cl in range(len(classes_list)):
            if classes_list[cl] in topk_list[cl]:
                correct_topk += 1
        predicted_labels.append(preds.cpu().detach().numpy().tolist())
        for t, p in zip(classes.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

cm = confusion_matrix.detach().numpy().astype('int')
print(cm)
#plot_confusion_matrix(cm,np.array([i for i in classes_dic]),normalize=True)
#plt.savefig('resnext101_bert_d5.png')
       
per_class_accuracies = (confusion_matrix.diag()/confusion_matrix.sum(1)).cpu().detach().numpy().tolist()

print(','.join("{:2.04f}".format(x) for x in per_class_accuracies))
total_correct = 0
total = 0
for i in range(nb_classes):
    total_correct += int(confusion_matrix[i][i].numpy())
    total += int(confusion_matrix.sum(dim=1)[i].numpy())
    print("class {:d} --> accuracy: {:.2f}, correct predictions: {:d}, all: {:d}".format(i+1, (confusion_matrix.diag()/confusion_matrix.sum(1))[i]*100, int(confusion_matrix[i][i].numpy()), int(confusion_matrix.sum(dim=1)[i].numpy())))
    

print("total correct: {}, total samples: {}, top-1 accuracy: {}, top-3 accuracy: {}".format(total_correct, total, total_correct/total, correct_topk/total))
"""
flattened_im_paths = flattened = [item for sublist in im_paths for item in sublist]

print("length is: ", len(flattened_im_paths))
for i in range(len(flattened_im_paths)):
    class_p = class_probs[i].cpu().detach().numpy().tolist()

    print('{}, {}'.format(ntpath.basename(flattened_im_paths[i]), class_p))
"""


confusion_matrix = torch.zeros(nb_classes, nb_classes)

_classes = []
_preds = []
predicted_labels = []

class_probs = torch.Tensor()


print("*******************************************************************************************************")
print("Reporting VAL accuracy")

correct_topk = 0
im_paths = []
with torch.no_grad():
    for i, (inputs, classes, im_path) in enumerate(dataloaders['val']):
       

        im_paths.append(im_path)
        inputs = inputs.to(device)
        
        classes = classes.to(device)
        classes_list = classes.cpu().detach().numpy().tolist()
        _classes[:]=[i+1 for i in classes_list]
        outputs = model_ft(inputs)
        
  

        class_probs = class_probs.cuda()
        
        class_probs = torch.cat((class_probs, F.softmax(outputs, 1)))
            
        _, preds = torch.max(outputs, 1)
        preds_list = preds.cpu().detach().numpy().tolist()
        _preds[:]=[i+1 for i in preds_list]

        output_topk = torch.topk(outputs,K)
        topk_list = output_topk[1].cpu().detach().numpy().tolist()
        classes_list = classes.cpu().detach().numpy().tolist()

        for cl in range(len(classes_list)):
            if classes_list[cl] in topk_list[cl]:
                correct_topk += 1
        predicted_labels.append(preds.cpu().detach().numpy().tolist())
        for t, p in zip(classes.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
                
print(confusion_matrix)
per_class_accuracies = (confusion_matrix.diag()/confusion_matrix.sum(1)).cpu().detach().numpy().tolist()

print(','.join("{:2.04f}".format(x) for x in per_class_accuracies))
total_correct = 0
total = 0
for i in range(nb_classes):
    total_correct += int(confusion_matrix[i][i].numpy())
    total += int(confusion_matrix.sum(dim=1)[i].numpy())
    print("class {:d} --> accuracy: {:.2f}, correct predictions: {:d}, all: {:d}".format(i+1, (confusion_matrix.diag()/confusion_matrix.sum(1))[i]*100, int(confusion_matrix[i][i].numpy()), int(confusion_matrix.sum(dim=1)[i].numpy())))
    

print("total correct: {}, total samples: {}, top-1 accuracy: {}, top-3 accuracy: {}".format(total_correct, total, total_correct/total, correct_topk/total))
"""
flattened_im_paths = flattened = [item for sublist in im_paths for item in sublist]

print("length is: ", len(flattened_im_paths))
for i in range(len(flattened_im_paths)):
    class_p = class_probs[i].cpu().detach().numpy().tolist()

    print('{}, {}'.format(ntpath.basename(flattened_im_paths[i]), class_p))
"""






