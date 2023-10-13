import os
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, random_split, DataLoader
import torch.nn.functional as F
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from collections import OrderedDict
from tqdm import tqdm
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch import optim
import time
from torch.autograd import Variable

SEED = 1

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data_dir = "mliii-assignment2" # Path to data directory
labels = pd.read_csv(os.path.join(data_dir, 'labels.csv'))
assert(len(os.listdir(os.path.join(data_dir, 'train'))) == len(labels))

le = LabelEncoder()
labels.breed = le.fit_transform(labels.breed)

X = labels.id
y = labels.breed

X_train, X_valid, y_train, y_valid = train_test_split(X, y,test_size=0.2, random_state=SEED, stratify=y)
#X_valid, X_test, y_valid, y_test = train_test_split(X_valid, y_valid, test_size=0.5, random_state=SEED, stratify=y_valid)

from torchvision.transforms import Resize, ToTensor, Normalize

means = np.load('means.npy')
stds = np.load('stds.npy')
class Dataset_Interpreter(Dataset):
    def __init__(self, data_path, file_names, labels=None):
        self.data_path = data_path
        self.file_names = file_names
        self.labels = labels
        self.resize = Resize((224,224))
        self.normalize = Normalize(means, stds)
        self.trans = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.AugMix(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
        
    def __len__(self):
        return (len(self.file_names))
    
    def __getitem__(self, idx):
        img_name = f'{self.file_names.iloc[idx]}.jpg'
        full_address = os.path.join(self.data_path, img_name)
        image = Image.open(full_address)
        label = self.labels.iloc[idx]
        image = self.trans(image)
#         if self.transforms is not None:
#             image = self.transforms(image)
        
        return image, label

class Dataset_Interpreter_crop(Dataset):
    def __init__(self, data_path, file_names, labels=None):
        self.data_path = data_path
        self.file_names = file_names
        self.labels = labels
        self.resize = Resize((224, 224))
        self.normalize = Normalize(means, stds)
        self.trans = transforms.Compose([
            transforms.TenCrop((100, 100)),
            transforms.Lambda(lambda crops: [transforms.Resize((224, 224))(crop) for crop in crops]),
            transforms.Lambda(lambda crops: [transforms.AugMix()(crop) for crop in crops]),
            transforms.Lambda(lambda crops: torch.stack([transforms.PILToTensor()(crop).float() for crop in crops])),
        ])
        self.to_tensor = ToTensor()
        
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        img_name = f'{self.file_names.iloc[idx]}.jpg'
        full_address = os.path.join(self.data_path, img_name)
        image = Image.open(full_address)
        label = self.labels.iloc[idx]
        image = self.resize(image)
        cropped_images = self.trans(image)
        transformed_images = self.normalize(cropped_images)
        
        return transformed_images, label
    
class Dataset_Interpreter_valid_test(Dataset):
    def __init__(self, data_path, file_names, labels=None):
        self.data_path = data_path
        self.file_names = file_names
        self.labels = labels
        self.resize = Resize((224,224))
        self.normalize = Normalize(means, stds)
        self.trans = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
        
    def __len__(self):
        return (len(self.file_names))
    
    def __getitem__(self, idx):
        img_name = f'{self.file_names.iloc[idx]}.jpg'
        full_address = os.path.join(self.data_path, img_name)
        image = Image.open(full_address)
        label = self.labels.iloc[idx]
        image = self.trans(image)
#         if self.transforms is not None:
#             image = self.transforms(image)
        
        return image, label
    
train_data = Dataset_Interpreter(data_path=data_dir+'/train/', file_names=X_train, labels=y_train)
crop_data = Dataset_Interpreter_crop(data_path=data_dir+'/train/', file_names=X_train, labels=y_train)
valid_data = Dataset_Interpreter_valid_test(data_path=data_dir+'/train/', file_names=X_valid, labels=y_valid)
#test_data = Dataset_Interpreter_valid_test(data_path=data_dir+'/train/', file_names=X_test, labels=y_test)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory

#Load trained model:
model_ft = torch.hub.load('pytorch/vision:v0.10.0', 'resnext101_32x8d', pretrained=True)
num_classes=120
num_ftrs = model_ft.fc.in_features
#model_ft.fc = nn.Linear(num_ftrs, num_classes)
model_ft.fc = nn.Sequential(
    nn.Linear(num_ftrs, 256),  # Additional linear layer with 256 output features
    nn.LeakyReLU(negative_slope=0.01, inplace=True),        
    nn.Dropout(0.6),               # Dropout layer with 50% probability
    nn.Linear(256, num_classes)    # Final prediction fc layer
)
#model_ft.load_state_dict(torch.load('saved_model.pth'))
criterion = nn.CrossEntropyLoss()
for param in model_ft.parameters():
    param.requires_grad = False
    
for param in model_ft.fc.parameters():
    param.requires_grad = True
optimizer_ft = optim.Adam(model_ft.fc.parameters(), lr=0.0001, weight_decay=0.005)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)
#model_ft = nn.DataParallel(model_ft)
#model_ft.load_state_dict(torch.load('saved_model_final.pth'))
#model_ft.load_state_dict(torch.load('saved_model_final_v3.pth'))
model_ft = nn.DataParallel(model_ft)
model_ft = model_ft.to(device)    

batch_size = 128
# Pytorch train and test sets
# data loader
train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = False)
valid_loader= DataLoader(valid_data, batch_size = batch_size, shuffle = False)
#test_loader = DataLoader(test_data, batch_size = batch_size, shuffle = False)
crop_loader = DataLoader(crop_data, batch_size = batch_size, shuffle = False)

train_losses = []
valid_losses = []

def train_model(model, optimizer, scheduler, train_loader, valid_loader, loss_module, num_epochs=100):
    min_valid_loss = 0.0
    model.eval()     
    for data_inputs, data_labels in valid_loader:
        data_inputs = data_inputs.to(device)
        data_labels = np.array(data_labels)
        data_labels = torch.from_numpy(data_labels)
        data_labels = data_labels.to(device)

        ## Step 2: Run the model on the input data
        preds = model(data_inputs)
        preds = preds.squeeze(dim=1) # Output is [Batch size, 1], but we want [Batch size]

        ## Step 3: Calculate the loss
        loss = loss_module(preds, data_labels)
        min_valid_loss += loss.item()
        
    # Training loop
    for epoch in tqdm(range(num_epochs)):
        train_loss = 0.0
        # Set model to train mode
        model.train()
        for data_inputs, data_labels in train_loader:
            ## Step 1: Move input data to device (only strictly necessary if we use GPU)
            data_inputs = data_inputs.to(device)
            data_labels = np.array(data_labels)
            data_labels = torch.from_numpy(data_labels)
            data_labels = data_labels.to(device)

            ## Step 2: Run the model on the input data
#             inputs = data_inputs.view(-1, 3, 224, 224)  
#             preds = model(inputs)
#             preds = preds.view(10, -1, 120)
#             preds = preds.max(dim=0)
            preds = model(data_inputs)
            preds = preds.squeeze(dim=1) # Output is [Batch size, 1], but we want [Batch size]

            ## Step 3: Calculate the loss
            #loss = loss_module(preds[0], data_labels)
            loss = loss_module(preds, data_labels)
            ## Step 4: Perform backpropagation
            # Before calculating the gradients, we need to ensure that they are all zero.
            # The gradients would not be overwritten, but actually added to the existing ones.
            optimizer.zero_grad()
            # Perform backpropagation
            loss.backward()

            ## Step 5: Update the parameters
            optimizer.step()
            
            ## Step 6: Add to loss
            train_loss += loss.item()
        scheduler.step()
        
        #Validation Loop
        valid_loss = 0.0
        model.eval()     
        for data_inputs, data_labels in valid_loader:
            data_inputs = data_inputs.to(device)
            data_labels = np.array(data_labels)
            data_labels = torch.from_numpy(data_labels)
            data_labels = data_labels.to(device)

            ## Step 2: Run the model on the input data
            preds = model(data_inputs)
            preds = preds.squeeze(dim=1) # Output is [Batch size, 1], but we want [Batch size]

            ## Step 3: Calculate the loss
            loss = loss_module(preds, data_labels)
            valid_loss += loss.item()

        print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss / len(train_loader)} \t\t Validation Loss: {valid_loss / len(valid_loader)}')
        train_losses.append(train_loss / len(train_loader))
        valid_losses.append(valid_loss / len(valid_loader))
        if min_valid_loss > valid_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = valid_loss
            # Saving State Dict
            torch.save(model.module.state_dict(), 'resnext_v3.pth')
train_model(model_ft, optimizer_ft, exp_lr_scheduler, train_loader, valid_loader, criterion, num_epochs = 70)

def train_model(model, optimizer, scheduler, train_loader, valid_loader, loss_module, num_epochs=100):
    min_valid_loss = 0.0
    model.eval()     
    for data_inputs, data_labels in valid_loader:
        data_inputs = data_inputs.to(device)
        data_labels = np.array(data_labels)
        data_labels = torch.from_numpy(data_labels)
        data_labels = data_labels.to(device)

        ## Step 2: Run the model on the input data
        preds = model(data_inputs)
        preds = preds.squeeze(dim=1) # Output is [Batch size, 1], but we want [Batch size]

        ## Step 3: Calculate the loss
        loss = loss_module(preds, data_labels)
        min_valid_loss += loss.item()
        
    # Training loop
    for epoch in tqdm(range(num_epochs)):
        train_loss = 0.0
        # Set model to train mode
        model.train()
        for data_inputs, data_labels in train_loader:
            ## Step 1: Move input data to device (only strictly necessary if we use GPU)
            data_inputs = data_inputs.to(device)
            data_labels = np.array(data_labels)
            data_labels = torch.from_numpy(data_labels)
            data_labels = data_labels.to(device)

            ## Step 2: Run the model on the input data
#             inputs = data_inputs.view(-1, 3, 224, 224)  
#             preds = model(inputs)
#             preds = preds.view(10, -1, 120)
#             preds = preds.max(dim=0)
            preds = model(data_inputs)
            preds = preds.squeeze(dim=1) # Output is [Batch size, 1], but we want [Batch size]

            ## Step 3: Calculate the loss
            #loss = loss_module(preds[0], data_labels)
            loss = loss_module(preds, data_labels)
            ## Step 4: Perform backpropagation
            # Before calculating the gradients, we need to ensure that they are all zero.
            # The gradients would not be overwritten, but actually added to the existing ones.
            optimizer.zero_grad()
            # Perform backpropagation
            loss.backward()

            ## Step 5: Update the parameters
            optimizer.step()
            
            ## Step 6: Add to loss
            train_loss += loss.item()
        scheduler.step()
        
        #Validation Loop
        valid_loss = 0.0
        model.eval()     
        for data_inputs, data_labels in valid_loader:
            data_inputs = data_inputs.to(device)
            data_labels = np.array(data_labels)
            data_labels = torch.from_numpy(data_labels)
            data_labels = data_labels.to(device)

            ## Step 2: Run the model on the input data
            preds = model(data_inputs)
            preds = preds.squeeze(dim=1) # Output is [Batch size, 1], but we want [Batch size]

            ## Step 3: Calculate the loss
            loss = loss_module(preds, data_labels)
            valid_loss += loss.item()

        print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss / len(train_loader)} \t\t Validation Loss: {valid_loss / len(valid_loader)}')
        train_losses.append(train_loss / len(train_loader))
        valid_losses.append(valid_loss / len(valid_loader))
        if min_valid_loss > valid_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = valid_loss
            # Saving State Dict
            torch.save(model.module.state_dict(), 'resnext_final_v3.pth')

model_ft = torch.hub.load('pytorch/vision:v0.10.0', 'resnext101_32x8d')
num_classes=120
num_ftrs = model_ft.fc.in_features
#model_ft.fc = nn.Linear(num_ftrs, num_classes)
model_ft.fc = nn.Sequential(
    nn.Linear(num_ftrs, 256),  # Additional linear layer with 256 output features
    nn.LeakyReLU(negative_slope=0.01, inplace=True),        
    nn.Dropout(0.6),               # Dropout layer with 50% probability
    nn.Linear(256, num_classes)    # Final prediction fc layer
)
#model_ft.load_state_dict(torch.load('saved_model.pth'))
criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.0001, weight_decay=0.005)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)
#model_ft = nn.DataParallel(model_ft)
#model_ft.load_state_dict(torch.load('saved_model_final.pth'))
model_ft.load_state_dict(torch.load('resnext_v3.pth'))
model_ft = nn.DataParallel(model_ft)
#model_ft = model_ft.to(device)   
train_model(model_ft, optimizer_ft, exp_lr_scheduler, train_loader, valid_loader, criterion, num_epochs = 70)

import torch.nn.functional as F

def eval_model(model, data_loader):
    model.eval() # Set model to eval mode
    true_preds, num_preds = 0., 0.

    with torch.no_grad(): # Deactivate gradients for the following code
        for data_inputs, data_labels in data_loader:

            # Determine prediction of model on dev set
            data_inputs = data_inputs.to(device)
            data_labels = data_labels.to(device)
            preds = model(data_inputs)
            preds = F.softmax(preds, dim=1)
            max_prob_labels = torch.argmax(preds, dim=1)
            
            # Keep records of predictions for the accuracy metric (true_preds=TP+TN, num_preds=TP+TN+FP+FN)
            true_preds += (max_prob_labels == data_labels).sum().item()
            num_preds += data_labels.shape[0]

    acc = true_preds / num_preds
    print(f"Accuracy of the model: {100.0*acc:4.2f}%")

model_ft = torch.hub.load('pytorch/vision:v0.10.0', 'resnext101_32x8d')
num_classes=120
num_ftrs = model_ft.fc.in_features
#model_ft.fc = nn.Linear(num_ftrs, num_classes)
model_ft.fc = nn.Sequential(
    nn.Linear(num_ftrs, 256),  # Additional linear layer with 256 output features
    nn.LeakyReLU(negative_slope=0.01, inplace=True),        
    nn.Dropout(0.2),               # Dropout layer with 20% probability
    nn.Linear(256, num_classes)    # Final prediction fc layer
)
model_ft.load_state_dict(torch.load('resnext_final_v3.pth'))
model_ft = nn.DataParallel(model_ft)
#model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.0001, weight_decay=0.0001)

# Assuming you have already defined 'model' and 'train_loader'
eval_model(model_ft, train_loader)
eval_model(model_ft, valid_loader)
#eval_model(model_ft, test_loader)

import pandas as pd

class Dataset_Interpreter_test(Dataset):
    def __init__(self, data_path, file_names, labels=None):
        self.data_path = data_path
        self.file_names = file_names
        self.labels = labels
        self.transforms = transforms
        self.resize = Resize((224,224))
        self.normalize = Normalize(means, stds)
        self.trans = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
        
    def __len__(self):
        return (len(self.file_names))
    
    def __getitem__(self, idx):
        img_name = f'{self.file_names.iloc[idx]}'
        full_address = os.path.join(self.data_path, img_name)
        image = Image.open(full_address)
        #image = self.resize(image)
        image = self.trans(image)
        #image = self.normalize(image)
#         if self.transforms is not None:
#             image = self.transforms(image)
        
        return image
    
files = [f for f in os.listdir(data_dir + '/test/') if os.path.isfile(os.path.join(data_dir, 'test', f)) and f.endswith(".jpg")]
files = {'id': files}
files = pd.DataFrame(files)
test_data = Dataset_Interpreter_test(data_path=data_dir+'/test/', file_names=files['id'])    
test_loader = DataLoader(test_data, batch_size = batch_size, shuffle = False)

def test_model(model, data_loader):
    model.eval() # Set model to eval mode
    predictions = []

    with torch.no_grad(): # Deactivate gradients for the following code
        for data_inputs in data_loader:

            # Determine prediction of model on dev set
            data_inputs = data_inputs.to(device)
            #data_labels = data_labels.to(device)
            preds = model(data_inputs)
            preds = F.softmax(preds, dim=1)
            for i in preds:
                predictions.append(i)
        
    return predictions
# Assuming you have already defined 'model' and 'train_loader'
raw_predictions = test_model(model_ft, test_loader)

predictions = []
for i in range(0, len(raw_predictions)):
    prediction = raw_predictions[i].cpu()
    image_id = files.iloc[i]['id']
    image_id = image_id.split('.jpg')[0]
    predictions.append({'image_id': image_id, 'probs': prediction})
    
def generate_submission(predictions, sample_submission_path, output_path):
    """
    Generate a Kaggle submission file based on the given predictions.

    Parameters:
    - predictions: A dictionary with image ids as keys and a list of 120 probabilities as values.
    - sample_submission_path: Path to the provided sample submission file.
    - output_path: Path to save the generated submission file.
    """
    # Load the sample submission
    sample_submission = pd.read_csv(sample_submission_path)
    
    # Replace the sample probabilities with the actual predictions
    for i in range(0, len(predictions)):
        prediction = predictions[i]
        preds = prediction['probs'].tolist()
        sample_submission.loc[sample_submission['id'] == prediction['image_id'], sample_submission.columns[1:]] = preds
    # Save the modified sample submission as the final submission
    sample_submission.to_csv(output_path, index=False)
generate_submission(predictions, data_dir + '/sample_submission.csv', 'my_submission_resnext_v3.csv')