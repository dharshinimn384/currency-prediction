# memory footprint support libraries/code
!ln -sf /opt/bin/nvidia-smi /usr/bin/nvidia-smi
!pip install gputil
!pip install psutil
!pip install humanize
import psutil
import humanize
import os
import GPUtil as GPU
GPUs = GPU.getGPUs()
# XXX: only one GPU on Colab and isn’t guaranteed
gpu = GPUs[0]
def printm():
    process = psutil.Process(os.getpid())
    print("Gen RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ), " | Proc size: " + humanize.naturalsize( process.memory_info().rss))
    print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB".format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil*100, gpu.memoryTotal))
printm()
from google.colab import drive
drive.mount('/content/drive')
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import gc
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms,datasets,models
from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.preprocessing import StandardScaler
import time
import datetime
from PIL import Image
import warnings
from tqdm.notebook import tqdm
import random
import pandas as pd
warnings.simplefilter('ignore')
torch.manual_seed(47)
np.random.seed(47)
random.seed(47)
torch.cuda.manual_seed(47)
torch.backends.cudnn.deterministic = True
save_path = "./model_currency.pth"

path = '/content/drive/MyDrive/currency rcognition/Train'
image_path = []
target = []
for i in os.listdir(path):
    for j in os.listdir(os.path.join(path, i)):
        image_path.append(os.path.join(path, i, j))
        target.append(i)
table = {'image_path': image_path, 'target': target}
train_df = pd.DataFrame(data=table)
train_df = train_df.sample(frac = 1).reset_index(drop=True)
path = '/content/drive/MyDrive/currency rcognition/Test'
image_path=[]
target=[]
for i in os.listdir(path):
    for j in os.listdir(os.path.join(path,i)):
        image_path.append(os.path.join(path,i,j))
        target.append(i)
table = {'image_path': image_path, 'target': target}
test_df = pd.DataFrame(data=table)
test_df = test_df.sample(frac = 1).reset_index(drop=True)
train_df.head()
label_mapping = {"5Hundrednote": 0,
                "1Hundrednote": 1,
                "2Hundrednote": 2,
                "Tennote": 3,
                "Fiftynote": 4,
                "Twentynote": 5,
                "2Thousandnote": 6}
train_df['target'] = train_df['target'].map(label_mapping).astype(int)
test_df['target'] = test_df['target'].map(label_mapping).astype(int)
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 12))
x = train_df.target.value_counts()
sns.barplot(x=x.index, y=x)
plt.gca().set_ylabel('samples')
plt.show()
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 12))
x = test_df.target.value_counts()
sns.barplot(x=x.index, y=x)
plt.gca().set_ylabel('samples')
plt.show()

class CustomDataset(Dataset):
    def __init__(self,dataframe,transform):
        self.dataframe = dataframe
        self.transform = transform
    def __len__(self):
        return self.dataframe.shape[0]
    def __getitem__(self,index):
        image = self.dataframe.iloc[index]['image_path']
        image = cv2.imread(image)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        label = int(self.dataframe.iloc[index]["target"])
        return {"image": torch.tensor(image, dtype=torch.float), "targets": torch.tensor(label, dtype = torch.long)}
    def get_model(classes=7):
    model = models.resnet50(pretrained=True)
    features = model.fc.in_features
    model.fc = nn.Linear(in_features = features, out_features = classes)
    return model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device
model = get_model()
model.to(device)
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])
test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])
optimizer  = optim.Adam(model.parameters(),lr = 0.00003)
loss_function = nn.CrossEntropyLoss()
train_dataset = CustomDataset(
dataframe=train_df,
transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size = 16, shuffle = True, num_workers = 4)
valid_dataset = CustomDataset(
dataframe=test_df,
transform=test_transform)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=4)
best_accuracy = 0

for epochs in tqdm(range(15),desc="Epochs"):
    model.train()
    for data_in_model in tqdm(train_loader, desc="Training"):
        inputs = data_in_model['image']
        target = data_in_model['targets']

        inputs = inputs.to(device, dtype = torch.float)
        targets = target.to(device, dtype = torch.long)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs,targets)
        loss.backward()
        optimizer.step()

    model.eval()
    final_targets = []
    final_outputs = []
    val_loss = 0
    with torch.no_grad():
        for data_in_model in tqdm(valid_loader, desc="Evaluating"):
            inputs = data_in_model['image']
            targets = data_in_model['targets']

            inputs = inputs.to(device, dtype = torch.float)
            targets = targets.to(device, dtype = torch.long)

            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            val_loss += loss
            _,predictions = torch.max(outputs, 1)

            targets = targets.detach().cpu().numpy().tolist()
            predictions = predictions.detach().cpu().numpy().tolist()

            final_targets.extend(targets)
            final_outputs.extend(predictions)
    PREDS = np.array(final_outputs)
    TARGETS = np.array(final_targets)
    acc = (PREDS == TARGETS).mean() * 100
    if(acc>best_accuracy):
        best_accuracy = acc
        torch.save(model.state_dict(), save_path)
    print("EPOCH: {}/10".format(epochs+1))
    print("ACCURACY---------------------------------------------------->{}".format(acc))
    print("LOSS-------------------------------------------------------->{}".format(val_loss))
    
    def test_model(model, image_path):

    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_image = transform(image).unsqueeze(0)


    model.eval()
    input_image = input_image.to(device, dtype=torch.float)
    with torch.no_grad():
        output = model(input_image)
    _, predicted_class = torch.max(output, 1)

    class_labels = ['5Hundrednote', '1Hundrednote', '2Hundrednote', 'Tennote', 'Fiftynote', 'Twentynote', '2Thousandnote']
    predicted_label = class_labels[predicted_class.item()]

    return predicted_label

test_image_path = "/content/drive/MyDrive/currency rcognition/1.jpg"
predicted_currency = test_model(model, test_image_path)

print(f"The predicted currency is: {predicted_currency}")


!pip install pyttsx3
!pip install gtts
from gtts import gTTS

def test_model(model, image_path):
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_image = transform(image).unsqueeze(0)

    model.eval()
    input_image = input_image.to(device, dtype=torch.float)
    with torch.no_grad():
        output = model(input_image)
    _, predicted_class = torch.max(output, 1)

    class_labels = ['5Hundrednote', '1Hundrednote', '2Hundrednote', 'Tennote', 'Fiftynote', 'Twentynote', '2Thousandnote']
    predicted_label = class_labels[predicted_class.item()]

    return predicted_label

test_image_path = "/content/drive/MyDrive/currency rcognition/Train/1Hundrednote/13.jpg"  # Replace with your actual path
predicted_currency = test_model(model, test_image_path)

print(f"The predicted currency is: {predicted_currency}")

tts = gTTS(text=predicted_currency, lang='en')
tts.save('predicted_currency3.mp3')


os.system('predicted_currency3.mp3')