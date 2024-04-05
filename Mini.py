import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
from PIL import Image, ImageFile
import os
import pandas as pd
from torch import nn, optim
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder      
import csv
from torchvision.transforms import v2

cropped_large= r'C:\Users\User\Desktop\50024\Mini_challenge\cropped_large'
cropped_small= r'C:\Users\User\Desktop\50024\Mini_challenge\cropped_small'
csv_small= r'C:\Users\User\Desktop\50024\Mini_challenge\train_small.csv'
csv_large= r'C:\Users\User\Desktop\50024\Mini_challenge\train.csv'
final_test= r'C:\Users\User\Desktop\50024\Mini_challenge\test'
org_train= r'C:\Users\User\Desktop\50024\Mini_challenge\train\train'
test_crop= r'C:\Users\User\Desktop\50024\Mini_challenge\test_crop3'

labels_df= pd.read_csv(csv_large)
encoder= LabelEncoder()
encoder.fit(labels_df['Category'])


ImageFile.LOAD_TRUNCATED_IMAGES = True

class dataset(Dataset):
    global encoder

    def __init__(self, folder_path, transform=None):
        self.folder_path= folder_path
        self.transform= transform
        self.img_paths= [os.path.join(folder_path, img_name) for img_name in os.listdir(folder_path)]
        labels_df= pd.read_csv(csv_large)
        self.img_labels= labels_df.set_index('ID')['Category'].to_dict()
        self.encoder= LabelEncoder()
        self.encoder.fit(labels_df['Category'])
        self.numeric_labels= {k: self.encoder.transform([v])[0] for k, v in self.img_labels.items()}
        # print("img_labels:", dict(list(self.img_labels.items())[:10]))
        # print("numeric_labels:", dict(list(self.numeric_labels.items())[:10]))


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path= self.img_paths[idx]
        img_id= int(os.path.splitext(os.path.basename(img_path))[0]) 
        img= Image.open(img_path).convert("RGB")   
        label= self.numeric_labels[img_id]  
        
        if self.transform:
            img= self.transform(img)
            
        label_tensor= torch.tensor(label, dtype=torch.long)
        return img, label_tensor, img_id

transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    v2.TrivialAugmentWide(),
    v2.TrivialAugmentWide(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


data= dataset(cropped_large, transform=transform)
# data= dataset(cropped_large, transform=None)
train_size= int(0.85*len(data))
test_size= len(data) - train_size 
train_dataset, test_dataset= random_split(data, [train_size, test_size])

train_dataset.dataset.transform= train_transform
test_dataset.dataset.transform= transform


train_loader= DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader= DataLoader(test_dataset, batch_size=32, shuffle=False)

data_final= dataset(cropped_large, transform=train_transform)
test_final= dataset(test_crop, transform=transform)
train_loader_final= DataLoader(data_final, batch_size=32, shuffle=True)
test_loader_final= DataLoader(test_final, batch_size=32, shuffle=False)


# #resnet18
# device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model= models.resnet18(pretrained=True)
# num_in= model.fc.in_features
# model.fc= torch.nn.Linear(num_in, 100)
# criterion= nn.CrossEntropyLoss() 
# optimizer= optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# model.to(device)
# model.train()


#resnet50
device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model= models.resnet50(pretrained=True)
num_in= model.fc.in_features
model.fc= torch.nn.Linear(num_in, 100)
criterion= nn.CrossEntropyLoss()  
optimizer= optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
model_path= r"C:\Users\User\Desktop\50024\Mini_challenge\resnet50aug2_40_full.pth"
model.load_state_dict(torch.load(model_path))
model.to(device)
model.train()
    

# #resnet101
# device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
# model= models.resnet101(pretrained=True)
# num_in= model.fc.in_features
# model.fc= torch.nn.Linear(num_in, 100)
# criterion= nn.CrossEntropyLoss()  
# optimizer= optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# # model_path= r"C:\Users\User\Desktop\50024\Mini_challenge\resnet101aug2_20.pth"
# # model.load_state_dict(torch.load(model_path))
# model.to(device)
# # model.train()


torch.cuda.empty_cache() 
print("emptied")

cnt=1

for epoch in range(10): 
    print(f"epoch {cnt} starts")
    running_loss= 0.0
    for images, labels, _ in train_loader_final:
        images, labels= images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs= model(images)
        loss= criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss+= loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader_final)}")
    cnt+=1


model_path = r"C:\Users\User\Desktop\50024\Mini_challenge\resnet50aug2_50_full.pth"
torch.save(model.state_dict(), model_path)

# model.eval()
# correct= 0
# total= 0
# with torch.no_grad():
#     for images, labels, _ in test_loader:
#         images, labels= images.to(device), labels.to(device)
#         outputs= model(images)
#         _, predicted= torch.max(outputs.data, 1)
#         total+= labels.size(0)   
#         correct+= (predicted==labels).sum().item()

# accuracy= 100 * correct / total
# print(f'Accuracy on the test set: {accuracy:.2f}%')


predictions= {}
predictions_with_ids= []
model.eval()
with torch.no_grad():
    for images, _, img_ids in test_loader_final:
        images= images.to(device)
        outputs= model(images)
        _, predicted= torch.max(outputs, 1)
        predicted_labels= encoder.inverse_transform(predicted.cpu().numpy())  
        for img_id, pred_label in zip(img_ids.numpy(), predicted_labels):
            predictions[img_id]= pred_label

# print(predictions)

with open('predicted_res50aug2_50.csv', 'w', newline='') as file:
    output= csv.writer(file)
    output.writerow(['Id', 'Category'])
    for img_id in sorted(predictions.keys()):
        output.writerow([img_id, predictions[img_id]])