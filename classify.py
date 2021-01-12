import glob
from itertools import chain
import os
import random
import zipfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
from linformer import Linformer
from vit_pytorch.efficient import ViT
#image is 640 by 480

device = 'cuda'
gamma = 0.7
batch_size = 1
epochs = 20
lr = 3e-5
seed = 33



def seed_everything(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True

seed_everything(seed)

efficient_transformer = Linformer(
	dim=1024,
	seq_len=769,  # 7x7 patches + 1 cls-token
	depth=12,
	heads=8,
	k=40
)


model = ViT(
	image_size = 640, #max(height, width)
	patch_size = 20, #common factor of both height and width
	num_classes = 1000,
	dim = 1024,
	transformer=efficient_transformer,
	channels=3,
).to(device)

# loss function
criterion = nn.CrossEntropyLoss()
# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

#how to modify to get scalar (speed) rather than classes?

train_dir = 'data/training'
train_list = glob.glob(os.path.join(train_dir,'*.png'))
print(f"Train Data: {len(train_list)}")

with open('data/training/train.txt') as f:
	lines = f.read().splitlines()

labels = [int(float(line)) for line in lines][1:]

train_list, valid_list = train_test_split(train_list, 
										  test_size=0.2,
										  random_state=33)

class SpeedDataset(Dataset):
	def __init__(self, file_list, transform=None):
		self.file_list = file_list
		self.transform = transform

	def __len__(self):
		self.filelength = len(self.file_list)
		return self.filelength

	def __getitem__(self, idx):
		img_path = self.file_list[idx]
		img = Image.open(img_path)
		label = labels[idx]

		return transforms.ToTensor()(img), label

train_data = SpeedDataset(train_list)
valid_data = SpeedDataset(valid_list)


train_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle=False) #might want to reconsider if I want these to be shuffled
valid_loader = DataLoader(dataset = valid_data, batch_size=batch_size, shuffle=True) 
print(train_loader)

del train_list
del valid_list


for epoch in range(epochs):
	epoch_loss = 0
	epoch_accuracy = 0

	for data, label in tqdm(train_loader):
		data = data.type(torch.FloatTensor)
		data = data.to(device)
		label = label.to(device)

		output = model(data)
		loss = criterion(output, label)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		acc = (output.argmax(dim=1) == label).float().mean()
		epoch_accuracy += acc / len(train_loader)
		epoch_loss += loss / len(train_loader)

	with torch.no_grad():
		epoch_val_accuracy = 0
		epoch_val_loss = 0
		for data, label in valid_loader:
			data = data.to(device)
			label = label.to(device)

			val_output = model(data)
			val_loss = criterion(val_output, label)

			acc = ((val_output.argmax(dim=1) - label)**2).float().mean()
			epoch_val_accuracy += acc / len(valid_loader)
			epoch_val_loss += val_loss / len(valid_loader)

	print(
		f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
	)