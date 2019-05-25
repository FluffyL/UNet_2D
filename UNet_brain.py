import os
import time
import nibabel as nib
import numpy as np

import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import tensor
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
from PIL import Image

from predata_2D import load_dataset,crop_data,convert_one_hot



RANDOM_SEED = 123
patch_size = 32
dataset_dir = '/home/lly/Desktop/intern_Xidian/WM&GM_segmentation/code/UNet_Pytorch/LPBA40/native_space/'
croped_dir = '/home/lly/Desktop/intern_Xidian/WM&GM_segmentation/code/UNet_Pytorch/LPBA40/train_data/2D/'
predict_dir = '/home/lly/Desktop/intern_Xidian/WM&GM_segmentation/code/UNet_Pytorch/predict/2D/'
s_train = 30#the num of subject used for training
s_predict = 2#the num of subject need to be predict

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



class MRIDataset(Dataset):
	"""loading MRI data"""
	def __init__(self, croped_dir, mode,t_num, img_transform=None, labels_transform=None):

		self.croped_dir = croped_dir
		self.mode = mode
		self.t_num = t_num
		#self.labels = convert_labels(labels)
		self.img_transform = img_transform
		self.labels_transform = labels_transform

	def __getitem__(self, index):
		index = index + 1
		[T1,label] = load_dataset(self.croped_dir, index, self.mode)
		T1 = T1[:,:,0]
		label = label[:,:,0]
		T1 = Image.fromarray(T1)
		label = Image.fromarray(label)
		
		if self.img_transform is not None:
			T1 = self.img_transform(T1)
		if self.labels_transform is not None:
			label = self.labels_transform(label)

		return T1, label

	def __len__(self):
		return self.t_num



img_transform = transforms.Compose([
	#transforms.RandomAffine(degrees=(-20, 20), translate=(0.15, 0.15)),
	transforms.Resize(size=(256,128)),
	transforms.ToTensor(), 
	])

labels_transform = transforms.Compose([
	#transforms.RandomAffine(degrees=(-20, 20), translate=(0.15, 0.15)),
	transforms.Resize(size=(256,128)),
	transforms.ToTensor(), 
	])



total_train = 0
total_test = 0
total_predict = 0

for subject_id in range(1,s_train+1):
	total_train = total_train + crop_data(dataset_dir, croped_dir, subject_id, total_train, mode=0, patch_size=32)

for subject_id in range(s_train+1,41):
	total_test = total_test + crop_data(dataset_dir, croped_dir, subject_id, total_test, mode=1, patch_size=32)
'''
for subject_id in range(s_predict):
	total_predict = total_predict + crop_data(predict_dir, croped_dir, subject_id, total_test, mode=2, patch_size=32)
'''

#total_train = 7680
#total_test = 2560
BATCH_SIZE = 8
train_dataset = MRIDataset(croped_dir,
			t_num = total_train,
			mode = 0,
			img_transform = img_transform,
			labels_transform = img_transform)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True)

valid_dataset = MRIDataset(croped_dir,
			t_num = total_train,
			mode = 1,
			img_transform = img_transform,
			labels_transform = img_transform)

valid_loader = DataLoader(dataset=valid_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True)
'''
predict_dataset = PredictDataset(dataset_dir = predict_dir,
			t_num = total_predict
			img_transform = None,
			labels_transform = None)

predict_loader = DataLoader(dataset=predict_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=4)
'''
torch.manual_seed(0)

num_epochs = 2
for epoch in range(num_epochs):

	for batch_idx, (T1, labels) in enumerate(train_loader):

		print('Epoch:', epoch+1, end='')
		print(' | Batch index:', batch_idx, end='')
		print(' | Batch size:', labels.size()[0])
		Tshape = T1.shape
		T1 = T1.view(-1, 1, Tshape[1],Tshape[2]).to(DEVICE)
		labels = labels.view(-1, 1,Tshape[1],Tshape[2]).to(DEVICE)
		print('Image shape', T1.shape)
		print('break minibatch for-loop')
		break

#################################
### UNet Model
#################################

class double_conv(nn.Module):
	'''(conv => BN => ReLU) * 2'''
	def __init__(self, in_ch, out_ch):
		super(double_conv, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, 3, padding=1),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_ch, out_ch, 3, padding=1),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True)
			)

	def forward(self, x):
		x = self.conv(x)
		return x


class inconv(nn.Module):
	def __init__(self, in_ch, out_ch):
		super(inconv, self).__init__()
		self.conv = double_conv(in_ch, out_ch)

	def forward(self, x):
		x = self.conv(x)
		return x


class down(nn.Module):
	def __init__(self, in_ch, out_ch):
		super(down, self).__init__()
		self.mpconv = nn.Sequential(
			nn.MaxPool2d(2),
			double_conv(in_ch, out_ch)
		)

	def forward(self, x):
		x = self.mpconv(x)
		return x


class up(nn.Module):
	def __init__(self, in_ch, out_ch):
		super(up, self).__init__()
		self.upconv = nn.Sequential(
 		double_conv(in_ch, out_ch),
		nn.Upsample(scale_factor=2, mode='nearest'))

	def forward(self, x):
		x = self.upconv(x)
		return x


class outconv(nn.Module):
	def __init__(self, in_ch, out_ch):
		super(outconv, self).__init__()
		self.conv = nn.Conv2d(in_ch, out_ch, 1)

	def forward(self, x):
		x = self.conv(x)
		return x
	
class UNet(nn.Module):
	def __init__(self, n_channels, n_classes):
		super(UNet, self).__init__()
		self.inc = inconv(n_channels, 32)
		self.down1 = down(32, 64)
		self.down2 = down(64, 128)
		self.down3 = down(128, 256)
		self.down4 = down(256, 512)
		self.down5 = down(512,1024)
		#self.center = UnetConvBlock(256,512,is_batchnorm = True)
		self.up5 = up(1024,512)
		self.up4 = up(512, 256)
		self.up3 = up(256, 128)
		self.up2 = up(128, 64)
		self.up1 = up(64, 32)
		self.up0 = up(32, n_classes)

	def forward(self, x):
		x1 = self.inc(x.float())
		x2 = self.down1(x1)
		x3 = self.down2(x2)
		x4 = self.down3(x3)
		x5 = self.down4(x4)
		x6 = self.down5(x5)
		x5 = self.up5(x6)
		x4 = self.up4(x5)
		x3 = self.up3(x4)
		x2 = self.up2(x3)
		x1 = self.up1(x2)
		x0 = self.up0(x1)
		s0 = x.size()[2]
		s1 = x.size()[3]
		self.out = nn.Upsample(size = (s0,s1))
		logists = self.out(x0)
		probas = F.softmax(logists)
		return logists,probas

#################################
### Model Initialization
#################################
torch.manual_seed(RANDOM_SEED)

model = UNet(n_channels=1, n_classes=4)
model = model.float()

optimizer = torch.optim.SGD(model.parameters(), lr=0.05,momentum=0.9)

#################################
### Training
#################################
def focal_loss(predict,label):
	gamma = 2
	alpha = 0.25

	label = convert_one_hot(label)
	
	label_num = label.shape
	ones = torch.ones(label_num)
	zeros = torch.zeros(label_num)
	
	return cost


def tversky(predict, label):
	#define loss function
	'''
	Dice : alpha=beta=0.5
	jaccard : alpha=beta=1
	variant Dice : alpha+beta=1
	'''
	alpha = 1
	beta = 1

	#convert labels to one_hot matrix
	label = convert_one_hot(label)

	label_num = label.shape
	ones = torch.ones(label_num)
	p0 = predict
	p1 = ones - p0
	g0 = label
	g1 = ones - g0

	TP = torch.sum(p0 * g0)
	FP = torch.sum(p1 * g0)
	FN = torch.sum(p0 * g1)
	tversky = TP / (TP + alpha*FN + beta*FP)
	cost = 1-tversky

	return cost

def Dice_3D(predict,label):
	label = convert_one_hot(label)

	label_num = label.shape
	ones = torch.ones(label_num)
	p0 = predict
	p1 = ones - p0
	g0 = label
	g1 = ones - g0

	TP = torch.sum(p0 * g0)
	FP = torch.sum(p1 * g0)
	FN = torch.sum(p0 * g1) 

	dice = 2*TP / (2*TP + FP + FN)
	cost = 1-dice
	return cost

def focal_loss(predict, label):
	gamma = 2
	alpha = 0.25

	pt_1 = torch.where(torch.equal(label,1), predict,torch.ones(predict))
	pt_0 = torch.where(torch.equal(label,0), predict,torch.ones(predict))
	#cost = (-1)*torch.sum(alpha*torch.pow(1. - pt_1, gamma) * torch.log(pt_1))-torch.sum((1-alpha
	return 0
						

def compute_epoch_loss(model, data_loader):
	curr_loss, num_examples = 0., 0
	
	with torch.no_grad():
		for T1, labels in data_loader:
			Tshape = T1.shape
			T1 = T1.to(DEVICE)
			labels = labels.long().to(DEVICE)
			logits, probas = model(T1)
			#loss = 0
			loss = Dice_3D(probas,labels)
			#cost = F.cross_entropy(logits, labels)
			
			num_examples += Tshape[0]*Tshape[1]*Tshape[2]*Tshape[3]
			curr_loss += loss

		curr_loss = curr_loss / num_examples
		return curr_loss

################################################
# THE compute_accuracy function
################################################

def compute_accuracy(model, data_loader):
	correct_pred, num_examples = 0, 0
	with torch.no_grad():
		for T1, labels in data_loader:
			T1 = T1.to(DEVICE)
			Tshape = T1.shape
			labels = labels.to(DEVICE)
			logits, probas = model.forward(T1)
			predicted_labels = torch.argmax(probas, 1)
			predicted_labels = predicted_labels.view(-1,1,Tshape[2],Tshape[3])
			num_examples += Tshape[0]*Tshape[1]*Tshape[2]*Tshape[3]
			correct_pred += (predicted_labels.float() == labels.float()).sum()
		return correct_pred.float()/num_examples * 100

if __name__ == '__main__':
	start_time = time.time()
	minibatch_cost = []
	epoch_cost = []

	NUM_EPOCHS = 4

	for epoch in range(NUM_EPOCHS):
		model.train()
		cost = 0
		for batch_idx, (T1, labels) in enumerate(train_loader):
			Tshape = T1.shape
			T1 = T1.to(DEVICE)
			labels = labels.to(DEVICE)
	            
			### FORWARD AND BACK PROP
			logits, probas = model(T1)
			cost = Dice_3D(probas, labels)
			#cost = F.cross_entropy(logits, labels)
			optimizer.zero_grad()
        
			cost.backward()
			minibatch_cost.append(cost)
			### UPDATE MODEL PARAMETERS
			optimizer.step()
	        
			### LOGGING
			if not batch_idx % 50:
				print ('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f' 
	 				%(epoch+1, NUM_EPOCHS, batch_idx, 
					len(train_loader), cost))
	        
		model.eval()
	    
		cost = compute_epoch_loss(model, train_loader)
		epoch_cost.append(cost)
	
		torch.save(model, os.path.join(predict_dir,'model.pt'))
		train_accuracy = compute_accuracy(model, train_loader)
		valid_accuracy = compute_accuracy(model, valid_loader)
	    
		print('Epoch: %03d/%03d Train Cost: %.4f' % (
			epoch+1, NUM_EPOCHS, cost))
		print('Train Accuracy: %.3f | Validation Accuracy: %.3f' % (train_accuracy, 		valid_accuracy))
		print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
	    
	print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
	'''

	plt.plot(range(len(minibatch_cost)), minibatch_cost)
	plt.ylabel('cost')
	plt.xlabel('Minibatch')
	plt.show()
	
	plt.plot(range(len(epoch_cost)), epoch_cost)
	plt.ylabel('cost')
	plt.xlabel('Epoch')
	plt.show()
	'''
	print('Test Accuracy: %.2f' % compute_accuracy(model, valid_loader))
	'''
	def predict_data(model, data_loader, predict_dir, total_num):
		global affine
		pre_name = total_num
		with torch.no_grad():
				 
			for T1 in data_loader:
				pre_name = pre_name + 1
				T1 = T1.view(-1,1,patch_size,patch_size,patch_size).to(DEVICE)
				logits, probas = model.forward(T1)
				predicted_labels = torch.argmax(probas, 1)
				for i in range(T1.shape[0]):
					save_T1 = nib.Nifti1Image(T1, affine)
					save_T1.to_filename(os.path.join(predict_dir, str(pre_name+i) + '_T1.nii.gz'))
					save_predict = nib.Nifti1Image(predicted_labels, affine)
					save_predict.to_filename(os.path.join(predict_dir, str	(pre_name+i) +'_predict.nii.gz'))

'''




