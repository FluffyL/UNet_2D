import os
import sys
import gzip
import shutil
import torch
import nibabel as nib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage import io

dataset_dir = '/home/lly/Desktop/intern_Xidian/WM&GM_segmentation/code/UNet_Pytorch/LPBA40/native_space/'
croped_dir = '/home/lly/Desktop/intern_Xidian/WM&GM_segmentation/code/UNet_Pytorch/LPBA40/train_data/2D/'



def load_dataset(croped_dir,index, mode):

	croped_name = str(index)

	if mode == 0:
		f1 = os.path.join(croped_dir,croped_name+'_train_T1.nii.gz')
		fl = os.path.join(croped_dir,croped_name+'_train_label.nii.gz')
		img_T1 = nib.load(f1)
		inputs_T1 = img_T1.get_data()
		img_label = nib.load(fl)
		inputs_label = img_label.get_data()
	elif mode == 1:
		f1 = os.path.join(croped_dir,croped_name+'_test_T1.nii.gz')
		fl = os.path.join(croped_dir,croped_name+'_test_label.nii.gz')
		img_T1 = nib.load(f1)
		inputs_T1 = img_T1.get_data()
		img_label = nib.load(fl)
		inputs_label = img_label.get_data()
	else:
		f1 = os.path.join(croped_dir,croped_name+'_predict.nii.gz')
		img_T1 = nib.load(f1)
		inputs_T1 = img_T1.get_data()
		inputs_label = None
	
	return [inputs_T1,inputs_label]
	
	
def crop_data(dataset_dir, croped_dir, subject_id, total_t, mode, patch_size):
	
	subject_name = 'S%d' % subject_id
	if subject_id<10:
		T1_name = 'S0%d' % subject_id
	else:
		T1_name = 'S%d' % subject_id

	if mode == 0:
		t_name = 'train'
	elif mode == 1:
		t_name = 'test'
	elif mode == 2:
		t_name = 'predict'
	
	print('##################################################')
	print('croping ' + t_name + ' subject ' + str(subject_id) +' image...')
	
	if mode == 0 or mode == 1:
		#get T1 images
		f1 = os.path.join(dataset_dir, subject_name+'/'+T1_name+'.native.mri.hdr')
		f1img = os.path.join(dataset_dir, subject_name+'/'+T1_name+'.native.mri.img')
		
		img_T1 = nib.load(f1)
		affine = img_T1.affine
		inputs_T1 = img_T1.get_data()


		#get labels for training data
		fl = os.path.join(dataset_dir, subject_name+'/tissue/'+T1_name+'.native.tissue.hdr')
		flimg = os.path.join(dataset_dir, subject_name+'/tissue/'+T1_name+'.native.tissue.img')
		img_label = nib.load(fl)
		inputs_label = img_label.get_data()
	
		for i in range(inputs_T1.shape[0]):

			croped_name = str(total_t+i+1)
			croped_T1 = inputs_T1[:,:,i]
			croped_label = inputs_label[:,:,i]
			save_T1 = nib.Nifti1Image(croped_T1, affine)
			save_label = nib.Nifti1Image(croped_label, affine)
			T1_path = os.path.join(croped_dir,croped_name+'_'+t_name+'_T1.nii.gz')
			label_path = os.path.join(croped_dir,croped_name+'_'+t_name+'_label.nii.gz')
			if not os.path.exists(T1_path):
				os.system(r'touch %s' % T1_path)
			save_T1.to_filename(T1_path)

			if not os.path.exists(label_path) :
				 os.system(r'touch %s' % label_path)
			save_label.to_filename(label_path)

		print('croping finished.Got ' + str(i+1) + ' patches from subject '+ str(subject_id) +'...')
		print('Total patches: ' + croped_name)

	else:
		data_list = os.listdir(dataset_dir) 
		f1 = os.path.join(dataset_dir,data_list[subject_id])

		img_T1 = nib.load(f1)
		affine = img_T1.affine
		inputs_T1 = img_T1.get_data()

		for i in range(inputs_T1.shape[0]):

			croped_name = str(total_t+i+1)
			croped_T1 = inputs_T1[:,:,i]
			save_T1 = nib.Nifti1Image(croped_T1, affine)
			save_T1.to_filename(os.path.join(croped_dir,croped_name+'_'+t_name+'_T1.nii'))
		print('croping finished.Got ' + str(i+1) + ' patches from subject '+ str(subject_id) +'...')
		print('Total patches: ' + croped_name)

		#******************havev't finished yet,for prediction data***************#
		
	return inputs_T1.shape[2]

def convert_one_hot(labels):
	
	A,_,C,D = labels.shape
	one_hot = torch.zeros([A,4,C,D])

	for a in range(A):
		for c in range(C):
			for d in range(D):
				if labels[a,0,c,d] == 0:
					one_hot[a,0,c,d] = 1
				elif labels[a,0,c,d] == 1:
					one_hot[a,1,c,d] = 1
				elif labels[a,0,c,d] == 2:
					one_hot[a,2,c,d] = 1
				elif labels[a,0,c,d] == 3:
					one_hot[a,3,c,d] = 1
	return one_hot



