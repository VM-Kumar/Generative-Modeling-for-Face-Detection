import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import multivariate_normal
from numpy.linalg import inv
import math
import cv2
from PIL import Image
from math import *
import glob



img_face_train=[]
for i in range(1000):
    pat=r'C:\Users\venkatesh\Desktop\project1_CV\dataset_face_train\im{}.jpg'.format(i+1)
    image=cv2.imread(pat)
    img_face_train.append(image)    
img_face_train=np.array(np.reshape(img_face_train,(1000,1200)))


img_face_test=[]
for i in range(100):
    pat=r'C:\Users\venkatesh\Desktop\project1_CV\dataset_face_test\im{}.jpg'.format(i+1)
    image=cv2.imread(pat)
    img_face_test.append(image)    
img_face_test=np.array(np.reshape(img_face_test,(100,1200)))   

img_nonface_train=[]
for i in range(1000):
    pat=r'C:\Users\venkatesh\Desktop\project1_CV\dataset_nonface_train\im{}.jpg'.format(i+1)
    image=cv2.imread(pat)
    img_nonface_train.append(image)    
img_nonface_train=np.array(np.reshape(img_nonface_train,(1000,1200)))   

img_nonface_test=[]
for i in range(100):
    pat=r'C:\Users\venkatesh\Desktop\project1_CV\dataset_nonface_test\im{}.jpg'.format(i+1)
    image=cv2.imread(pat)
    img_nonface_test.append(image)   
img_nonface_test=np.array(np.reshape(img_nonface_test,(100,1200)))



def MLE(image,n):
	shape = (20, 20,3)
	sum_x  = image.sum(axis = 0)
	mu = (sum_x/(len(image)))
	print('Mean(MLE).shape  : ', mu.shape)
	mu_cap = np.reshape(mu, shape)
	cv2.imwrite('C:/Users/venkatesh/Desktop/project1_CV/results/gaussian/Mean{}.jpg'.format(n),mu_cap)

	sub_var = np.square(np.subtract(image, mu))
	sum_var_x = sub_var.sum(axis = 0)
	covar = (sum_var_x/len(image))
	covar_cap  = np.sqrt(covar)
	print('Covariance(MLE).shape : ', covar.shape)
	covar_cap = np.reshape(covar_cap, shape)
	cv2.imwrite('C:/Users/venkatesh/Desktop/project1_CV/results/gaussian/Covar{}.jpg'.format(n),covar_cap)

	return image, mu, covar



def PlotROC(prob_face, prob_nonface, num_of_images, no_roc):
	no_roc = 100
	term1=np.subtract( prob_face , prob_nonface)
	print('Term1 shape: ', term1.shape)

	threshold = np.linspace(np.min(term1), np.max(term1), no_roc)
	print('threshold.shape : ', threshold.shape)

	TP = []
	TN = []
	FP = []
	FN = []

	for k in range(len(threshold)):
		TP.append(term1[:100] >= threshold[k])
		FN.append(term1[:100] < threshold[k])
		TN.append(term1[100:200] < threshold[k])
		FP.append(term1[100:200] >= threshold[k])	
	TP = np.sum(TP, axis = 1)
	TN = np.sum(TN, axis = 1)
	FP = np.sum(FP, axis = 1)
	FN = np.sum(FN, axis = 1)
	#print(TP, TN, FP, FN)
	FPRate = np.sum(prob_face[100:200] > prob_nonface[100:200])/100
	FNRate = np.sum(prob_face[:100] < prob_nonface[:100])/100
	MCRate = (FPRate + FNRate)/2

	print('False Positive Rate : ', FPRate)
	print('False Negative Rate : ', FNRate)
	print('Misclassification Rate : ', MCRate)

	plt.plot(FP/100,TP/100, marker='.')
	plt.title('ROC curve')
	plt.xlabel('False Positive Rate ')	
	plt.ylabel('True Positive Rate ')
	plt.xlim(0,1)
	plt.ylim(0,1)	
	plt.show()

def Norm(img,mu,sigma):
    sigma_term=np.linalg.slogdet(sigma)[1]*(-1/2)
    print(img.shape,mu.shape,sigma.shape)
    t=np.zeros(200)
    for i in range(200):
        t[i]=sigma_term - np.matmul(np.transpose(img[i]-mu),np.matmul(np.linalg.pinv(sigma),(img[i]-mu)))
    return t




print('Testing MLE')
a, mu_face, covar_face = MLE(img_face_train,1)
b, mu_nonface, covar_nonface = MLE(img_nonface_train,2)
image_face = img_face_test
image_nonface = img_nonface_test
print(image_face.shape)
print(image_nonface.shape)
covar_face = np.diag(covar_face)
covar_nonface = np.diag(covar_nonface)
image = np.append(image_face, image_nonface, axis = 0)
num_of_images = len(image)
print('Image shape : ', image.shape)
print('Mu face shape : ', mu_face.shape, ', Sigma face shape : ', covar_face.shape)
print('Mu non-face shape : ', mu_nonface.shape, ', Sigma non-face shape : ', covar_nonface.shape)
face_norm = Norm(image, mu_face, covar_face)
nonface_norm = Norm(image, mu_nonface, covar_nonface)
print(face_norm.shape)
print(nonface_norm.shape)
face_norm = np.reshape(face_norm, (num_of_images))
nonface_norm = np.reshape(nonface_norm, (num_of_images))
print(face_norm.shape)
print(nonface_norm.shape)
PlotROC(face_norm, nonface_norm, num_of_images, no_roc = 5)


