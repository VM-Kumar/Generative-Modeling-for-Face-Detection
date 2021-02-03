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
    image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    img_face_train.append(image)    
img_face_train=np.array(np.reshape(img_face_train,(1000,100)))


img_face_test=[]
for i in range(100):
    pat=r'C:\Users\venkatesh\Desktop\project1_CV\dataset_face_test\im{}.jpg'.format(i+1)
    image=cv2.imread(pat)
    image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    img_face_test.append(image)    
img_face_test=np.array(np.reshape(img_face_test,(100,100)))   

img_nonface_train=[]
for i in range(1000):
    pat=r'C:\Users\venkatesh\Desktop\project1_CV\dataset_nonface_train\im{}.jpg'.format(i+1)
    image=cv2.imread(pat)
    image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    img_nonface_train.append(image)    
img_nonface_train=np.array(np.reshape(img_nonface_train,(1000,100)))   

img_nonface_test=[]
for i in range(100):
    pat=r'C:\Users\venkatesh\Desktop\project1_CV\dataset_nonface_test\im{}.jpg'.format(i+1)
    image=cv2.imread(pat)
    image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    img_nonface_test.append(image)   
img_nonface_test=np.array(np.reshape(img_nonface_test,(100,100)))



def MLE(image):
	shape = (10,10)
	sum_x  = image.sum(axis = 0)
	mu = (sum_x/(len(image)))
	print('Mean(MLE).shape  : ', mu.shape)
	mu_cap = np.reshape(mu, shape)

	sub_var = np.square(np.subtract(image, mu))
	sum_var_x = sub_var.sum(axis = 0)
	covar = (sum_var_x/len(image))
	covar_cap  = np.sqrt(covar)
	print('Covariance(MLE).shape : ', covar.shape)
	covar_cap = np.reshape(covar_cap, shape)


	return image, mu, covar



def PlotROC(prob_face, prob_nonface, num_of_images, no_roc):
	no_roc = 100
	term1=np.subtract( prob_face , prob_nonface)

	threshold = np.linspace(np.min(term1), np.max(term1), no_roc)

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

	
def F_A(image,n):
	x, mu_mle, covar_mle = MLE(image)
	num_of_images = len(x)
	D = len(x[0])

	K = 10 
	mu = np.reshape(mu_mle, (100))
	mu_print = np.reshape(mu_mle, (10,10))
	phi = np.random.random((D, K))
	sigma = np.diag(np.reshape(covar_mle, (100)))
	cv2.imwrite('C:/Users/venkatesh/Desktop/project1_CV/results/FactorAnalysis{}/Mean'.format(n) + '.jpg', mu_print)
	no_of_iterations = 10
	for no in range(no_of_iterations):
		#### E-Step
		sigma_inv = inv(sigma)
		sigI_phi = np.matmul(sigma_inv, phi)
		phiT_sigI_phi = np.matmul(np.transpose(phi), sigI_phi)
		phiT_sigI_phiI = inv(phiT_sigI_phi + np.identity(K))
		phiT_sig = np.matmul(np.transpose(phi), sigma_inv)
		phi_sigma_complete = np.matmul(phiT_sigI_phiI, phiT_sig)
		x_mu = np.zeros((1100, 100))
		x_mu = x - mu
		E_hi = np.matmul(x_mu, np.transpose(phi_sigma_complete))
		E_hi = np.reshape(E_hi, (num_of_images, K, 1))

		e_hi_hiT_term2 = np.zeros((num_of_images, K, K))
		for i in range(num_of_images):
			e_hi_hiT_term2[i] = np.matmul(E_hi[i], np.transpose(E_hi[i]))
		e_hi_hiT = e_hi_hiT_term2 + phiT_sigI_phiI

		#### M-Step
		phi_term1temp = np.zeros((num_of_images, D, K))
		x_mu = np.reshape(x_mu, (num_of_images, 100, 1))
		for i in range(num_of_images):
			phi_term1temp[i] = np.matmul(x_mu[i], np.transpose(E_hi[i]))
		phi_term1 = np.sum(phi_term1temp, axis = 0)
		phi_term2 = inv(np.sum(e_hi_hiT, axis = 0))
		phi = np.matmul(phi_term1, phi_term2)
		x = np.reshape(x, (num_of_images, 100, 1))
		sigma_term1 = np.zeros((num_of_images, 100, 100))
		sigma_term2 = np.zeros((num_of_images, 100, 100))
		sigma_temp = np.zeros((num_of_images, 100))
		for i in range(num_of_images):
			sigma_term1[i] = np.matmul(x_mu[i], np.transpose(x_mu[i]))
			temp = np.matmul(E_hi[i], np.transpose(x[i]))
			sigma_term2[i] = np.matmul(phi, temp)
			sigma_temp[i] = np.diag(sigma_term1[i] - sigma_term2[i])
		x = np.reshape(x, (num_of_images, 100))
		sigma = np.sum(sigma_temp, axis = 0)
		sigma = sigma/num_of_images
		sigma_print = np.sqrt(np.reshape(sigma, (10,10)))
		sigma = np.diag(np.reshape(sigma, (100)))
		cv2.imwrite('C:/Users/venkatesh/Desktop/project1_CV/results/FactorAnalysis{}/Sigma_iteration'.format(n) + str(no+1) + '.jpg', sigma_print)
	return mu, sigma, phi

image_face = img_face_test
image_nonface = img_nonface_test
mu_face, covar_face, phi_face =F_A( img_face_train,1)
mu_nonface, covar_nonface, phi_nonface = F_A(img_nonface_train,2)

print('\nTesting \n')

image = np.append(image_face, image_nonface, axis = 0)
num_of_images = len(image)


new_covar_face = np.add(np.matmul(phi_face, np.transpose(phi_face)), covar_face)	
new_covar_nonface = np.add(np.matmul(phi_nonface, np.transpose(phi_nonface)), covar_nonface)

face_norm = multivariate_normal.logpdf(image, mu_face, new_covar_face)
nonface_norm = multivariate_normal.logpdf(image, mu_nonface, new_covar_nonface)
PlotROC(face_norm, nonface_norm, num_of_images, no_roc = 5)



