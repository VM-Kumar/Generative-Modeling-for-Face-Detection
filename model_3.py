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
	mu_cap = np.reshape(mu, shape)


	sub_var = np.square(np.subtract(image, mu))
	sum_var_x = sub_var.sum(axis = 0)
	covar = (sum_var_x/len(image))
	covar_cap  = np.sqrt(covar)
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


def T_D(image,n):
	x, mu_mle, covar_mle = MLE(image)
	D = len(x)
	shape = len(x[0])

	mu = x[0]
	sigma = np.diag(covar_mle)
	v = 6.6
	no_of_iterations = 10
	print('No of iterations : ', no_of_iterations)
	for no in range(no_of_iterations):
		# print('\n Iteration ', (no+1))
		
		# E-Step
		e_num = v + D
		x_mu = x - mu
		inv = np.linalg.pinv(sigma)  
		mat = np.matmul(x_mu, inv)
		e_denom = v + np.diag(np.matmul(mat, x_mu.T))
		E_h_i = np.divide(e_num,e_denom)
		print(E_h_i.shape)

		# M-Step
		mu_num = np.dot(E_h_i,x)
		den = np.sum(E_h_i, axis = 0)
		mu = np.divide(mu_num, den)

		mu_new = np.reshape(mu, (10,10))
		cv2.imwrite('C:/Users/venkatesh/Desktop/project1_CV/results/Tdistribution{}/Mean_iteration'.format(n) + str(no+1) + '.jpg', mu_new)

		x_mu = x -mu
		sigma_num = np.matmul(E_h_i,  np.square(x_mu))
		sigma = np.divide(sigma_num, den)
		sigma_new = np.reshape(sigma, (10,10))
		sigma_new = np.sqrt(sigma_new)
		sigma = np.diag(sigma)
		cv2.imwrite('C:/Users/venkatesh/Desktop/project1_CV/results/Tdistribution{}/Covar_iteration'.format(n) + str(no+1) + '.jpg',sigma_new)
	return mu, sigma, v


def multivariate_t_distribution(x,mu,Sigma,v,d):
	term1 = -(1/2)*np.log(np.prod(np.diag(np.sqrt(Sigma))))
	x_mu = x-mu
	temp1 = np.matmul(x_mu, inv(Sigma))
	power_dot = np.diag(np.matmul(temp1, np.transpose(x_mu)))
	term2 = -((d+v)/2)*np.log(1 + (power_dot/v))
	d = term1 + term2
	return d


image_face = img_face_test
image_nonface = img_nonface_test
mu_face, covar_face, v_face = T_D(img_face_train,1)
mu_nonface, covar_nonface, v_nonface = T_D(img_nonface_train,2)
print('\nTesting \n')

image = np.append(image_face, image_nonface, axis = 0)
num_of_images = len(image)
face_norm = multivariate_t_distribution(image, mu_face, covar_face, v_face, num_of_images)
nonface_norm = multivariate_t_distribution(image, mu_nonface, covar_nonface, v_nonface, num_of_images)

PlotROC(face_norm, nonface_norm, num_of_images, no_roc = 5)

