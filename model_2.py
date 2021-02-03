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
	#cv2.imwrite('C:/Users/venkatesh/Desktop/project1_CV/results/MOG/Mean.jpg',cv2.resize(mu_cap,(250,250), interpolation = cv2.INTER_AREA))

	sub_var = np.square(np.subtract(image, mu))
	sum_var_x = sub_var.sum(axis = 0)
	covar = (sum_var_x/len(image))
	covar_cap  = np.sqrt(covar)
	covar_cap = np.reshape(covar_cap, shape)
	#cv2.imwrite('C:/Users/venkatesh/Desktop/project1_CV/results/MOG/Covar.jpg',cv2.resize(covar_cap,(250,250), interpolation = cv2.INTER_AREA))

	return image, mu, covar



def PlotROC(prob_face, prob_nonface, num_of_images, no_roc):
	no_roc = 100
	term1=np.subtract( prob_face , prob_nonface)
	print('Term1 shape: ', term1.shape)

	threshold = np.linspace(np.min(term1), np.max(term1), no_roc)
	print('threshold.shape : ', threshold.shape)
	print(threshold[:10])
	print(term1[:10])

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
    t=np.zeros(1000)
    for i in range(1000):
        t[i]=sigma_term - np.matmul(np.transpose(img[i]-mu),np.matmul(np.linalg.pinv(sigma),(img[i]-mu)))
    #t.dropna(inplace=True)
    return t


def MOGM(image,n):
	x, mu_mle, covar_mle = MLE(image)
	print('\nMOGM - EM Algorithm \n')
	num_of_images = len(x)
	shape = len(x[0])

	## E - Step ##
	K = 5 #hidden variable
	lmbda = np.ones((K))/K
	mu = np.zeros((K, shape))
	for k in range(K):
		mu[k] = x[k*5]
	sigma = np.array([np.diag(covar_mle)]*K)
	
	no_of_iterations = 10
	
	for no in range(no_of_iterations):

		## E-Step
		a = np.zeros((K,num_of_images))
		b = np.zeros((K,num_of_images))
		for k in range(K):
			a[k] = Norm(x, mu[k], sigma[k])
			b[k] = lmbda[k]*a[k]
		c_sum = np.sum(b, axis = 0)
		for k in range(K):
			b[k] = b[k]/c_sum
		mean_b_k = np.mean(np.mean(b, axis = 0), axis = 0)
		for i in range(K):
			for j in range(num_of_images):
				if(math.isnan(b[i][j])):
					
					b[i][j] = mean_b_k
		r_ik = b
		
		## M-Step
		sum_ri = r_ik.sum(axis= 1)
		sum_sum_ri = sum_ri.sum(axis= 0)
		lmbda = sum_ri/sum_sum_ri

		mu_new = np.zeros((K,10,10))
		mu_numerator = np.zeros((K, 100))
		for k in range(K):
			mu_numerator[k] = np.matmul(r_ik[k], x)
			mu[k] = mu_numerator[k]/sum_ri[k]
			mu_new[k] = np.reshape(mu[k], (10,10))
			cv2.imwrite('C:/Users/venkatesh/Desktop/project1_CV/results/MOG{}/Mean_iteration_'.format(n) + str(no+1) + '_k_' + str(k) + '.jpg',mu_new[k])

		sigma_new = np.zeros((K,10,10))
		sigma_temp = np.zeros((K,100))
		for k in range(K):
			sigma_num = np.matmul(r_ik[k], np.square(x-mu[k]))
			print(sigma_num.shape)
			sigma_temp[k] = sigma_num/sum_ri[k]
			sigma[k] = np.diag(sigma_temp[k])
			sigma_new[k] = np.sqrt(np.reshape(sigma_temp[k], (10,10)))
			cv2.imwrite('C:/Users/venkatesh/Desktop/project1_CV/results/MOG{}/Covar_iteration_'.format(n) + str(no+1) + '_k_' + str(k) + '.jpg',sigma_new[k])

	print(r_ik)
	print('Finished MOGM')
	return lmbda, mu, sigma

	

	

image_face = img_face_test
image_nonface = img_nonface_test
lmbda_face, mu_face, covar_face = MOGM(img_face_train,1)
lmbda_nonface, mu_nonface, covar_nonface = MOGM(img_nonface_train,2)

print('\nTesting \n')
image = np.append(image_face, image_nonface, axis = 0)
num_of_images = len(image)


norm_0_face = lmbda_face[0]*multivariate_normal.pdf(image, mu_face[0], covar_face[0])
norm_1_face = lmbda_face[1]*multivariate_normal.pdf(image, mu_face[1], covar_face[1])
face_norm = norm_0_face + norm_1_face

norm_0_nonface = lmbda_nonface[0]*multivariate_normal.pdf(image, mu_nonface[0], covar_nonface[0])
norm_1_nonface = lmbda_nonface[1]*multivariate_normal.pdf(image, mu_nonface[1], covar_nonface[1])
nonface_norm = norm_0_nonface + norm_1_nonface


nonface_norm[nonface_norm == 0] = 3e-220
face_norm[face_norm == 0] = 3e-220
prob_face = np.log((face_norm)/(face_norm + nonface_norm))
prob_nonface = np.log((nonface_norm)/(face_norm + nonface_norm))
PlotROC(prob_face, prob_nonface, num_of_images, no_roc = 5)
	






