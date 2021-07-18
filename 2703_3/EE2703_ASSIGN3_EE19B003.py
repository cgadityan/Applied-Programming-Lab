import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.special import *
from pylab import *

sigma = np.logspace(-1,-3,9)

print('Sigma :')
print(sigma)


data = np.loadtxt(fname = "fitting.dat", delimiter = " ")


df =  pd.DataFrame(data)

plt0 = plt.figure(0)###################################################################
#Plotting all data points
for i in range(1,10):
	plt.plot(df.iloc[:,0], df.iloc[:,i], label = "Noise: sigma: {}".format(sigma[i-1]))


#To find a fitting function, we need J2(t) from the data points
######################finding mean value from data points #############################
##########################"to get the fitting data"####################################

mean = df.iloc[:,1:].mean(axis=1)

#since the given function has A=1.05 and B=-0.105
A_true = 1.05 				#true
B_true = -0.105
J2 = (mean + 0.105*df.iloc[:,0])/1.05
t = df.iloc[:,0]

#Converting pandas dataframe into numpy array to work with
temp = np.array(t)
t_arr = np.array(temp).T

temp = np.array(J2)
J2_arr = np.array(temp).T

#no.of rows in data
r = 101

cols = np.array(df.iloc[:,1:])


#True value of the data
Orig = A_true*J2_arr+B_true*t_arr

#Fitting function plot for "Q4"
#########################################################################################
plt.plot(t, Orig, label="A={},B={}".format(A_true,B_true),color='black', linewidth=4)
plt.title("Q4 : Function with the true value")
plt.grid(True)
plt.xlabel('t')
plt.ylabel('f(t)')
plt.legend()

# Function to calculate the value based on A and B in "Q4"
def g1(A = A_true, B = B_true):
	Func = A*J2 +B*t
	return Func
#Numpy version of g Function 'g'
def g(A = A_true, B = B_true):
	Func = A*J2_arr +B*t_arr
	return Func


 #Plot of Errorbar in "Q5"
plt2 = plt.figure(1)######################################################################
plt.errorbar( t[::5], cols[:,0][::5], yerr = sigma[0] ,fmt = 'o',label = "Errorbar")
plt.plot(t, Orig, label="A={},B={}".format(A_true,B_true),color='black', linewidth=2)

plt.xlabel('t')
plt.grid(True)
plt.legend()
plt.title('Q5 : Errorbar of first column')


#Checking for matrix to be equal to the function created in "Q4" for "Q6"
M = c_[J2_arr,t_arr]
p = np.array([A_true,B_true]).T
Func = pd.Series(np.dot(M,p))

#"g1" is used here to prove matrices and function are equal, as it is in pandas dataframe to match 
if Func.equals(g1()):
	print('\nThe Martix and the function are equal')
else:
	print('\nThe Martix and the function are "not" equal')


#Ranging values for A and B as said in "Q7"
A = np.arange(0,2,0.1)
B = np.arange(-0.2,0,0.01)


#Calculating Mean square error
MSE = np.zeros([20,20])
a = 0.0
b = 0.0
min_MSE = 10000
for i in range(0,20):
	for j in range(0,20):
		MSE[i][j]  = 1/r * np.sum(np.square(cols[:,0] - g(A[i],B[j])))
		#print(MSE[i][j])
		if MSE[i][j] < min_MSE:
			min_MSE = MSE[i][j]
			a = A[i]
			b = B[j]


#Best estimate of A and B for minimum error asked in "Q8"
print("\nBest estimate of A and B : {}  {}".format(a,b))


#Contour plot for "Q8"
plt.figure(2)#################################################
plt.contour(B,A,MSE)
plt.title('Q8: Contour plot of Mean Square Error')
plt.plot(b,a,'-or')
plt.xlabel('B')
plt.ylabel('A')
plt.grid(True)


err = np.zeros((2,9))

for i in range(0,9):
	p = np.linalg.lstsq(M, cols[:,i],rcond=-1)
	err[:,i] = np.array(p[0])

#To subtract from estimated A,B array whose error needs to be plotted
true_val = np.zeros((2,9))

for i in range(0,9):
	true_val[0,i]=A_true
	true_val[1,i]=B_true

err = err - true_val
err = abs(err)

#Contour plot for  "Q10"
plt.figure(3)#################################################

plt.plot(sigma,err[0,:],'-o')

plt.plot(sigma,err[1,:],'-o')
plt.legend(["A: Error ", "B: Error"])
plt.title('Q10: Variation of error with Noise')
plt.xlabel('Noise Standard Deviation')
plt.ylabel('MS Error')
plt.grid(True)

#Contour plot for "Q11"
plt.figure(4)#################################################
plt.loglog(sigma,abs(err[0,:]),'-o')
plt.loglog(sigma,abs(err[1,:]),'-o')
plt.legend(["Aerr","Berr"])
plt.title('Q11: Variation of error with Noise in a log,log scale')
plt.xlabel('Sigma stddev of noise')
plt.ylabel('MS Error')
plt.grid(True)

plt.show()




