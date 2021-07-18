#"   Assignment #4  
#	Fourier approximations	"


import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate 

PI = np.pi


#Question 1
def exp(x):
	y = np.exp(x)
	return y

def coscos(x):
	y = np.cos(x)
	z = np.cos(y)
	return z


#X = np.arange(0,2*PI,0.01)

#Plots in question 1 is plotted along with estimated values of coeff (Ac)
#Figure 1 and Figure 2 defined in the end


#Functions to be integrated using quad
def u(x, k, i):
	if i==0:
		return exp(x)*np.cos(k*x)
	if i==1:
		return coscos(x)*np.cos(k*x)

def v(x, k, i):
	if i==0:
		return exp(x)*np.sin(k*x)
	if i==1:
		return coscos(x)*np.sin(k*x)	


#Question 2 
A0 = []
B0 = []
A1 = []
B1 = []

#i = 0 represents exp(x)
#i = 1 represents coscos(x)


#Calculating coefficients # 51 coefficients
for i in range(0, 51):
	A0.append(integrate.quad(u,0,2*PI,args=(i,0))[0])
	A1.append(integrate.quad(u,0,2*PI,args=(i,1))[0])
	B0.append(integrate.quad(v,0,2*PI,args=(i,0))[0])
	B1.append(integrate.quad(v,0,2*PI,args=(i,1))[0])

np.array(A0)
np.array(B0)
np.array(A1)
np.array(B1)

#Making constant coeff. and Assigning final values to the coefficients
A0 = np.divide(A0,PI)
A0[0]= A0[0]/2
A1 = np.divide(A1,PI)
A1[0] = A1[0]/2
B0 = np.divide(B0,PI)
B1 = np.divide(B1,PI)


#Coeff in the form given in Q3
C0 = [] #For Func 1
C1 = [] #For Func 2

C0.append(A0[0])
C1.append(A1[0])

#Question 3
for i in range(1,51):
	C0.append(A0[i])
	C0.append(B0[i])
	C1.append(A1[i])
	C1.append(B1[i])



#defining N from 0 tp 51 
N = range(0,51)


# Plots for question 3 are plotted along with the estimated coeff as asked in Question 5


#Question 4
#defining vector x
x = np.linspace(0,2*PI,400)

#print(x)
a0 = []
b0 = []
a1 = []
b1 = []


b = exp(x)   # f has been written to take a vector
A = np.zeros((400,101))     # allocate space for A
A[:,0]=1              # col 1 is all ones
for k in range(1,51):
	A[:,2*k-1] = np.cos(k*x) # cos(kx) column
	A[:,2*k]   = np.sin(k*x)   # sin(kx) column #endfor

c0 = np.linalg.lstsq(A,b,rcond=None)[0]        # the ’[0]’ is to pull out the# best fit vector. 
						#lstsq returns a list.


a0.append(c0[0])
b0.append(0)
for i in range(1,51):
	a0.append(c0[2*i-1])
	b0.append(c0[2*i])


b = coscos(x)   # f has been written to take a vector
A = np.zeros((400,101))     # allocate space for A
A[:,0]=1              # col 1 is all ones
for k in range(1,51):
	A[:,2*k-1]= np.cos(k*x) # cos(kx) column
	A[:,2*k]  = np.sin(k*x)   # sin(kx) column #endfor

c1 = np.linalg.lstsq(A,b,rcond=None)[0]        # the ’[0]’ is to pull out the# best fit vector. 
						#lstsq returns a list.
a1.append(c1[0])
b1.append(0)
for i in range(1,51):
	a1.append(c1[2*i-1])
	b1.append(c1[2*i])



a0 = np.array(a0)
a1 = np.array(a1)
b0 = np.array(b0)
b1 = np.array(b1)


X = np.linspace(-2*PI,4*PI,1200)


Error0A = abs(A0) - abs(a0)
Error0B = abs(B0) - abs(b0)
Error1A = abs(A1) - abs(a1)
Error1B = abs(B1) - abs(b1)




print("\nErrors in Difference of coefficientsobtained by integration and Lstsq method: ")
print("\n For exp() An and Bn:")

print(Error0A)
print(Error0B)
print("\nFor cos(cos()) An and Bn:")
print(Error1A)
print(Error1B)


#Plots for Question 1 and Question 6 in Figure 1 and 2
#Extended plot from -2*PI to 4*PI for Q1
#Estimated plot from lstsq 

plt.figure(1)
plt.semilogy(X,exp(X),label = 'Original Func')
plt.semilogy(x,np.dot(A,c0),'og',label = 'Estimated Func')

plt.title(" Exponential Function ")
plt.grid(True)
plt.xlabel('x')
plt.ylabel('exp(x)')
plt.legend()
plt.semilogy()
#
#
#
plt.figure(2)
plt.plot(X,coscos(X),label = 'Original Func')
plt.plot(x,np.dot(A,c1),'og', label = 'Estimated Func')

plt.title(" Cos(Cos()) Function ")
plt.grid(True)
plt.xlabel('x')
plt.ylabel('cos(cos(x))')
plt.legend()


#For estimated coeff
n = range(0,51)
#Plots for Question 3 and also estimated coeff by lstsq method from Question 5
ne = range(1,52)
#Shifting the error plot by one unit to distinguish clearly between the estimated coeff and 
##Coeff by direct integration
##As error in cos(cos()) is in the order of 1e-15 for cos(cos()) function


plt.figure(3)
plt.semilogy(n,abs(A0),'or',label = 'An')
plt.semilogy(n,abs(B0),'ob',label = 'Bn')
plt.semilogy(ne, abs(a0),'og',label = 'an estimated')
plt.semilogy(ne, abs(b0),'oy',label = 'bn estimated')

plt.title("Semilogy plot for coeff of exp(x)")
plt.xlabel('n')
plt.ylabel('An or Bn')
plt.grid(True)
plt.legend()

plt.figure(4)
plt.loglog(n,abs(A0),'or',label = 'An')
plt.loglog(n,abs(B0),'ob',label = 'Bn')
plt.loglog(ne, abs(a0),'og',label = 'an estimated')
plt.loglog(ne, abs(b0),'oy',label = 'bn estimated')

plt.title("Loglog plot for coeff of exp(x)")
plt.xlabel('n')
plt.ylabel('An or Bn')
plt.grid(True)
plt.legend()



plt.figure(5)
plt.semilogy(n,abs(A1),'or',label = 'An')
plt.semilogy(n,abs(B1),'ob',label = 'Bn')
plt.semilogy(ne, abs(a1),'og',label = 'an estimated')
plt.semilogy(ne, abs(b1),'oy',label = 'bn estimated')

plt.title("Semilogy plot for coeff of coscos(x)")
plt.xlabel('n')
plt.ylabel('An or Bn')
plt.grid(True)
plt.legend()

plt.figure(6)
plt.loglog(n,abs(A1),'or',label = 'An')
plt.loglog(n,abs(B1),'ob',label = 'Bn')
plt.loglog(ne, abs(a1),'og',label = 'an estimated')
plt.loglog(ne, abs(b1),'oy',label = 'bn estimated')

plt.title("Loglog plot for coeff of coscos(x)")
plt.xlabel('n')
plt.ylabel('An or Bn')
plt.grid(True)
plt.legend()


plt.show()