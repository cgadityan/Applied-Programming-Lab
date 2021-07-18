import numpy as np
import matplotlib.pyplot as plt
import sys 
from pylab import *
import pandas as pd

#Default values
n = 100  # spatial grid size.
M = 5    # number of electrons injected per turn.
nk = 500 # number of turns to simulate.
u0 = 5   # threshold velocity.
p = 0.25 # probability that ionization will occur
Msig = 2 # Std dev of distribution to insert electrons

if len(sys.argv)==7:
    n=sys.argv[0]  # spatial grid size.
    M=sys.argv[1]    # number of electrons injected per turn.
    nk=sys.argv[2] # number of turns to simulate.
    u0=sys.argv[3]  # threshold velocity.
    p=sys.argv[4] # probability that ionization will occur
    Msig=sys.argv[5] #deviation of elctrons injected per turn

if len(sys.argv)==6:
    n=sys.argv[0]  
    M=sys.argv[1]   
    nk=sys.argv[2] 
    u0=sys.argv[3]  
    p=sys.argv[4]
   
if len(sys.argv)==5:
    n=sys.argv[0]  
    M=sys.argv[1]   
    nk=sys.argv[2] 
    u0=sys.argv[3]  

if len(sys.argv)==4:
    n=sys.argv[0]  
    M=sys.argv[1]   
    nk=sys.argv[2]

if len(sys.argv)==3:
    n=sys.argv[0]  
    M=sys.argv[1]   
  
if len(sys.argv)==2:
    n=sys.argv[0]  
 

#Defining arrays for postion,displacement and Velocity
xx = np.zeros((n*M))
u = np.zeros((n*M))
dx = np.zeros((n*M))


# Empty lists to get...
I = []	#Intensity
X = []	#Position
V = []	#Velocity

#loop
for k in range(nk):
	ii = where(xx>0)

	#Updating Position, Dispalcement and Velocity
	dx[ii] = u[ii] + 0.5
	xx[ii] += dx[ii]
	u[ii] += 1

	#Finding Positions of electrons with postion above n and collision velocity
	iix = where(xx > n)
	kk = where(u >= u0)

	#assigning zero parameters for passed out electrons
	xx[iix] = 0
	u[iix] = 0
	dx[iix] = 0

	#Getting the no. of electrons that will actually collide and lose energy by probability provided
	ll = np.where(rand(len(kk[0])) <= p)
	#Getting the indices
	kl = kk[0][ll]
	#Zero velocity after collision
	u[kl] = 0

	#Postion of collision that happened, using a random no.
	rho = rand(len(kl))
	xx[kl] = xx[kl] - dx[kl]*rho

	#Intensity 
	I.extend(xx[kl].tolist())
	#
	#No. of electrons inserted based on the space that is available
	m = int(rand()*Msig+M)

	i0 = where(xx == 0)		#Empty spaces
	nv=(min(n*M-len(i0),m)) #if no empty spaces are left
	xx[i0[:nv]]=1 			#inject the new electrons
	u[i0[:nv]]=0 		#with velocity zero
	dx[i0[:nv]]=0		#and displacement zero

	#Add to velocities and Positions
	X.extend(xx.tolist())
	V.extend(u.tolist())



#Plots that are required


figure(0)
plt.hist(X, bins=np.arange(0,101,0.5), rwidth=0.7)
plt.title('Population plot of X ')
plt.xlabel('$x$')
plt.ylabel('Number of electrons')


figure(1)
histogram = plt.hist(I, bins=np.arange(0,101,0.5), rwidth=0.7)
plt.title('Intensity histogram ')
plt.xlabel('x')
plt.ylabel('Intensity')

y = histogram[0]
bins = histogram[1]

xpos = 0.5*(bins[1:-2][::2]+bins[3:][::2])

print("\tIntensity Data")

data = c_[xpos,y[1:][::2]]
df = pd.DataFrame(data , columns=['Xposition','Count'])

print(df)

plt.figure(2)
plt.plot(X,V,'bo')
plt.title('Electron Phase Space')
plt.xlabel('Position-X')
plt.ylabel('Velocity-V')

plt.show()

