import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Q1
# Pseudo code in the report

#Q2
"""
Defining Meshgrid
"""
# Permeability of vaccum
mu0 = 4*np.pi*1e-7
# Radius of the loop
r = 0.1#(m)
k = 1/r
# 3x3x1000 mesh
x = np.linspace(-0.00999,0.01,3)
y = np.linspace(-0.01,0.01,3)
z = np.linspace(0.01,10,1000)
# indexing to make sure there is no exchange while defining r__mesh vector
X,Y,Z = np.meshgrid(x,y,z,indexing = 'ij')

#Q3
"""
Breaking the loop into sections, Defining 
and Plotting Current elements in the midpoints of
positional elements in the loop 
"""
# Breaking loop into 100 sections
sec = 100
phi = np.array(np.linspace(0, 2*np.pi, sec, endpoint = False))
print(r'Shape of phi :',np.shape(phi))
# Plotting current elements
Io = 4*np.pi/mu0
r_ = np.array([r*np.cos(phi),r*np.sin(phi)]).T
print("Shape of vector r' ", np.shape(r_))

I = Io*np.array([-np.sin(phi)*np.cos(phi), np.cos(phi)*np.cos(phi)]).T

Ix = 0.5*(I[0:-1,0] + I[1:,0])
Iy = 0.5*(I[0:-1,1] + I[1:,1]) 

# r vector
plt.figure(0)
plt.scatter(r_[:,0],r_[:,1])
plt.title('Loop section points')
plt.xlabel("x $\longrightarrow$")
plt.ylabel("y $\longrightarrow$")
plt.grid()

plt.figure(1)
plt.quiver(r_[:-1,0],r_[:-1,1],Ix,Iy)
plt.xlabel("x $\longrightarrow$")
plt.ylabel("y $\longrightarrow$")
plt.title("Quiver Plot of Current in the Loop")
plt.grid()

#Q4
"""
Defining Vectors r(r__mesh) , r'(r_) , dl
"""
r__mesh = np.array((X,Y,Z))
print('Shape of the vector r containing all points:',r__mesh.shape)


dl = (2*np.pi*r/sec)*np.array([-np.sin(phi),np.cos(phi)]).T
print('Shape of dl vector: ',dl.shape)

#Q5
"""
calc(l) function that computes either Rijkl or Aijkl(Commented)
"""

def calc(l):
	r_l = np.array((r_[l,0],r_[l,1],0))

	Rijkl = np.sqrt((r__mesh[0,:,:,:]- r_l[0])**2 + (r__mesh[1,:,:,:]- r_l[1])**2 + r__mesh[2,:,:,:]**2)
	
	#Q6 & Q7
	"""
	Computing Aijkl and Adding it to A vector
	"""
	#Aijkl = np.cos(phi[l])*np.exp(-1j*k*Rijkl)/Rijkl
	#return Aijkl

	return Rijkl


A_x = np.zeros(X.shape)
A_y = np.zeros(Y.shape)


for l in range(100):
	R = calc(l)

	Ax = np.cos(phi[l])*dl[l][0]*np.exp(-1j*k*R)/R
	A_x = A_x + Ax
	Ay = np.cos(phi[l])*dl[l][1]*np.exp(-1j*k*R)/R
	A_y = A_y + Ay

	#Q6 & Q7

	# If Aijkl is returned
	#Aijkl = calc(l)

	#A_x = A_x + Aijkl*dl[l][0]
	#A_y = A_y + Aijkl*dl[l][1]

Rijkl = calc(0)

#Vector Rijkl = Distance of an element in loop from 9000 meshgrid points 3*3*1000

print('Shape of Rijkl from calc function:',Rijkl.shape)
#Q8
"""
Computation of Magnetic field by using the vectorized form of A
"""
# In SI Units
B_z = (A_y[-1,1,:] - A_x[1,-1,:] - A_y[0,1,:] + A_x[1,0,:])/(4*1e-4)


print('Shape of B(z) containing Magnetic fields at all the points on z-axis:',B_z.shape)

#Q9
"""
Loglog plot of Magnetic Field
"""
plt.figure(2)
plt.loglog(z,np.abs(B_z),label = 'Bz(z)')
plt.title('Magnetic field plot')
plt.ylabel(r"$B_z(z)$ $\longrightarrow$")
plt.xlabel("z $\longrightarrow$")
plt.grid()

#Q10
"""
Fitting the Field into the function c*z^(b)
"""
A=np.hstack([np.ones(len(B_z[300:]))[:,np.newaxis],np.log(z[300:])[:,np.newaxis]])

log_c,b = np.linalg.lstsq(A,np.log(np.abs(B_z[300:])),rcond = -1)[0]
c = np.exp(log_c)

#Adding the fitted graph along with the Field plot
y = c*np.power(z,b)
plt.loglog(z,y,label= r'$B_z$ Fitted function')
plt.legend()
print('The values of b and c in cz^(b) fitted in Bz are : \nb = {}\nc = {}'.format(b,c))

plt.show()