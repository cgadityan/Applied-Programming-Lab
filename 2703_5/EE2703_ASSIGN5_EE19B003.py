import numpy as np
from pylab import *
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3

#Default values for the essentialities of the problem
Nx = 25
Ny = 25
radius = 8
Niter = 1500


#Assigning values to the main features through user given arguments
if len(sys.argv)==2:
	Nx = int(sys.argv[1])
elif len(sys.argv)==3:
	Nx = int(sys.argv[1])
	Ny = int(sys.argv[2])
elif len(sys.argv)==4:
	Nx = int(sys.argv[1])
	Ny = int(sys.argv[2])
	radius = float(sys.argv[3])
elif len(sys.argv)==5:
	Nx = int(sys.argv[1])
	Ny = int(sys.argv[2])
	radius = float(sys.argv[3])
	Niter = int(sys.argv[4])

#Defining phi(Potential)
phi = np.zeros((Ny,Nx))

#Axis defns
x = np.linspace(-Nx/2+1,Nx/2,Nx)
y = np.linspace(-Ny/2+1,Ny/2,Ny)

#Meshgrids
Y,X = np.meshgrid(y,x)

ii = np.where(X**2+Y**2 <= radius*radius)
#Defining central lead potential as ##########  "1"
phi[ii] = 1.0

#COntour plot of potential
figure(1)
plt.contourf(Y,X,phi)
plt.title('Contour plot of potential')
plt.ylabel('Phi vertical')
plt.xlabel('Phi Horizontal')


error = np.zeros((Niter,1))

#Calculating error by running iterations on phi
for k in range(Niter):
	oldphi = phi.copy()
	#Special case of laplace equation 
	phi[1:-1,1:-1]=0.25*(phi[1:-1,0:-2]+ phi[1:-1,2:]+phi[0:-2,1:-1]+phi[2:,1:-1])
	#Boundary Conditions
	phi[1:-1,0]=phi[1:-1,1]
	phi[1:-1,-1]=phi[1:-1,-2]
	phi[0,1:-1]=phi[1,1:-1]
	phi[ii] = 1.0
	#calculating error in each iteration
	error[k] = (abs(phi - oldphi)).max()


itera = range(0,Niter)
#Finding fit for after 500 iterations

#Original error plot
plt.figure(2)
plt.semilogy(itera[::100],error[::100],label = 'Original Error')
plt.legend()
plt.title('Original Error Semilog Plot')
plt.xlabel('Iterations')
plt.ylabel('Error')

#Function to find Fitting values, Coefficients in an arbitrary exponential function 
#And the estimated error through Least squares Method
def Fitting(error,iterations):
	x = len(error)
	Variable_matrix = np.zeros((x,2))
	Variable_matrix[:,0] = 1
	Variable_matrix[:,1] = iterations
	Error_matrix = np.zeros((x,1))
	Error_matrix = np.log(error)
	Estimated_matrix = np.linalg.lstsq(Variable_matrix,Error_matrix,rcond=-1)[0]
	Estimated_coeff = Estimated_matrix
	Estimated_error = np.exp(np.dot(Variable_matrix,Estimated_matrix))
	return Estimated_coeff,Estimated_error

Coeff_above_500, Estimated_error_above_500 = Fitting(error[501:],itera[501:])
Coeff_all, Estimated_error_all = Fitting(error,itera)

print('Estimated coefficients of Fitting overall error is : \nA:{} \nB:{}'.format(Coeff_all[0],Coeff_all[1]))
print('Estimated coefficients of Fitting error above 500 iterations is : \nA:{} \nB:{}'.format(Coeff_above_500[0],Coeff_above_500[1]))

#Estimated error plots overall and above 500 iterations
plt.figure(3)
plt.semilogy(itera[501:],Estimated_error_above_500,label = 'Estimated error ablove 500')
plt.semilogy(itera,Estimated_error_all,label = 'Estimated error all')
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('Error')


#Surface plot of potential through mpl3d toolkit
fig4=figure(4)     # open a new figure
ax=p3.Axes3D(fig4) # Axes3D is the means to do a surface plot
plt.title("The 3-D surface plot of the potential")
surf = ax.plot_surface(Y,X,phi.T,rstride = 1,cstride = 1,cmap = cm.jet,linewidth=0, antialiased=False)

#Defining and Calculating currents 
Jx = np.zeros((Ny,Nx))
Jy = np.zeros((Ny,Nx))

Jx[1:-1, 1:-1] = (phi[1:-1, 0:-2] - phi[1:-1, 2:])*0.5
Jy[1:-1, 1:-1] = (phi[2:, 1:-1] - phi[0:-2, 1:-1])*0.5


#Quiver plot 1 volt points and Current flow
plt.figure(5)
plt.quiver(-Y,-X,Jx,Jy)
plt.scatter(x[ii[0]],y[ii[1]],color='r')
plt.title('Path of current: Quiver plot')

#Conductivity of #copper
sigma = 5.78e-9

Heat = 1/sigma*(Jx**2+Jy**2)
#Plot of Heated regions
plt.figure(6)
plt.contourf(-Y,-X,Heat,cmap='magma')
plt.title('Heat regions')

plt.show()








