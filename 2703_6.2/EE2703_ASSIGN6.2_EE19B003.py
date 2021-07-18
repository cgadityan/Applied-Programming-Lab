import numpy as np
import scipy.signal as sp
from pylab import *
import matplotlib.pyplot as plt

#################  P1 ##################
def solve_x_t(d):
	x_s = sp.lti([1,d],np.polymul([1,0,2.25], np.polyadd(np.polymul([1, d],[1, d]),[2.25])))
	t_s = np.linspace(0,50,1000)
	t,x = sp.impulse(x_s,None,t_s )
	return t,x

t,x = solve_x_t(0.5)
figure(1)
plt.plot(t,x)
plt.grid()
plt.xlabel('t')
plt.ylabel('x(t)')
plt.title('Plot for problem 1, Impulse response, decay = 0.5' )

#################  P2 ###################
t,x = solve_x_t(0.05)
figure(2)
plt.plot(t,x)
plt.grid()
plt.xlabel('t')
plt.ylabel('x(t)')
plt.title('Plot for problem 2, Impulse response, decay = 0.05' )


#################  P3 ###################
x_f_s = sp.lti([1],[1,0,2.25])
t_v = linspace(0,50,10000)

def f_t(t,w):
	return np.exp(-0.05*t)*np.cos(w*t)

figure(3)
for w in np.arange(1.4,1.6,0.05):
	t,h,svec = sp.lsim2(x_f_s,U = f_t(t_v,w),T = t_v)
	plt.plot(t,h,label = 'frequency(w) = {}'.format(w))

plt.grid()
plt.xlabel('t')
plt.ylabel('h(t) Output with various Freq')
plt.title('Plot for problem 3, omega = 1.4 to 1.6, decay = 0.05' )
plt.legend()

#################  P4 ###################
x_4 = sp.lti([1,0,2],[1,0,3,0])
y_4 = sp.lti([2],[1,0,3,0])

figure('4a')
t,x = sp.impulse(x_4,None, np.linspace(0,20,2000))
plt.plot(t,x)
plt.grid()
plt.xlabel('t')
plt.ylabel('x(t)')
plt.title('Plot for problem 4, impulse response x(t) ' )

figure('4b')
t,y = sp.impulse(y_4,None, np.linspace(0,20,2000))
plt.plot(t,y)
plt.grid()
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Plot for problem 4, impulse response y(t) ' )


#################  P5 ####################
r_l_c = sp.lti([1e12],[1,1e8,1e12])

w,S,phi = r_l_c.bode()
figure('5a')
plt.semilogx(w,S)
plt.grid()
plt.xlabel('log(w)')
plt.ylabel('Abs(rlc(s))')
plt.title('Plot for problem 5, Magnitude plot of RLC system' )


figure('5b')
plt.semilogx(w,phi)
plt.grid()
plt.xlabel('log(w)')
plt.ylabel('Argument(rlc(s))')
plt.title('Plot for problem 5, Phase plot of RLC system' )

################  P6 ####################

t_rlc = np.arange(0,30e-6,1e-8)
v_i_st = np.cos(1e3*t_rlc) - np.cos(1e6*t_rlc)
t_rlc_2 = np.arange(0,0.01,1e-7)
v_i_lt = np.cos(1e3*t_rlc_2) - np.cos(1e6*t_rlc_2)


t,v_o,svec = sp.lsim(r_l_c, v_i_st, t_rlc)
figure('6a')
plt.plot(t,v_o)
plt.grid()
plt.xlabel('t')
plt.ylabel('v_o(t)')
plt.title('Plot for problem 6, Output of RLC system(Short term)' )


t,v_o,svec = sp.lsim(r_l_c, v_i_lt, t_rlc_2)
figure('6b')
plt.plot(t,v_o)
plt.grid()
plt.xlabel('t')
plt.ylabel('v_o(t)')
plt.title('Plot for problem 6, Output of RLC system(Long term)' )

plt.show()