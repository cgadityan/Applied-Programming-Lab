from pylab import *
from sympy import symbols
from numpy import pi as pi


def plotting(y, num, lim, tit, yl = 2,step = 512.0):
	Y=fftshift(fft(y))/step
	w=linspace(-64,64,int(step+1));
	w=w[:-1]
	figure(num)
	subplot(2,1,1)
	plot(w,abs(Y),lw=2)
	xlim([-lim,lim])
	ylabel(r"$|Y|$",size=16)
	title('Spectrum of {}'.format(tit))
	grid(True)
	subplot(2,1,2)
	ii=where(abs(Y)>1e-3)
	plot(w[ii],angle(Y[ii]),'go',lw=2)
	xlim([-lim,lim])
	ylim([-yl,yl])
	ylabel(r"Phase of $Y$",size=12)
	xlabel(r"$\omega$",size=12)
	grid(True)


x=rand(128)

X = fft(x)
x_Computed = ifft(X)
plt.figure(0)
t = np.linspace(-40, 40, 129)
t = t[:-1]
plt.plot(t, x, 'b', label='Original $x(t)$', lw=2)
plt.plot(t, abs(x_Computed), 'g', label='Computed $x(t)$', lw=2)
plt.xlabel(r'$t\ \to$')
plt.grid()
plt.legend()
plt.title('Comparison of actual and computed $x(t)$')
maxError = max(np.abs(x_Computed-x))
print('Magnitude of maximum error between actual and computed values of the random sequence:{}'.format(maxError)) 

# DFT of sin(5t)

x=linspace(0,2*pi,129);
x=x[:-1]
y=sin(5*x)
Y=fftshift(fft(y))/128.0
w=linspace(-64,63,128)
figure(3)
subplot(2,1,1)
plot(w,abs(Y),lw=2)
xlim([-10,10])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of $\sin(5t)$")
grid(True)
subplot(2,1,2)
plot(w,angle(Y),'ro',lw=2)
ii=where(abs(Y)>1e-3)
plot(w[ii],angle(Y[ii]),'go',lw=2)
xlim([-10,10])
ylabel(r"Phase of $Y$",size=12)
xlabel(r"$k$",size=12)
grid(True)



x=linspace(-4*pi,4*pi,513);
x=x[:-1]

#DFT of cos(10t)*(1 +0.1cos(t))

y=(1+0.1*cos(x))*cos(10*x)
tit = str("(1+0.1*cos(t))*cos(10*t)")
plotting(y,4,20,tit)

#DFT of sin^3(t)

y= sin(x)**3
tit = str("sin(t)**3")
plotting(y,5,15,tit)


#DFT of cos^3(t)

y=cos(x)**3
tit = str("cos(t)**3")
plotting(y,6,15,tit)


#DFT of cos(20t+ 5cos(t))

y= cos(20*x + 5*cos(x))
tit= str('cos(20*x + 5*cos(x))')
plotting(y,7,40,tit)


## Spectrum of Gaussian

### Phase and Magnitude of estimated Gaussian Spectrum
# I have chosen a window from [-8pi, 8pi] and took 512 points in that interval
t =  linspace(-8*pi, 8*pi, 513)
t = t[:-1]
xTrueGaussian = exp(-(t**2)/2)
Y = fftshift(fft(ifftshift(xTrueGaussian)))*8/512.0
fig6 = plt.figure(8)
fig6.suptitle(r'Comparison of spectrum of $e^{-\frac{t^2}{2}}$')
YMag = abs(Y)
YPhase = angle(Y)
absentFreqs = where(YMag < 1e-3)
YPhase[absentFreqs] = 0
w = linspace(-40, 40, 513)
w = w[:-1]
subplot(221)
plot(w, YMag, lw=2)
xlim([-10, 10])
ylabel(r'$\|Y\|$')
title("Estimated Spectrum")
grid()
subplot(223)
plot(w, YPhase, 'ro', lw=2)
xlim([-10, 10])
ylim([-pi, pi])
ylabel(r'$\angle Y$')
xlabel(r'$k\ \to$')
grid()

### Phase and Magnitude of true Gaussian spectrum
trueY = exp(-(w**2)/2)/sqrt(2*pi)
trueYMag = abs(trueY)
trueYPhase = angle(trueY)
subplot(222)
plot(w, trueYMag)
xlim([-10, 10])
title("True Spectrum")
grid()
subplot(224)
plot(w, trueYPhase, 'ro')
xlim([-10, 10])
ylim([-pi, pi])
xlabel(r'$k\ \to$')
grid()



show()