import cmath
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
from scipy.linalg import lstsq
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


PI = np.pi
figNum = 0
showAll = True

# Functions used
def hamming_window(n):
    '''
                    0.54 + 0.46*cos(2πn/(N−1)), |n| <= (N-1)/2
        w[n] =
                    0, otherwise
    '''

    N = n.size
    window = np.zeros(N)
    window = 0.54 + 0.46*np.cos((2*PI*n)/(N-1))
    return fft.fftshift(window)

def plot_signal(t, x, figTitle, style='b-', blockFig=False, showFig=True):
    global figNum
    plt.figure(figNum)
    plt.title(figTitle)
    plt.grid()
    plt.plot(t, x, style)
    if(showFig):
        plt.show(block=blockFig)
    figNum+=1


def plot_spectrum(figTitle, w, Y, magStyle='b-', phaseStyle='ro', xLimit=None, yLimit=None, showFig=False):
    global figNum
    plt.figure(figNum)
    plt.suptitle(figTitle)
    plt.subplot(211)
    plt.grid()
    plt.plot(w, abs(Y), magStyle, lw=2)
    plt.ylabel(r"$\|Y\|$")
    if (xLimit):
        plt.xlim(xLimit)
    if (yLimit):
        plt.ylim(yLimit)
    plt.subplot(212)
    plt.grid()
    plt.plot(w, np.angle(Y), phaseStyle, lw=2)
    plt.xlim(xLimit)
    plt.ylabel(r"$\angle Y$")
    plt.xlabel(r"$\omega\ \to$")
    if(showFig):
        plt.show()
    figNum+=1

# Example 1 - sin(sqrt(2)t)

    ## Without windowing

t = np.linspace(-PI, PI, 65)[:-1]
dt = t[1]-t[0]
fmax = 1/dt
y = np.sin(cmath.sqrt(2)*t)
y[0] = 0
y = fft.fftshift(y)
Y = fft.fftshift(fft.fft(y))/64.0
w = np.linspace(-PI*fmax, PI*fmax, 65)[:-1]
plot_signal(t, y, r"$sin(\sqrt{2}t)$",style = 'or')
plot_spectrum(r"Spectrum of $sin(\sqrt{2}t)$", w, Y, xLimit=[-10, 10], showFig=showAll)

    ## Windowing with Hamming Window

t = np.linspace(-PI, PI, 65)[:-1]
dt = t[1]-t[0]
fmax = 1/dt
n = np.arange(64)
y = np.sin(cmath.sqrt(2)*t) * hamming_window(n)
y[0] = 0
y = fft.fftshift(y)
Y = fft.fftshift(fft.fft(y))/64.0
w = np.linspace(-PI*fmax, PI*fmax, 65)[:-1]
plot_spectrum(r"Spectrum of $sin(\sqrt{2}t) * w(t)$", w, Y, xLimit=[-8, 8], showFig=showAll)


# Question 2 - spectrum of (cos(0.86 t))**3

    ## Without windowing

t = np.linspace(-4*PI, 4*PI, 257)[:-1]
dt = t[1]-t[0]
fmax = 1/dt
y = np.cos(0.86*t)**3
y[0] = 0
y = fft.fftshift(y)
Y = fft.fftshift(fft.fft(y))/256.0
w = np.linspace(-PI*fmax, PI*fmax, 257)[:-1]
plot_spectrum(r"Spectrum of $cos^3(0.86t)$", w, Y, xLimit=[-8, 8], showFig=showAll)

    ## Windowing with Hamming Window

t = np.linspace(-4*PI, 4*PI, 257)[:-1]
dt = t[1]-t[0]
fmax = 1/dt
n = np.arange(256)
y = (np.cos(0.86*t))**3 * hamming_window(n)
y[0] = 0
y = fft.fftshift(y)
Y = fft.fftshift(fft.fft(y))/256.0
w = np.linspace(-PI*fmax, PI*fmax, 257)[:-1]
plot_spectrum(r"Spectrum of $cos^3(0.86t) * w(t)$", w, Y, xLimit=[-8, 8], showFig=showAll)


def estimate_w_delta(w, wo, Y, do, pow=2):
    wEstimate = np.sum(abs(Y)**pow * abs(w))/np.sum(abs(Y)**pow) # weighted average
    print("wo = {:.03f}\t\two (Estimated) = {:.03f}".format(wo, wEstimate))

    t = np.linspace(-PI, PI, 129)[:-1]
    y = np.cos(wo*t + do)

    c1 = np.cos(wEstimate*t)
    c2 = np.sin(wEstimate*t)
    A = np.c_[c1, c2]
    vals = lstsq(A, y)[0]
    dEstimate = np.arctan2(-vals[1], vals[0])
    print("do = {:.03f}\t\tdo (Estimated) = {:.03f}".format(do, dEstimate))


# Question 3 - Estimation of w, d in cos(wt + d)
wo = 1.35
d = PI/2

print("Question 3:")
t = np.linspace(-PI, PI, 129)[:-1]
trueCos = np.cos(wo*t + d)
fmax = 1.0/(t[1]-t[0])
n = np.arange(128)
y = trueCos.copy()*hamming_window(n)
y = fft.fftshift(y)
Y = fft.fftshift(fft.fft(y))/128.0
w = np.linspace(-PI*fmax, PI*fmax, 129)[:-1]
plot_spectrum(r"Spectrum of $cos(\omega_o t + \delta) \cdot w(t)$", w, Y, xLimit=[-4, 4], showFig=showAll)
estimate_w_delta(w, wo, Y, d, pow=1)

# Question 4 - Estimation of w, d in noisy cos(wt + d)

print("\nQuestion 4:")
trueCos = np.cos(wo*t + d)
noise = 0.1*np.random.randn(128)
n = np.arange(128)
y = (trueCos + noise)*hamming_window(n)
fmax = 1.0/(t[1]-t[0])
y = fft.fftshift(y)
Y = fft.fftshift(fft.fft(y))/128.0
w = np.linspace(-PI*fmax, PI*fmax, 129)[:-1]
plot_spectrum(r"Spectrum of $(cos(\omega_o t + \delta) + noise) \cdot w(t)$", w, Y, xLimit=[-4, 4], showFig=showAll)
estimate_w_delta(w, wo, Y, d, pow=1)

# Question 5 - DFT of chirp

# chirp function used
def chirp(t):
    return np.cos(16*(1.5*t + (t**2)/(2*PI)))

t = np.linspace(-PI, PI, 1025)[:-1]
x = chirp(t)
plot_signal(t, x, r"$cos(16(1.5 + \frac{t}{2\pi})t)$")
fmax = 1.0/(t[1]-t[0])
X = fft.fftshift(fft.fft(x))/1024.0
w = np.linspace(-PI*fmax, PI*fmax, 1025)[:-1]
plot_spectrum(r"DFT of $cos(16(1.5 + \frac{t}{2\pi})t)$", w, X, 'b-', 'r.-', [-75, 75], showFig=showAll)

n = np.arange(1024)
x = chirp(t)*hamming_window(n)
plot_signal(t, x, r" $cos(16(1.5 + \frac{t}{2\pi})t) \cdot w(t)$")
X = fft.fftshift(fft.fft(x))/1024.0
plot_spectrum(r"DFT of $cos(16(1.5 + \frac{t}{2\pi})t) \cdot w(t)$", w, X, 'b-', 'r.-', [-75, 75], showFig=showAll)

# Question 6 - Time evolution of DFT of chirp signal

# calculates DFT of x, taking every batchSize samples
def STFT(x, t, batchSize=64):
    t_batch = np.split(t, 1024//batchSize)
    x_batch = np.split(x, 1024//batchSize)
    X = np.zeros((1024//batchSize, batchSize), dtype=complex)
    for i in range(1024//batchSize):
        X[i] = fft.fftshift(fft.fft(x_batch[i]))/batchSize
    return X

# plots the STFT
def plot3DSTFT(t, w, X, colorMap=cm.viridis, showFig=showAll):
    global figNum

    t = t[::64]
    w = np.linspace(-fmax*PI,fmax*PI,65)[:-1]
    t, w = np.meshgrid(t, w)

    fig = plt.figure(figNum)
    ax = fig.add_subplot(211, projection='3d')
    surf = ax.plot_surface(w, t, abs(X).T, cmap=colorMap)
    fig.colorbar(surf)
    plt.xlabel(r"Frequency $\to$")
    plt.ylabel(r"Time $\to$")
    plt.title(r"Magnitude $\|Y\|$")

    ax = fig.add_subplot(212, projection='3d')
    surf = ax.plot_surface(w, t, np.angle(X).T, cmap=colorMap)
    fig.colorbar(surf)
    plt.xlabel(r"Frequency $\to$")
    plt.ylabel(r"Time $\to$")
    plt.title(r"Angle $\angle Y$")
    if showFig:
        plt.show()

    figNum+=1

x = chirp(t)
X = STFT(x, t)
plot3DSTFT(t, w, X, colorMap=cm.plasma)

x = chirp(t)*hamming_window(np.arange(1024))
X = STFT(x, t)
plot3DSTFT(t, w, X)