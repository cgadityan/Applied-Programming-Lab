from sympy import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp

PI = np.pi
t = np.linspace(0,0.01,int(1e6))
s = symbols('s')

#CIRCUIT 1
def Lowpass(R1,R2,C1,C2,G):
	s=symbols('s')
	A=Matrix([[0,0,1,-1/G],
			  [-1/(1+s*R2*C2),1,0,0],
			  [0,-G,G,1],
			  [-1/R1-1/R2-s*C1,1/R2,0,s*C1]])
	b=Matrix([0,0,0,-1/R1])
	V = A.inv()*b
	return V[3]

def sympy_to_LTI(H):
    num, den = H.as_numer_denom()
    numer = Poly(num, s)
    denom = Poly(den, s)
    numeratorCoeffs = numer.all_coeffs()
    denominatorCoeffs = denom.all_coeffs()
    for i in range(len(numeratorCoeffs)):
        x = float(numeratorCoeffs[i])
        numeratorCoeffs[i] = x
    for j in range(len(denominatorCoeffs)):
        x = float(denominatorCoeffs[j])
        denominatorCoeffs[j] = x
    return sp.lti(numeratorCoeffs, denominatorCoeffs)


Vi = np.heaviside(t,1)*(np.sin(2e3*PI*t) + np.cos(2e6*PI*t))
plt.figure('Vi_t')
plt.plot(t, Vi)
plt.title('Input signal')
plt.xlabel(r'$t\ \to$')
plt.ylabel(r'$V_i(t)\ \to$')
plt.xlim(0, 1e-3)
plt.grid()

s = symbols('s')

H1 = Lowpass(10000,10000,1e-9,1e-9,1.586)

print(H1)

# Obtaining magnitude plot using "lambdify"

# print('G=1000')
# w = logspace(0,8,801)

# ss = 1j*w

# f_c1 = lambdify(s,H1,"numpy")
# figure(0)
# #Magnitude response of output
# loglog(w,abs(f_c1(ss)))
# grid(True)

f_c1_s = sympy_to_LTI(H1)

# Bode plots for Lowpass Filter
w_c1,mag_c1,phi_c1 = sp.bode(f_c1_s, np.linspace(1, 1e6, int(1e6)))

plt.figure('1_Magnitude')
plt.semilogx(w_c1,mag_c1)
plt.title('Magnitude plot for lowpass')
plt.xlabel(r'$\omega \ \to$')
plt.ylabel(r'$20log(\|H(j\omega)\|)$')

plt.figure('1_Phase')
plt.semilogx(w_c1,phi_c1)
plt.title('Phase plot for lowpass')
plt.xlabel(r'$\omega \ \to$')
plt.ylabel(r'$\angle H(j\omega)$')

# Output for Vi in lowpass
t,c1_vi,svec = sp.lsim(f_c1_s,Vi,t)
plt.figure('1_V0')
plt.plot(t, c1_vi)
plt.title('Output of Vi in Lowpass')
plt.xlabel(r'$t\ \to$')
plt.ylabel(r'$V_o(t)\ \to$')
plt.xlim(0,1e-3)
plt.grid()

# Step response of Lowpass Filter
t,c1 = sp.step(f_c1_s, None, t)
plt.figure('1_step_response')
plt.title('Step response of Lowpass filter')
plt.xlabel(r'$t\ \to$')
plt.ylabel(r'$V_o(t)\ \to$')
plt.plot(t, c1)
plt.xlim(0, 1e-3)
plt.grid()


#CIRCUIT 2


def HighPass(R1, R3, C1, C2, G):
    s = symbols('s')
    A = Matrix([[0, -1, 0, 1/G],
                [s*C2*R3/(s*C2*R3+1), 0, -1, 0],
                [0, G, -G, 1],
                [-s*C2-1/R1-s*C1, 0, s*C2, 1/R1]])

    b = Matrix([0,
                0,
                0,
                -s*C1])
    return (A.inv()*b)[3]

H2 = HighPass(10000, 10000, 1e-9, 1e-9, 1.586)
print(H2)
f_c2_s = sympy_to_LTI(H2)

# Bode plots for Highpass Filter
w_c2,mag_c2,phi_c2 = sp.bode(f_c2_s,np.linspace(1,1e6,int(1e6)))

plt.figure('2_Magnitude')
plt.semilogx(w_c2,mag_c2)
plt.title('Magnitude plot for highpass')
plt.xlabel(r'$\omega \ \to$')
plt.ylabel(r'$20log(\|H(j\omega)\|)$')

plt.figure('2_Phase')
plt.semilogx(w_c2,phi_c2)
plt.title('Phase plot for highpass')
plt.xlabel(r'$\omega \ \to$')
plt.ylabel(r'$\angle H(j\omega)$')

# Output of Vi in Highpass

t,c2,svec = sp.lsim(f_c2_s,Vi,t)
plt.figure('2_V0')
plt.plot(t,c2)
plt.title('Output of Vi in Highpass')
plt.xlabel(r'$t\ \to$')
plt.ylabel(r'$V_o(t)\ \to$')
plt.xlim(0,1e-4)
plt.grid()

#Question 4

Vi_4_lf = np.exp(-t)*np.sin(2*PI*t) # decay coefficient = 1

Vi_4_hf = np.exp(-100*t)*np.sin(2e6*PI*t) # decy coefficient = 100

t, c2_d,svec = sp.lsim(f_c2_s, Vi_4_lf, np.linspace(0,1e-4,int(1e6)))

plt.figure('c2_damped_input_lowfreq')
plt.plot(t,c2_d)
plt.title('Circuit_2_damped_input_lowfreq')
plt.xlabel(r'$t\ \to$')
plt.ylabel(r'$V_o(t)\ \to$')
plt.grid()

t, c2_d,svec = sp.lsim(f_c2_s, Vi_4_hf, np.linspace(0,1e-4,int(1e6)))

plt.figure('c2_damped_input_highfreq')
plt.plot(t,c2_d)
plt.title('Circuit_2_damped_input_highfreq')
plt.xlabel(r'$t\ \to$')
plt.ylabel(r'$V_o(t)\ \to$')
plt.grid()

#Question 5
#Vi = 1/s (Step response of Highpass filter)
t, c2_u = sp.step(f_c2_s ,None,t)

plt.figure('2_step_response')
plt.plot(t,c2_u)
plt.title(r'Step Response of Highpass filter')
plt.xlabel(r'$t\ \to$')
plt.ylabel(r'$V_o(t)\ \to$')
plt.xlim(0,1e-4)
plt.grid()

plt.show()