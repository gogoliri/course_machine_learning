# Author: Khoa Pham Dinh
# ID: 050359620
# email: khoa.phamdinh@tuni.fi
import numpy as np
import matplotlib.pyplot as plt

# define parameters
A = 1
f0 = 0.1 
phi = 0

F_s = 100 # sample rate per second 
T = 9    # total time in seconds
n = np.arange(T * F_s) # discrete times n=0,1,.., T*F_s-1
f = 0.1 # frequency
sigma_w = 0.5 # variance

# signal without noise
x = np.cos(2 * np.pi * 0.1 * n)

# some zeros at the beginning and end
x[:5*F_s] = 0
x[6*F_s:] = 0

# add noise
x_noisy = x +  sigma_w * np.random.randn(*x.shape)

#filter
L = 100 # Filter length
n = np.arange(L)

# Filter cos(2pi*0.1*n)
s = np.cos(2*np.pi*0.1*n)

# General filter
si = np.exp(-2 * np.pi * 1j * 0.1* n )

#convolution with real value window
y = np.convolve(x_noisy, np.flip(s), 'same')

# convolution with complex value window
yi = np.convolve(x_noisy, np.flip(si), 'same')

# Plot x, x_noisy and 2 filters
fig, axs = plt.subplots(2,2)
fig.set_size_inches(18.5, 10.5, forward=True)

# Plot x_clean
axs[0,0].autoscale(enable=True, axis='x', tight=True)
axs[0,0].set_title("x_clean")
axs[0,0].plot(x)

# Plot x_noisy
axs[0,1].autoscale(enable=True, axis='x', tight=True)
axs[0,1].set_title("x_noisy")
axs[0,1].plot(x_noisy)

# Plot cos filter
axs[1,0].autoscale(enable=True, axis='x', tight=True)
axs[1,0].set_title("Filter cos(2pi*0.1*n) on noisy data")
axs[1,0].plot(y)

#Plot general complex filter
axs[1,1].autoscale(enable=True, axis='x', tight=True)
axs[1,1].set_title("Absolute value(blue) of filter exp(-2pi*i*n) on noisy data")
axs[1,1].plot(abs(yi))
# Uncomment code below to plot Real and Im of the convoluted yi
#axs[1,1].plot(np.real(yi),'g--')
#axs[1,1].plot(np.imag(yi),'r--')

# Save image
fig.savefig('ex3plot.jpg', dpi=100)





