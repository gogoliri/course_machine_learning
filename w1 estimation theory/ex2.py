# Author: Khoa Pham Dinh
# email: khoa.phamdinh@tuni.fi
import matplotlib.pyplot as plt
import numpy as np

# The submitted image is plotted with the parameters below
# Form a sinusoidal signal
phi = 0.6090665392794814
A = 0.6669548209299414
sigmaSq = 1  # 1.2
N = 160
n = np.arange(N)
f0 = 0.06752728319488948
x0 = A * np.cos(2 * np.pi * f0 * n + phi)
nf = 100000  # Number of f for estimation
# Add noise to the signal
# random error multiply with sigma, not sigma square
x = x0 + np.sqrt(sigmaSq) * np.random.randn(x0.size)
# Estimation parameters
A_hat = A * 1
phi_hat = phi * 1
fRange = np.linspace(0, 0.5, nf)

# 1
# SSE
# Initialize s[n] sequence
Sn = np.zeros((nf, N))
# Calculate values of s[n]
for i in range(nf):
    Sn[i] = A_hat * np.cos(2 * np.pi * fRange[i] * n + phi_hat)

# Calculate Loss for difference values of f0
L = np.sum((x - Sn) ** 2, 1)

# Find the index of estimated f0 for sse method
# which makes Loss the smallest
f_est_index = np.argmin(L)
# min Loss 
Lsse = L[f_est_index]

# Likelihood
# Calculate likelihood for different value of f
lkh = (1 / np.sqrt(2 * np.pi * sigmaSq)) * np.exp((-1 / (2 * sigmaSq)) * np.sum((x - Sn) ** 2, 1))

# Take the index of f_est that return the maximum likelihood
f_lkh_index = np.argmax(lkh)
# The maximum likelihood value
lkh_max = lkh[f_lkh_index]

# Print true f0
print(f"True f0: {f0} \n")
# Print MSE from SSE
print(f"Min Mean Square Error: {Lsse / N} \n")
print(f"Estimated f0 with SSE method: {fRange[f_est_index]} \n")
# Print Likelihood
print(f"Maximum Likelihood {lkh_max} \n")
print(f"Estimated f0 with Likelihood method: {fRange[f_lkh_index]} \n")

fig, axs = plt.subplots(2, 2)
fig.set_size_inches(18.5, 10.5, forward=True)

# Plot the function estimated f0 and true sinusoid function
axs[1, 1].plot(n, x0)
axs[1, 1].plot(n, Sn[f_est_index], "r--")
axs[1, 1].set_title(f"True f0 = {f0} (blue) and estimated f0 = {fRange[f_est_index]} (red)")

# Plot the square error
axs[0, 0].plot(fRange, L)
axs[0, 0].set_title("Square Error")

# Plot the likelhood
axs[0, 1].plot(fRange, lkh)
axs[0, 1].set_title("Likelihood")

# Plot the sinusoid and the measured signal
axs[1, 0].plot(n, x0)
axs[1, 0].plot(n, x, "go")
axs[1, 0].set_title("Sinusoid function (blue) and noisy sample (green)")
plt.show()
fig.savefig('ex2plot.jpg', dpi=100)
# Realization 1: With small number of f (nf = 100) and very small variance (sigma square = 0.001), the likelihood method
# fails to estimate f0

# Realization 2: With sigma square at 4 (others parameter is the same as a and b), the method fail at detecting f0 (
# sometimes)

# Realization 3: I found that sigma square =2.75 is the maximum noise at which our estimation is still work robustly (
# systematically)

# Realization 4: When add error to A and phi, at sigma square = 2.75 the method can still detects f0
# when I at 50% error for both A hat and phi hat most of the times
