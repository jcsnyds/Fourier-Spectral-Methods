import numpy as np
import cupy as cp
import matplotlib.pyplot as plt


N = 256
nu = 5e-3
amp = 1.0
alpha = 0.0
n_f = 4
t_end = 1000.0
CFL = 0.5
plot_every = 2


L  = 2.0 * np.pi
dx = L / N
x  = cp.arange(N) * dx
X, Y = cp.meshgrid(x, x)

k = cp.fft.fftfreq(N, d=1.0 / N)
KX, KY= cp.meshgrid(k, k)
K2 = KX**2 + KY**2
K2_safe = cp.where(K2 == 0.0, 1.0, K2)

k_cut = N / 3
mask = (cp.abs(KX) <= k_cut) & (cp.abs(KY) <= k_cut)

U = amp / (nu * n_f**2)
print(f"U = {U:.2f}, Re = {U / (nu * n_f):.0f}")

F_hat = cp.fft.fft2(-amp * n_f * cp.cos(n_f * Y))

def velocity(w_hat):
    psi_hat = w_hat / K2_safe
    u = cp.fft.ifft2(1j * KY * psi_hat).real
    v = cp.fft.ifft2(-1j * KX * psi_hat).real
    return u, v

def dw_hat_dt(w_hat):
    u, v = velocity(w_hat)
    dw_dx = cp.fft.ifft2(1j * KX * w_hat).real
    dw_dy = cp.fft.ifft2(1j * KY * w_hat).real
    advection_hat = cp.fft.fft2(u * dw_dx + v * dw_dy) * mask
    return -advection_hat - (nu * K2 + alpha) * w_hat + F_hat

def rungekutta_4(w_hat, dt):
    k1 = dw_hat_dt(w_hat)
    k2 = dw_hat_dt(w_hat + 0.5 * dt * k1)
    k3 = dw_hat_dt(w_hat + 0.5 * dt * k2)
    k4 = dw_hat_dt(w_hat + dt * k3)
    return (w_hat + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)) * mask

cp.random.seed(0)
noise_hat = cp.fft.fft2(cp.random.standard_normal((N, N))) * mask * cp.exp(-K2 / 20.0)
noise = cp.fft.ifft2(noise_hat).real
noise = noise / cp.abs(noise).max()

w0 = -U * n_f * cp.cos(n_f * Y) + 0.1 * U * n_f * noise
w_hat = cp.fft.fft2(w0) * mask

w_hat[0, 0] = 0
t = 0.0

plt.ion()
fig, ax = plt.subplots(figsize=(6, 5.5))
w_max = 1.2 * U * n_f
image = ax.imshow(cp.fft.ifft2(w_hat).real.get(), origin="lower", extent=[0, L, 0, L], cmap="RdBu", vmin=-w_max/1, vmax=w_max/1)
fig.colorbar(image, label="vorticity")
ax.set_xlabel("x")
ax.set_ylabel("y")

step = 0
while t < t_end:
    u, v = velocity(w_hat)
    u_max = float(cp.maximum(cp.abs(u).max(), cp.abs(v).max()))
    dt = CFL * dx / u_max

    w_hat = rungekutta_4(w_hat, dt)
    t += dt
    step += 1

    if step % plot_every == 0:
        image.set_data(cp.fft.ifft2(w_hat).real.get()) 
        ax.set_title(f"t = {t:.2f}")
        plt.pause(0.001)
        energy = 0.5 * float(cp.mean(u**2.0 + v**2.0))
        print(f"step {step:6d}  t = {t:7.3f}  dt = {dt:.2e}  energy = {energy:.4f}")

plt.ioff()
plt.show()
