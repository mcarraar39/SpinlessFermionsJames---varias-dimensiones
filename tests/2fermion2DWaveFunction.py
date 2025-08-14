import numpy as np
import matplotlib.pyplot as plt

# --- parámetros del gráfico -------------------------------------------------
grid_range = 4.0      # rango de coordenadas  −R … R
grid_size  = 400      # resolución (grid_size × grid_size puntos)
x2_fixed, y2_fixed = 0.0, 0.0   # fijamos el segundo fermión en el origen
# ---------------------------------------------------------------------------

# grid uniforme para (x1,y1)
x = np.linspace(-grid_range, grid_range, grid_size)
y = np.linspace(-grid_range, grid_range, grid_size)
X1, Y1 = np.meshgrid(x, y, indexing='ij')

# coordenadas del segundo fermión (constantes)
X2 = x2_fixed
Y2 = y2_fixed

# --- función de onda --------------------------------------------------------
#  Ψ(x1,y1,x2,y2) = √2 / π · exp(−(x1² + y1² + x2² + y2²)/2) · (y2 − y1)
psi = (np.sqrt(2) / np.pi) * np.exp(
        -0.5 * (X1**2 + Y1**2 + X2**2 + Y2**2)
      ) * (Y2 - Y1)

prob_density = psi**2       # |Ψ|²

# --- dibujo -----------------------------------------------------------------
plt.figure(figsize=(6,6))
plt.imshow(prob_density,
           extent=[-grid_range, grid_range, -grid_range, grid_range],
           origin='lower')
plt.xlabel(r'$x_1$')
plt.ylabel(r'$y_1$')
plt.title(r'Densidad de probabilidad $|\Psi(x_1,y_1;0,0)|^2$')
plt.colorbar(label=r'$|\Psi|^2$')
plt.tight_layout()
plt.show()
