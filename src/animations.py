import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_sampler2D(sampler, sweeps: int = 100, xlim=(-4, 4), ylim=(-4, 4), resolution: int = 100):
    """
    Visualiza la evolución del primer fermión con |\psi(x,y)|^2 como fondo.
    """
    net = sampler.network
    device = next(net.parameters()).device
    net.eval()

    print("▶ Reseteando walkers...")
    sampler.reset_walkers()

    positions_history = []

    @torch.no_grad()
    def run_sweeps_and_record():
        for _ in range(sweeps):
            chains, _ = sampler(n_sweeps=1)  # [nwalkers, dof, dim]
            pos = chains[:, 0, :]  # primer fermión
            assert pos.shape[-1] == 2, f"Esperaba coordenadas 2D, pero recibí {pos.shape}"
            positions_history.append(pos.cpu().numpy())

    run_sweeps_and_record()

    # ===== Evaluar |\psi|² en una malla =====
    x = np.linspace(*xlim, resolution)
    y = np.linspace(*ylim, resolution)
    xx, yy = np.meshgrid(x, y)  # atención: xx es eje horizontal, yy eje vertical

    grid_points = np.stack([xx, yy], axis=-1).reshape(-1, 2)  # [res², 2]
    dof = sampler.dof

    walker_grid = np.zeros((grid_points.shape[0], dof, 2), dtype=np.float32)
    walker_grid[:, 0, :] = grid_points  # solo el primer fermión se mueve

    input_tensor = torch.tensor(walker_grid, device=device)

    # Desactivar preentrenamiento solo mientras se evalúa la función
    original_pretrain = net.pretrain
    net.pretrain = False

    with torch.no_grad():
        _, logabs = net(input_tensor)
        psi2 = torch.exp(2 * logabs).cpu().numpy().reshape(resolution, resolution)

    net.pretrain = original_pretrain

    # ========== Animación ==========
    fig, ax = plt.subplots(figsize=(6, 6))

    im = ax.imshow(psi2, extent=(*xlim, *ylim), origin='lower',
                   cmap='viridis', alpha=0.6, aspect='auto')
    sc = ax.scatter([], [], s=10, c='blue', alpha=0.8)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_title("Evolución del primer fermión")

    def update(frame):
        pos = positions_history[frame]
        sc.set_offsets(pos)
        ax.set_title(f"Sweep {frame + 1}")
        return sc,

    print("▶ Mostrando animación...")
    ani = FuncAnimation(fig, update, frames=len(positions_history),
                        interval=100, blit=True, repeat=False)
    plt.tight_layout()
    plt.show()

