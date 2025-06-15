import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime
import shutil
import os

def animate_sampler2D(sampler, sweeps: int = 100, grid_range: float = 4.0, grid_size: int = 100, save: bool = False):
    """
    Visualize the walkers of the first fermion with the wavefunction of the network in the background.
    """
    net = sampler.network
    device = next(net.parameters()).device
    net.eval()

    #print("▶ Reset walkers...")
    #sampler.reset_walkers()

    positions_history = []

    @torch.no_grad()
    def run_sweeps_and_record():
        for _ in range(sweeps):
            chains, _ = sampler(n_sweeps=1)  # [nwalkers, dof, dim]
            pos = chains[:, 0, :]  # first fermion
            assert pos.shape[-1] == 2, f"Expected 2D coordinates, but got shape: {pos.shape}"
            positions_history.append(pos.cpu())

    run_sweeps_and_record()

    # ===== Evaluate |ψ|^2 on the grid =====
    x = torch.linspace(-grid_range, grid_range, grid_size)
    y = torch.linspace(-grid_range, grid_range, grid_size)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    A = sampler.dof
    D = sampler.dim
    npoints = grid_size * grid_size

    grid_points = torch.zeros((npoints, A, D))
    grid_points[:, 0, 0] = X.flatten()
    grid_points[:, 0, 1] = Y.flatten()

    input_tensor = grid_points.to(device)

    original_pretrain = net.pretrain
    net.pretrain = False
    with torch.no_grad():
        _, logabs = net(input_tensor)
        psi2 = (2 * logabs).cpu().reshape(grid_size, grid_size)  # ← log(|ψ|²)
    net.pretrain = original_pretrain

    # ========== Animation ==========
    fig, ax = plt.subplots(figsize=(6, 6))

    im = ax.imshow(psi2,
               extent=[-grid_range, grid_range, -grid_range, grid_range],
               origin='lower', cmap='viridis',
               alpha=0.6, aspect='auto')
    sc = ax.scatter([], [], s=10, c='blue', alpha=0.8)

    ax.set_xlim(-grid_range, grid_range)
    ax.set_ylim(-grid_range, grid_range)
    ax.set_title("First fermion's walkers evolution")

    def update(frame):
        pos = positions_history[frame]
        sc.set_offsets(pos)
        ax.set_title(f"Sweep {frame + 1}")
        return sc,

    print("▶ Creating animation...")

    ani = FuncAnimation(fig, update, frames=len(positions_history), interval=100, blit=True, repeat=False)
    fig.ani = ani  # Prevent garbage collection

    if save:
        # Check ffmpeg availability
        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "❌ ffmpeg not found. Please install it and make sure it's in your PATH.\n"
                "On Windows with conda: conda install -c conda-forge ffmpeg"
            )

        os.makedirs("results/animations", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/animations/walkers_animation_{timestamp}.mp4"
        ani.save(filename, writer="ffmpeg", fps=10)
        print(f"✅ Animation saved at: {filename}")

    plt.tight_layout()
    plt.show()

