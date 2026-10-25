"""animate.py - Path animations for RBE-550 Assignment 4 (VALET).

For each vehicle (robot, car, truck + trailer) this plans one run and saves:
  * <vehicle>_snapshots.png - periodic footprint snapshots superimposed on the lot
  * <vehicle>.gif           - a short animation of the vehicle driving start -> parked
GIFs are written with Pillow (already installed), so no ffmpeg is needed.

Usage:
    python animate.py                 # all three vehicles, default seeds
    python animate.py car 3           # one vehicle, a specific lot seed
"""
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")                 # render to files only
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon

from environment import generate_lot, plot_lot
from vehicles import Car, DiffDriveRobot, TruckTrailer
from planner import plan_vehicle
from truck_planner import plan_truck

DEFAULT_SEEDS = {"robot": 0, "car": 0, "truck": 0}
COLORS = {"robot": "tab:green", "car": "tab:blue", "truck": "tab:purple"}


def plan(kind, seed):
    lot = generate_lot(kind, seed=seed)
    if kind == "truck":
        veh = TruckTrailer()
        res, _ = plan_truck(lot, veh)
    else:
        veh = Car() if kind == "car" else DiffDriveRobot()
        res, _ = plan_vehicle(lot, veh)
    return lot, veh, res


def draw_path(ax, res):
    """Path line: blue = forward, orange = reverse."""
    P, D = res.path, res.dirs
    for i in range(len(P) - 1):
        ax.plot(P[i:i + 2, 0], P[i:i + 2, 1], lw=1.2,
                color="tab:blue" if D[i + 1] > 0 else "tab:orange")


def snapshots(kind, lot, veh, res, n_snaps=14):
    fig, ax = plt.subplots(figsize=(7, 7))
    plot_lot(lot, ax=ax)
    draw_path(ax, res)
    idx = np.linspace(0, len(res.path) - 1, n_snaps).astype(int)
    for j, i in enumerate(idx):
        alpha = 0.15 + 0.6 * j / (n_snaps - 1)          # later = darker
        for box in veh.footprints(res.path[i]):
            ax.add_patch(Polygon(box, closed=True, fc=COLORS[kind], ec="k",
                                 lw=0.6, alpha=alpha))
    ax.set_title(f"{kind}: start -> parked ({len(res.path)} poses, "
                 f"planned in {res.time:.1f} s)")
    fig.tight_layout()
    fig.savefig(f"{kind}_snapshots.png", dpi=110)
    plt.close(fig)


def gif(kind, lot, veh, res, seconds=8, fps=15):
    n_frames = seconds * fps
    idx = np.linspace(0, len(res.path) - 1, n_frames).astype(int)
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_lot(lot, ax=ax)
    draw_path(ax, res)
    patches = [ax.add_patch(Polygon(np.zeros((4, 2)), closed=True, fc=COLORS[kind],
                                    ec="k", alpha=0.85))
               for _ in veh.footprints(res.path[0])]
    title = ax.set_title(kind)

    def update(f):
        s = res.path[idx[f]]
        for p, box in zip(patches, veh.footprints(s)):
            p.set_xy(box)
        title.set_text(f"{kind}: {'forward' if res.dirs[idx[f]] > 0 else 'reverse'}")
        return patches + [title]

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    anim.save(f"{kind}.gif", writer=PillowWriter(fps=fps))
    plt.close(fig)


def run(kind, seed):
    print(f"planning {kind} (seed {seed}) ...", flush=True)
    lot, veh, res = plan(kind, seed)
    if not res.success:
        print(f"  {kind}: no path ({res.reason}) - try another seed")
        return
    snapshots(kind, lot, veh, res)
    gif(kind, lot, veh, res)
    print(f"  saved {kind}_snapshots.png and {kind}.gif")


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        k = sys.argv[1]
        run(k, int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_SEEDS[k])
    else:
        for k, s in DEFAULT_SEEDS.items():
            run(k, s)