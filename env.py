"""environment.py - Parking lot world for RBE-550 Assignment 4 (VALET).

Coordinate convention
---------------------
World frame: x points East, y points North, units are meters.
The lot is GRID x GRID cells, each CELL meters square (12 x 12 cells, 3 m -> 36 m x 36 m).
Grid indexing: grid[ix, iy] with ix = column (0 = West) and iy = row counted from the
SOUTH (0 = South edge). Cell (ix, iy) covers x in [ix*CELL, (ix+1)*CELL] and
y in [iy*CELL, (iy+1)*CELL].

Start is the Northwest corner (ix=0, iy=GRID-1); parking is in the Southeast corner.

NOTE: Figure 5 of the assignment (exact parking slot geometry) was not readable from the
PDF text, so the slot below is a PLACEHOLDER. Edit PARKING_SPECS to match Figure 5a/5b.
"""
from dataclasses import dataclass, field
import numpy as np

GRID = 12
CELL = 3.0
SIZE = GRID * CELL  # 36 m
DENSITY = 0.10

# Free tetrominoes as (dx, dy) cell offsets (Figure 3: I, L, S, T).
TETROMINOES = {
    "I": [(0, 0), (0, 1), (0, 2), (0, 3)],
    "L": [(0, 0), (0, 1), (0, 2), (1, 2)],
    "S": [(0, 0), (0, 1), (1, 1), (1, 2)],
    "T": [(0, 0), (0, 1), (0, 2), (1, 1)],
}


def _orientations(cells):
    """All distinct rotations/reflections of a cell list, normalized to min (0, 0)."""
    out = set()
    pts = list(cells)
    for _ in range(4):
        pts = [(-y, x) for x, y in pts]  # rotate 90 deg
        for flip in (False, True):
            q = [(-x, y) for x, y in pts] if flip else pts
            mx = min(p[0] for p in q)
            my = min(p[1] for p in q)
            out.add(tuple(sorted((x - mx, y - my) for x, y in q)))
    return [list(o) for o in out]


@dataclass
class ParkingSpec:
    """Parking geometry for one vehicle type (PLACEHOLDER - match Figure 5)."""
    slot_cells: list          # cells that must be free (the parking slot)
    wall_cells: list          # obstacle cells flanking the slot
    goal_pose: tuple          # (x, y, theta) of the vehicle reference point at the goal
    goal_tol: tuple = (0.5, 0.5, 0.15)  # (dx, dy, dtheta) tolerance, meters/radians


# Slot along the south wall, two cells (6 m) long, flanked by obstacles on both ends.
_south_slot = ParkingSpec(
    slot_cells=[(9, 0), (10, 0)],
    wall_cells=[(8, 0), (11, 0)],
    goal_pose=(CELL * 10.0, CELL * 0.5, 0.0),  # center of the slot, facing East
)

PARKING_SPECS = {
    "robot": _south_slot,
    "car": _south_slot,
    "truck": _south_slot,  # TODO: use the Figure 5b configuration for truck + trailer
}


@dataclass
class ParkingLot:
    grid: np.ndarray                      # bool [GRID, GRID], True = obstacle
    start_cell: tuple = (0, GRID - 1)     # Northwest
    spec: ParkingSpec = None
    seed: int = None

    # ---- geometry helpers -------------------------------------------------
    def obstacle_rects(self):
        """Axis-aligned obstacle cells as (xmin, ymin, xmax, ymax) in meters."""
        rects = []
        for ix, iy in zip(*np.nonzero(self.grid)):
            rects.append((ix * CELL, iy * CELL, (ix + 1) * CELL, (iy + 1) * CELL))
        return rects

    def start_pose(self, heading=-np.pi / 2):
        """Pose near the NW cell center. Default heading faces South, down the west wall;
        pass heading=0 to face East. Adjust to taste."""
        ix, iy = self.start_cell
        return ((ix + 0.5) * CELL, (iy + 0.5) * CELL, heading)

    def goal_pose(self):
        return self.spec.goal_pose


def _free_cells(vehicle):
    spec = PARKING_SPECS[vehicle]
    keep = {(0, GRID - 1), (1, GRID - 1)}  # start cell and its eastern neighbor
    keep.update(map(tuple, spec.slot_cells))
    return keep


def generate_lot(vehicle="car", density=DENSITY, seed=None, max_tries=1000):
    """Random tetromino obstacle field, then overwrite the parking area."""
    rng = np.random.default_rng(seed)
    spec = PARKING_SPECS[vehicle]
    keep_free = _free_cells(vehicle)
    grid = np.zeros((GRID, GRID), dtype=bool)
    target = int(round(density * GRID * GRID))
    shapes = [_orientations(c) for c in TETROMINOES.values()]

    tries = 0
    while grid.sum() < target and tries < max_tries:
        tries += 1
        orient = shapes[rng.integers(len(shapes))]
        cells = orient[rng.integers(len(orient))]
        w = max(c[0] for c in cells) + 1
        h = max(c[1] for c in cells) + 1
        ox = rng.integers(0, GRID - w + 1)
        oy = rng.integers(0, GRID - h + 1)
        placed = [(ox + dx, oy + dy) for dx, dy in cells]
        if any(grid[c] or c in keep_free for c in placed):
            continue
        for c in placed:
            grid[c] = True

    # Parking area overwrites whatever is there.
    for c in spec.slot_cells:
        grid[tuple(c)] = False
    for c in spec.wall_cells:
        grid[tuple(c)] = True
    for c in [(0, GRID - 1), (1, GRID - 1)]:
        grid[c] = False
    return ParkingLot(grid=grid, spec=spec, seed=seed)


def plot_lot(lot, ax=None, show_goal=True):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    for k in range(GRID + 1):
        ax.plot([0, SIZE], [k * CELL] * 2, color="0.85", lw=0.5, zorder=0)
        ax.plot([k * CELL] * 2, [0, SIZE], color="0.85", lw=0.5, zorder=0)
    for (x0, y0, x1, y1) in lot.obstacle_rects():
        ax.add_patch(Rectangle((x0, y0), CELL, CELL, color="k"))
    if show_goal:
        for ix, iy in lot.spec.slot_cells:
            ax.add_patch(Rectangle((ix * CELL, iy * CELL), CELL, CELL,
                                   fill=False, ec="green", lw=2))
        gx, gy, _ = lot.spec.goal_pose
        ax.plot(gx, gy, "g*", ms=12)
    sx, sy, _ = lot.start_pose()
    ax.plot(sx, sy, "bo")
    ax.set_xlim(0, SIZE)
    ax.set_ylim(0, SIZE)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    return ax


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    lot = generate_lot("car", seed=1)
    print("obstacle cells:", int(lot.grid.sum()), "of", GRID * GRID)
    plot_lot(lot)
    plt.title("Parking lot (blue = start, green = parking slot)")
    plt.show()