"""environment.py

Coordinate convention
---------------------
World frame: x points East, y points North, units are meters.
The lot is GRID x GRID cells, each CELL meters square (12 x 12 cells, 3 m -> 36 m x 36 m).
Grid indexing: grid[ix, iy] with ix = column (0 = West) and iy = row counted from the
SOUTH (0 = South edge). Cell (ix, iy) covers x in [ix*CELL, (ix+1)*CELL] and
y in [iy*CELL, (iy+1)*CELL].

Start is the Northwest corner (ix=0, iy=GRID-1); parking is in the Southeast corner.

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

    slot_cells: list          # cells in the block forced FREE (incl. the whole top row)
    wall_cells: list          # cells in the block forced to OBSTACLE (bottom row walls)
    goal_box: tuple           # green rectangle in Fig. 5: (xmin, ymin, xmax, ymax) [m]
    goal_pose: tuple          # (x, y, theta) at the center of the green rectangle
    goal_tol: tuple = (0.5, 0.5, 0.15)  # (dx, dy, dtheta) tolerance, meters/radians


_BLOCK_COLS = range(GRID - 6, GRID)        # columns 6..11
_TOP_ROW = [(c, 1) for c in _BLOCK_COLS]


def _box(c0, c1):
    """Green-rectangle in cell units (x: c0..c1, y: 0.26..0.76)."""
    return (c0 * CELL, 0.26 * CELL, c1 * CELL, 0.76 * CELL)



_car_slot = ParkingSpec(
    slot_cells=_TOP_ROW + [(8, 0), (9, 0)],
    wall_cells=[(6, 0), (7, 0), (10, 0), (11, 0)],
    goal_box=_box(8.29 - 0.0, 9.71),
    goal_pose=(CELL * 9.0, CELL * 0.5, 0.0),
)

_truck_slot = ParkingSpec(
    slot_cells=_TOP_ROW + [(6, 0), (7, 0), (8, 0), (9, 0)],
    wall_cells=[(10, 0), (11, 0)],
    goal_box=_box(6.1, 9.9),                   # whole rig must fit in the 4-cell free run
    goal_pose=(CELL * 8.0, CELL * 0.5, 0.0),   # center of the free run (x = 24 m)
)

PARKING_SPECS = {
    "robot": _car_slot,
    "car": _car_slot,
    "truck": _truck_slot,
}


@dataclass
class ParkingLot:
    grid: np.ndarray                      # bool [GRID, GRID], True = obstacle
    start_cell: tuple = (0, GRID - 1)     # Northwest
    spec: ParkingSpec = None
    seed: int = None

    #geometry helpers
    def obstacle_rects(self):
        rects = []
        for ix, iy in zip(*np.nonzero(self.grid)):
            rects.append((ix * CELL, iy * CELL, (ix + 1) * CELL, (iy + 1) * CELL))
        return rects

    def start_pose(self, heading=0.0):
        ix, iy = self.start_cell
        return ((ix + 0.5) * CELL, (iy + 0.5) * CELL, heading)

    def goal_pose(self):
        return self.spec.goal_pose


# start cells kept free. fits the robot and the 5.2 m car 
#the truck + trailer rig is ~11.65 m long, needs 4 cells
START_CELLS = {
    "robot": [(0, GRID - 1), (1, GRID - 1)],
    "car": [(0, GRID - 1), (1, GRID - 1)],
    "truck": [(0, GRID - 1), (1, GRID - 1), (2, GRID - 1), (3, GRID - 1)],
}


def _free_cells(vehicle):
    spec = PARKING_SPECS[vehicle]
    keep = set(START_CELLS[vehicle])
    keep.update(map(tuple, spec.slot_cells))
    return keep


def generate_lot(vehicle="car", density=DENSITY, seed=None, max_tries=1000):
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

    #parking area overwrites spots
    for c in spec.slot_cells:
        grid[tuple(c)] = False
    for c in spec.wall_cells:
        grid[tuple(c)] = True
    for c in START_CELLS[vehicle]:
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
        x0, y0, x1, y1 = lot.spec.goal_box
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="green", lw=2))
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