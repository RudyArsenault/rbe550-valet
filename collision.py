"""collision.py 

Method: SAT between each vehicle's oriented bounding box

Two oriented rectangles (or an OBB and an axis-aligned rectangle) are disjoint IFF there
is an axis on which their projections do not overlap.
For rectangles, only 4 candidate axes needed: the 2 edge normals of each box. For an OBB vs. an axis-aligned cell the
axes are x, y (the cell's edges) and the OBB's two edge directions.

To stay fast inside a planner, only cells that overlap the OBB's axis-aligned bounding
box are tested (a broad phase), so the cost per check does not grow with map size.
"""
import numpy as np
from environment import CELL, GRID, SIZE


def cell_corners(ix, iy):
    x0, y0 = ix * CELL, iy * CELL
    return np.array([[x0, y0], [x0 + CELL, y0], [x0 + CELL, y0 + CELL], [x0, y0 + CELL]])


def _project(poly, axis):
    d = poly @ axis
    return d.min(), d.max()


def sat_overlap(poly_a, poly_b, eps=1e-9):
    #true if two convex polygons (Nx2 arrays) overlap 

   # Returns (overlap, axis_of_min_penetration) - the second item is None when they are
   # separated, otherwise the axis with the smallest overlap (used for the diagrams)
    best = None
    for poly in (poly_a, poly_b):
        n = len(poly)
        for i in range(n):
            edge = poly[(i + 1) % n] - poly[i]
            axis = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(axis)
            if norm < eps:
                continue
            axis /= norm
            a0, a1 = _project(poly_a, axis)
            b0, b1 = _project(poly_b, axis)
            overlap = min(a1, b1) - max(a0, b0)
            if overlap <= eps:          # gap or just touching = separating axis found
                return False, None
            if best is None or overlap < best[0]:
                best = (overlap, axis)
    return True, best[1]


class CollisionChecker:
    #checks vehicle states against a ParkingLot

    def __init__(self, lot, vehicle):
        self.lot = lot
        self.vehicle = vehicle
        self.grid = lot.grid

    #single box
    def box_hits(self, corners):
        #return list of (ix, iy) obstacle cells the box overlaps, plus 'boundary' flag
        hits = []
        out_of_bounds = bool(np.any(corners < 0.0) or np.any(corners > SIZE))
        lo = np.floor(corners.min(axis=0) / CELL).astype(int)
        hi = np.floor(corners.max(axis=0) / CELL).astype(int)
        for ix in range(max(lo[0], 0), min(hi[0], GRID - 1) + 1):
            for iy in range(max(lo[1], 0), min(hi[1], GRID - 1) + 1):
                if self.grid[ix, iy] and sat_overlap(corners, cell_corners(ix, iy))[0]:
                    hits.append((ix, iy))
        return hits, out_of_bounds

    #vehicle
    def in_collision(self, state):
        if hasattr(self.vehicle, "jackknifed") and self.vehicle.jackknifed(state):
            return True
        for box in self.vehicle.footprints(state):
            hits, oob = self.box_hits(box)
            if hits or oob:
                return True
        return False

    def path_in_collision(self, states):
        return any(self.in_collision(s) for s in states)

    def report(self, state):
        """Per-box detail, used by the diagrams: list of (corners, hit_cells, out_of_bounds)."""
        return [(b, *self.box_hits(b)) for b in self.vehicle.footprints(state)]



# Diagrams for the report

def plot_check(checker, state, ax=None, title=None):
    """Draw the lot, the vehicle boxes, and highlight overlapping cells in red."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Polygon
    from environment import plot_lot
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    plot_lot(checker.lot, ax=ax)
    collided = checker.in_collision(state)
    for corners, hits, oob in checker.report(state):
        for ix, iy in hits:
            ax.add_patch(Rectangle((ix * CELL, iy * CELL), CELL, CELL, color="red", alpha=0.7))
        ax.add_patch(Polygon(corners, closed=True, fill=True,
                             fc=("tab:red" if (hits or oob) else "tab:cyan"),
                             ec="k", alpha=0.5))
    ax.set_title(title or ("COLLISION" if collided else "free"))
    return ax


if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt
    from environment import generate_lot
    from vehicles import Car, TruckTrailer

    #unit tests
    a = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], float)
    assert sat_overlap(a, a + 1.0)[0]                       #overlapping
    assert not sat_overlap(a, a + 2.0)[0]                   #touching at a corner
    assert not sat_overlap(a, a + [3.0, 0])[0]              #clear gap
    th = np.pi / 4                                          #rotated square vs. square
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    diamond = (np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) @ R.T) + [3.5, 1.0]
    assert not sat_overlap(a, diamond)[0]   # rotated box just misses (left tip at x~2.09)
    assert sat_overlap(a, diamond - [0.8, 0])[0]            #tip pokes in (x~1.29)

    lot = generate_lot("car", seed=1)
    car = Car()
    chk = CollisionChecker(lot, car)
    assert not chk.in_collision((15.0, 28.0, 0.3))          #open space
    assert chk.in_collision((0.5, 0.5, 0.0))                #sticks out of the lot
    assert chk.in_collision((22.0, 16.5, 0.0))              # inside the obstacle bar
    print("SAT unit tests passed")

    #demo  figur with a few poses
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    poses = [(15.0, 28.0, 0.3), (22.0, 20.0, 0.0), (8.0, 10.0, 1.2)]
    for ax, p in zip(axs, poses):
        plot_check(chk, p, ax=ax)
    plt.tight_layout()
    plt.savefig("collision_demo.png", dpi=80)
    print("saved collision_demo.png")