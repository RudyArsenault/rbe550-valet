"""planner.py - Hybrid A* for RBE-550 Assignment 4 (VALET).

Hybrid A* (Dolgov et al., 2008) searches a discretized (x, y, theta) grid, but every node
stores the CONTINUOUS pose reached by driving a kinematically feasible motion primitive,
so the final path is drivable. Pieces:

  * Motion primitives  - short forward/reverse arcs at several steering angles
                         (+ turn-in-place for the diff-drive robot).
  * Cost               - arc length, extra cost for reversing, switching direction,
                         and steering.
  * Heuristic          - obstacle-aware 2D distance to the goal (Dijkstra on a 0.5 m grid).
  * Analytic expansion - periodically try a Reeds-Shepp curve straight to the goal and
                         accept it if collision free (this does the final parking move).
  * Collision checking - vectorized perimeter sampling + obstacle-corner-in-box test for
                         speed during search; returned paths are re-verified with the exact
                         SAT checker in collision.py.

If no path exists the planner returns a PlanResult with success=False and a reason,
instead of raising - the assignment requires graceful failure.

AI-assisted code (Claude, Anthropic).
"""
import heapq
import math
import time
from dataclasses import dataclass, field

import numpy as np

from environment import CELL, GRID, SIZE
from reeds_shepp import rs_paths, sample_path, mod2pi
from vehicles import Car, DiffDriveRobot


# ---------------------------------------------------------------------------
# Configuration and result types
# ---------------------------------------------------------------------------
@dataclass
class PlannerConfig:
    xy_res: float = 1.0          # m, search grid resolution
    yaw_bins: int = 72           # 5 degree heading bins
    step: float = 1.6            # m, arc length of one motion primitive
    n_steer: int = 5             # steering angles per direction
    sample_ds: float = 0.2       # m, collision-check spacing along a primitive
    reverse_cost: float = 1.5    # multiplier on length driven backward
    switch_cost: float = 3.0     # penalty for each forward<->reverse change
    steer_cost: float = 0.2      # penalty per meter at full steering
    steer_change_cost: float = 0.3
    h_weight: float = 1.5        # >1 = weighted A* (faster, slightly suboptimal)
    rs_near: float = 15.0        # m, always try Reeds-Shepp when this close to the goal
    rs_every: int = 10           # otherwise try it every N expansions
    rs_max_candidates: int = 12  # how many RS curves to collision-check per attempt
    timeout: float = 60.0        # s
    max_expansions: int = 150000


@dataclass
class PlanResult:
    success: bool
    path: np.ndarray = None      # (N, 3) poses [x, y, theta]
    dirs: np.ndarray = None      # (N,) +1 forward / -1 reverse
    reason: str = ""
    expansions: int = 0
    time: float = 0.0
    cost: float = float("inf")


# ---------------------------------------------------------------------------
# Vehicle adapters: footprint geometry, primitives, and goal conversion
# ---------------------------------------------------------------------------
class _Model:
    """Common geometry for a single-box vehicle whose reference point is (x, y)."""

    def __init__(self, back, front, width):
        self.back, self.front, self.hw = back, front, width / 2.0
        # perimeter sample points in the body frame, ~0.1 m apart
        pts = []
        for (ax, ay), (bx, by) in zip(
                [(-back, -self.hw), (front, -self.hw), (front, self.hw), (-back, self.hw)],
                [(front, -self.hw), (front, self.hw), (-back, self.hw), (-back, -self.hw)]):
            n = max(2, int(math.ceil(math.hypot(bx - ax, by - ay) / 0.1)))
            t = np.linspace(0.0, 1.0, n, endpoint=False)
            pts.append(np.stack([ax + (bx - ax) * t, ay + (by - ay) * t], axis=1))
        self.perimeter = np.concatenate(pts)
        self.center_offset = (front - back) / 2.0   # box center ahead of the ref point

    def ref_from_center(self, cx, cy, th):
        """Reference-point pose that puts the box center at (cx, cy)."""
        return (cx - self.center_offset * math.cos(th),
                cy - self.center_offset * math.sin(th), th)


class CarModel(_Model):
    def __init__(self, car: Car, cfg: PlannerConfig):
        super().__init__(car.rear_overhang, car.length - car.rear_overhang, car.width)
        self.L = car.wheelbase
        self.rs_radius = car.wheelbase / math.tan(car.phi_max)
        steers = np.linspace(-car.phi_max, car.phi_max, cfg.n_steer)
        # primitive = (direction, curvature, steer normalized to [-1, 1], in_place)
        self.prims = [(d, math.tan(p) / self.L, p / car.phi_max, False)
                      for d in (1, -1) for p in steers]


class RobotModel(_Model):
    def __init__(self, robot: DiffDriveRobot, cfg: PlannerConfig):
        super().__init__(robot.length / 2, robot.length / 2, robot.width)
        self.rs_radius = 0.75
        ks = [-1 / 0.75, -1 / 2.0, 0.0, 1 / 2.0, 1 / 0.75]
        self.prims = [(d, k, k * 0.75, False) for d in (1, -1) for k in ks]
        # turn in place by +/- 45 degrees (direction 0 = no translation)
        self.prims += [(0, s * math.pi / 4, 0.0, True) for s in (1, -1)]


def make_model(vehicle, cfg):
    if isinstance(vehicle, Car):
        return CarModel(vehicle, cfg)
    if isinstance(vehicle, DiffDriveRobot):
        return RobotModel(vehicle, cfg)
    raise NotImplementedError("truck + trailer planner comes next")


def propagate(state, prim, step, ds):
    """Drive one primitive exactly (arcs are integrated in closed form).
    Returns (n, 3) poses along it, excluding the start pose."""
    x, y, th = state
    d, k, _, in_place = prim
    if in_place:                               # k holds the rotation angle
        n = max(2, int(math.ceil(abs(k) / 0.1)))
        a = th + k * np.arange(1, n + 1) / n
        return np.stack([np.full(n, x), np.full(n, y), a], axis=1)
    n = max(2, int(math.ceil(step / ds)))
    s = d * step * np.arange(1, n + 1) / n     # signed arc length
    if abs(k) < 1e-9:
        return np.stack([x + s * math.cos(th), y + s * math.sin(th), np.full(n, th)], axis=1)
    a = th + k * s
    return np.stack([x + (np.sin(a) - math.sin(th)) / k,
                     y - (np.cos(a) - math.cos(th)) / k, a], axis=1)


# ---------------------------------------------------------------------------
# Fast vectorized collision test used inside the search
# ---------------------------------------------------------------------------
class FastCollision:
    def __init__(self, lot, model):
        self.grid = lot.grid
        self.m = model
        corners = set()
        for ix, iy in zip(*np.nonzero(lot.grid)):
            for cx in (ix, ix + 1):
                for cy in (iy, iy + 1):
                    corners.add((cx * CELL, cy * CELL))
        self.corners = np.array(sorted(corners)) if corners else np.zeros((0, 2))

    def hits(self, poses):
        """True if any pose in (K, 3) collides or leaves the lot."""
        poses = np.atleast_2d(poses)
        c, s = np.cos(poses[:, 2]), np.sin(poses[:, 2])
        P = self.m.perimeter
        wx = poses[:, None, 0] + c[:, None] * P[None, :, 0] - s[:, None] * P[None, :, 1]
        wy = poses[:, None, 1] + s[:, None] * P[None, :, 0] + c[:, None] * P[None, :, 1]
        if (wx.min() < 0) or (wy.min() < 0) or (wx.max() > SIZE) or (wy.max() > SIZE):
            return True
        ix = np.minimum((wx / CELL).astype(int), GRID - 1)
        iy = np.minimum((wy / CELL).astype(int), GRID - 1)
        if self.grid[ix, iy].any():
            return True
        if len(self.corners):                  # obstacle corner inside the vehicle box
            dx = self.corners[None, :, 0] - poses[:, None, 0]
            dy = self.corners[None, :, 1] - poses[:, None, 1]
            lx = dx * c[:, None] + dy * s[:, None]
            ly = -dx * s[:, None] + dy * c[:, None]
            inside = (lx > -self.m.back) & (lx < self.m.front) & (np.abs(ly) < self.m.hw)
            if inside.any():
                return True
        return False


# ---------------------------------------------------------------------------
# Heuristic: obstacle-aware 2D distance-to-goal (Dijkstra, 8-connected)
# ---------------------------------------------------------------------------
def holonomic_heuristic(lot, goal_xy, res=0.5):
    n = int(round(SIZE / res))
    sub = int(round(CELL / res))
    occ = np.kron(lot.grid, np.ones((sub, sub), dtype=bool))
    dist = np.full((n, n), np.inf)
    gx = min(int(goal_xy[0] / res), n - 1)
    gy = min(int(goal_xy[1] / res), n - 1)
    dist[gx, gy] = 0.0
    pq = [(0.0, gx, gy)]
    moves = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
             (1, 1, 1.4142), (1, -1, 1.4142), (-1, 1, 1.4142), (-1, -1, 1.4142)]
    while pq:
        d, i, j = heapq.heappop(pq)
        if d > dist[i, j]:
            continue
        for di, dj, w in moves:
            a, b = i + di, j + dj
            if 0 <= a < n and 0 <= b < n and not occ[a, b]:
                nd = d + w * res
                if nd < dist[a, b]:
                    dist[a, b] = nd
                    heapq.heappush(pq, (nd, a, b))
    return dist, res


# ---------------------------------------------------------------------------
# Hybrid A*
# ---------------------------------------------------------------------------
@dataclass(order=True)
class _Node:
    f: float
    g: float = field(compare=False)
    state: tuple = field(compare=False)
    parent: tuple = field(compare=False)
    traj: np.ndarray = field(compare=False)
    dir: int = field(compare=False)
    steer: float = field(compare=False)


class HybridAStar:
    def __init__(self, lot, vehicle, cfg: PlannerConfig = None):
        self.lot = lot
        self.vehicle = vehicle
        self.cfg = cfg or PlannerConfig()
        self.model = make_model(vehicle, self.cfg)
        self.coll = FastCollision(lot, self.model)

    # --- helpers ---------------------------------------------------------
    def _key(self, s):
        c = self.cfg
        return (int(s[0] // c.xy_res), int(s[1] // c.xy_res),
                int((mod2pi(s[2]) + math.pi) // (2 * math.pi / c.yaw_bins)) % c.yaw_bins)

    def _h(self, s):
        i = min(int(s[0] / self.hres), self.hdist.shape[0] - 1)
        j = min(int(s[1] / self.hres), self.hdist.shape[1] - 1)
        return self.cfg.h_weight * self.hdist[i, j]

    def _prim_cost(self, prim, parent):
        c = self.cfg
        d, k, steer, in_place = prim
        if in_place:
            return 1.0 * abs(k)                 # ~1 m-equivalent per radian
        cost = c.step * (1.0 if d > 0 else c.reverse_cost)
        cost += c.step * c.steer_cost * abs(steer)
        cost += c.steer_change_cost * abs(steer - parent.steer)
        if parent.dir != 0 and d != parent.dir:
            cost += c.switch_cost
        return cost

    def _rs_cost(self, lengths, first_dir):
        c = self.cfg
        cost, prev = 0.0, first_dir
        for l in lengths:
            d = 1 if l >= 0 else -1
            cost += abs(l) * (1.0 if d > 0 else c.reverse_cost)
            if prev != 0 and d != prev:
                cost += c.switch_cost
            prev = d
        return cost

    def _try_analytic(self, node):
        """Reeds-Shepp shot to each goal pose; returns (traj, dirs, cost) or None."""
        best = None
        for goal in self.goals:
            for i, (_, L, T) in enumerate(rs_paths(node.state, goal, self.model.rs_radius)):
                if i >= self.cfg.rs_max_candidates:
                    break
                cost = self._rs_cost(L, node.dir)
                if best and cost >= best[2]:
                    continue
                pts, dirs = sample_path(node.state, L, T, self.model.rs_radius,
                                        self.cfg.sample_ds)
                if not self.coll.hits(pts):
                    best = (pts[1:], dirs[1:], cost)   # keep the cheapest (not shortest)
        return best

    def _at_goal(self, s):
        tx, ty, tth = self.tol
        return any(abs(s[0] - g[0]) < tx and abs(s[1] - g[1]) < ty
                   and abs(mod2pi(s[2] - g[2])) < tth for g in self.goals)

    def _reconstruct(self, nodes, key, tail=None):
        segs, dsegs = [], []
        if tail is not None:
            segs.append(tail[0])
            dsegs.append(tail[1])
        while key is not None:
            n = nodes[key]
            if n.traj is not None:
                segs.append(n.traj)
                dsegs.append(np.full(len(n.traj), n.dir if n.dir != 0 else 1))
            else:
                segs.append(np.array([n.state]))
                dsegs.append(np.array([1]))
            key = n.parent
        path = np.concatenate(segs[::-1])
        dirs = np.concatenate(dsegs[::-1])
        return path, dirs

    # --- main entry ------------------------------------------------------
    def plan(self, start, goals, tol=(0.5, 0.5, 0.15)):
        """start: (x, y, th) reference pose. goals: list of acceptable (x, y, th) poses.
        Returns PlanResult (never raises for an unsolvable lot)."""
        t0 = time.time()
        cfg = self.cfg
        self.goals = [tuple(g) for g in goals]
        self.tol = tol
        if self.coll.hits(np.array([start])):
            return PlanResult(False, reason="start pose is in collision", time=time.time() - t0)
        self.goals = [g for g in self.goals if not self.coll.hits(np.array([g]))]
        if not self.goals:
            return PlanResult(False, reason="goal pose is in collision", time=time.time() - t0)
        self.hdist, self.hres = holonomic_heuristic(self.lot, self.goals[0][:2])
        if not np.isfinite(self._h(start)):
            return PlanResult(False, reason="goal unreachable: parking area is walled off",
                              time=time.time() - t0)

        start = tuple(float(v) for v in start)
        root = _Node(self._h(start), 0.0, start, None, None, 0, 0.0)
        k0 = self._key(start)
        nodes = {k0: root}
        open_heap = [(root.f, 0, k0)]
        closed = set()
        counter, expansions = 1, 0

        while open_heap:
            if time.time() - t0 > cfg.timeout:
                return PlanResult(False, reason=f"timeout after {cfg.timeout:.0f} s",
                                  expansions=expansions, time=time.time() - t0)
            if expansions >= cfg.max_expansions:
                return PlanResult(False, reason="expansion limit reached",
                                  expansions=expansions, time=time.time() - t0)
            _, _, key = heapq.heappop(open_heap)
            if key in closed:
                continue
            closed.add(key)
            node = nodes[key]
            expansions += 1

            if self._at_goal(node.state):
                path, dirs = self._reconstruct(nodes, key)
                return PlanResult(True, path, dirs, "reached goal tolerance",
                                  expansions, time.time() - t0, node.g)

            near = min(math.hypot(node.state[0] - g[0], node.state[1] - g[1])
                       for g in self.goals) < cfg.rs_near
            if near or expansions % cfg.rs_every == 0:
                shot = self._try_analytic(node)
                if shot is not None:
                    path, dirs = self._reconstruct(nodes, key, (shot[0], shot[1]))
                    return PlanResult(True, path, dirs, "analytic Reeds-Shepp expansion",
                                      expansions, time.time() - t0, node.g + shot[2])

            for prim in self.model.prims:
                traj = propagate(node.state, prim, cfg.step, cfg.sample_ds)
                end = tuple(float(v) for v in traj[-1])
                ck = self._key(end)
                if ck in closed or ck == key:
                    continue
                if self.coll.hits(traj):
                    continue
                g = node.g + self._prim_cost(prim, node)
                old = nodes.get(ck)
                if old is None or g < old.g:
                    h = self._h(end)
                    if not np.isfinite(h):
                        continue
                    nodes[ck] = _Node(g + h, g, end, key, traj, prim[0], prim[2])
                    heapq.heappush(open_heap, (g + h, counter, ck))
                    counter += 1

        return PlanResult(False, reason="no path: search space exhausted",
                          expansions=expansions, time=time.time() - t0)


# ---------------------------------------------------------------------------
# Parallel-parking maneuver, planned BACKWARDS ("unparking")
# ---------------------------------------------------------------------------
def unpark(coll, model, parked, k_max, max_moves=20, exit_len=4.0, ds=0.05, margin=0.10,
           side=None):
    """Shuffle out of a tight slot: alternate forward / reverse at full lock (steering
    toward the open side when going forward, away from it in reverse) until a forward
    full-lock arc of `exit_len` meters is collision free. Each move stops `margin`
    meters before contact.

    The slot is 6 m long for a 5.2 m car (0.4 m each end), so a coarse search cannot
    find this; but kinematics are reversible, so driving OUT and replaying it backwards
    is a valid way IN. Returns (poses (N,3) from parked to exit, dirs (N,), n_moves) or
    None.
    """
    s = tuple(float(v) for v in parked)
    poses, dirs = [s], [1]
    # side = +1 turns left going forward, -1 right. Default: toward north (the open side
    # of the slot): facing east (cos > 0), forward + left turns north.
    if side is None:
        side = 1.0 if math.cos(s[2]) >= 0 else -1.0
    d = 1
    for move in range(1, max_moves + 1):
        k = side * k_max * d
        travelled = 0.0
        while travelled < 8.0:
            look = propagate(s, (d, k, 0, False), ds + margin, ds)
            if coll.hits(look):
                break
            step = propagate(s, (d, k, 0, False), ds, ds)
            s = tuple(float(v) for v in step[-1])
            poses.append(s)
            dirs.append(d)
            travelled += ds
        exit_arc = propagate(s, (1, side * k_max, 0, False), exit_len, 0.1)
        if not coll.hits(exit_arc):
            for p in exit_arc:
                poses.append(tuple(float(v) for v in p))
                dirs.append(1)
            return np.array(poses), np.array(dirs), move
        d = -d
    return None


def _reverse_maneuver(poses, dirs):
    """Replay a maneuver backwards: drop the last pose (it is where the search ends),
    reverse the order, and flip every driving direction."""
    return poses[::-1][1:], -dirs[1:][::-1]


# ---------------------------------------------------------------------------
# Convenience: plan a vehicle on a lot from the NW start to the SE slot
# ---------------------------------------------------------------------------
def plan_vehicle(lot, vehicle, cfg=None, both_directions=True):
    """Start at the NW corner facing East, park in the SE slot (either heading if
    both_directions). Car: Hybrid A* to the exit pose of the unpark maneuver, then the
    maneuver replayed backwards. Robot: Hybrid A* straight to the slot."""
    planner = HybridAStar(lot, vehicle, cfg)
    m = planner.model
    sx, sy, _ = lot.start_pose()
    start = (sx, sy, 0.0) if isinstance(vehicle, DiffDriveRobot) else m.ref_from_center(3.0, sy, 0.0)
    gx, gy, gth = lot.goal_pose()
    headings = [gth, mod2pi(gth + math.pi)] if both_directions else [gth]
    parked = [m.ref_from_center(gx, gy, th) for th in headings]

    if not isinstance(vehicle, Car):
        return planner.plan(start, parked), planner

    k_max = 1.0 / m.rs_radius
    maneuvers = []
    for p in parked:
        if planner.coll.hits(np.array([p])):
            continue
        man = unpark(planner.coll, m, p, k_max)
        if man is not None:
            maneuvers.append(man)
    if not maneuvers:
        return PlanResult(False, reason="cannot maneuver out of / into the parking slot"), planner
    maneuvers.sort(key=lambda mm: mm[2])          # fewest shuffles first
    exits = [tuple(mm[0][-1]) for mm in maneuvers]

    # The start cells are 6 m long for a 5.2 m car: if an obstacle sits right next to
    # them, no full-length primitive fits, so shuffle out of the start pocket first.
    head_p, head_d, escape_note = None, None, ""
    free_children = sum(not planner.coll.hits(propagate(start, pr, planner.cfg.step,
                                                        planner.cfg.sample_ds))
                        for pr in m.prims)
    if free_children < 3:
        escapes = [e for e in (unpark(planner.coll, m, start, k_max, side=sd)
                               for sd in (-1.0, 1.0)) if e is not None]
        if escapes:
            head_p, head_d, n = min(escapes, key=lambda e: e[2])
            start = tuple(head_p[-1])
            escape_note = f"{n}-move start escape + "

    res = planner.plan(start, exits, tol=(0.05, 0.05, 0.02))   # RS shot lands exactly
    if not res.success:
        if free_children < 3 and head_p is None:
            res.reason = "vehicle is boxed in at the start (no room to turn out)"
        return res, planner
    if head_p is not None:
        res.path = np.concatenate([head_p[:-1], res.path])
        res.dirs = np.concatenate([head_d[:-1], res.dirs])
        res.reason = escape_note + res.reason
    end = res.path[-1]
    i = int(np.argmin([math.hypot(end[0] - e[0], end[1] - e[1]) + abs(mod2pi(end[2] - e[2]))
                       for e in exits]))
    tail_p, tail_d = _reverse_maneuver(maneuvers[i][0], maneuvers[i][1])
    res.path = np.concatenate([res.path, tail_p])
    res.dirs = np.concatenate([res.dirs, tail_d])
    res.reason += f" + {maneuvers[i][2]}-move parking maneuver"
    return res, planner


def plot_result(lot, vehicle, res, ax=None, every=15, title=None):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from environment import plot_lot
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    plot_lot(lot, ax=ax)
    if res.success:
        P, D = res.path, res.dirs
        for i in range(len(P) - 1):
            ax.plot(P[i:i + 2, 0], P[i:i + 2, 1], color=("tab:blue" if D[i + 1] > 0 else "tab:orange"),
                    lw=1.5)
        for i in list(range(0, len(P), every)) + [len(P) - 1]:
            for box in vehicle.footprints(P[i]):
                ax.add_patch(Polygon(box, closed=True, fill=False, ec="0.4", lw=0.6))
    ax.set_title(title or (f"{vehicle.name}: {'OK' if res.success else 'FAIL'} "
                           f"({res.time:.1f} s, {res.expansions} exp)"))
    return ax


if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt
    from environment import generate_lot
    from collision import CollisionChecker

    kinds = [("car", Car()), ("robot", DiffDriveRobot())]
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    for row, (kind, veh) in enumerate(kinds):
        for col, seed in enumerate(range(4)):
            lot = generate_lot(kind, seed=seed)
            res, _ = plan_vehicle(lot, veh)
            ok = res.success and not CollisionChecker(lot, veh).path_in_collision(res.path)
            print(f"{kind:5s} seed {seed}: success={res.success} exact_check_ok={ok} "
                  f"t={res.time:5.1f}s exp={res.expansions} reason='{res.reason}'")
            plot_result(lot, veh, res, ax=axs[row, col])
    plt.tight_layout()
    plt.savefig("planner_demo.png", dpi=70)
    print("saved planner_demo.png")
    plt.show()   # close the window to exit