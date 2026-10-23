"""truck_planner.py - Hybrid A* for the truck + trailer (RBE-550 Assignment 4, VALET).

Same algorithm as planner.py, extended to the 4-D state (x, y, th0, th1):
  (x, y)  truck rear-axle center = hitch point,  th0 truck heading,  th1 trailer heading.

Kinematics (Figure 9, one trailer), integrated along arc length s:
    dth0/ds = tan(phi) / L          dth1/ds = sin(th0 - th1) / d1

Differences from the car planner:
  * the search grid also bins the hitch angle (th0 - th1), so states with the same truck
    pose but a different trailer angle are kept apart;
  * every primitive and Reeds-Shepp shot is rolled out with the trailer equation and
    rejected if the cab OR the trailer collides, or the rig jackknifes
    (|th0 - th1| > hitch_angle_max);
  * Reeds-Shepp curves only describe the truck, so a shot counts only if the trailer it
    drags along also ends inside the goal region.

Goal (Figure 5b, boundary at our discretion per Sec. 3.2): rig parked along the south
wall facing East, nose within ~1.5 m of the east wall, cab and trailer axles in the
bottom row (y in [1.0, 2.0] m), truck and trailer both within 8 deg of East.

AI-assisted code (Claude, Anthropic).
"""
import heapq
import math
import time
from dataclasses import dataclass, field

import numpy as np

from environment import CELL
from planner import (PlannerConfig, PlanResult, FastCollision, _Model,
                     holonomic_heuristic, propagate)
from reeds_shepp import rs_paths, sample_path, mod2pi
from vehicles import TruckTrailer


class TruckModel:
    def __init__(self, truck: TruckTrailer, cfg: PlannerConfig):
        self.truck = truck
        self.cab = _Model(truck.rear_overhang, truck.length - truck.rear_overhang, truck.width)
        self.trl = _Model(truck.trailer_length / 2, truck.trailer_length / 2,
                          truck.trailer_width)
        self.L, self.d1 = truck.wheelbase, truck.d1
        self.hitch_max = truck.hitch_angle_max
        self.rs_radius = self.L / math.tan(truck.phi_max)
        steers = np.linspace(-truck.phi_max, truck.phi_max, cfg.n_steer)
        self.prims = [(d, math.tan(p) / self.L, p / truck.phi_max, False)
                      for d in (1, -1) for p in steers]

    def front(self, s):
        """x, y of the middle of the truck's front bumper."""
        f = self.cab.front
        return s[0] + f * math.cos(s[2]), s[1] + f * math.sin(s[2])


def trailer_rollout(th1_0, truck_poses, prev_pose, d1):
    """Integrate the trailer heading along a sequence of truck poses (midpoint / RK2 in
    arc length). truck_poses: (n, 3); prev_pose: the pose before the first one.
    Returns (n,) th1. The signed step is the travel along the truck heading (negative
    when reversing). Everything but the 2 sin() calls per step is precomputed in numpy."""
    P = np.vstack([np.asarray(prev_pose, float)[None, :3], truck_poses[:, :3]])
    dth = (np.diff(P[:, 2]) + np.pi) % (2 * np.pi) - np.pi
    amid = P[:-1, 2] + 0.5 * dth                         # truck heading mid-step
    ds = np.diff(P[:, 0]) * np.cos(amid) + np.diff(P[:, 1]) * np.sin(amid)
    h = (ds / d1).tolist()
    a0 = P[:-1, 2].tolist()
    am = amid.tolist()
    out = [0.0] * len(h)
    t1 = th1_0
    sin = math.sin
    for i in range(len(h)):
        k1 = sin(a0[i] - t1)
        t1 = t1 + h[i] * sin(am[i] - (t1 + 0.5 * h[i] * k1))
        out[i] = t1
    return np.array(out)


class TruckCollision:
    def __init__(self, lot, model: TruckModel):
        self.m = model
        self.cab = FastCollision(lot, model.cab)
        self.trl = FastCollision(lot, model.trl)

    def hits(self, states):
        """states: (K, 4). Collision, leaving the lot, or jackknife -> True."""
        S = np.atleast_2d(states)
        if np.any(np.abs(mod2pi(S[:, 2] - S[:, 3])) > self.m.hitch_max):
            return True
        if self.cab.hits(S[:, :3]):
            return True
        d1 = self.m.d1
        trailer = np.stack([S[:, 0] - d1 * np.cos(S[:, 3]),
                            S[:, 1] - d1 * np.sin(S[:, 3]), S[:, 3]], axis=1)
        return self.trl.hits(trailer)


def approach_heuristic(lot, goal_xy, inflate=1.0, res=0.5):
    """Obstacle-aware 2D distance-to-goal shaped for a trailer.

    A trailer cannot be dropped into the parking row from above, it has to drive in along
    the row heading East. So for the HEURISTIC ONLY the row above the parking block is
    treated as blocked (the 2D path then enters the slot from the West), and obstacles
    are inflated by half the rig width so the path avoids gaps the truck cannot use.
    This makes the heuristic inadmissible (paths may be slightly suboptimal) but it
    steers the 4-D search toward approaches that actually work.
    """
    from scipy.ndimage import binary_dilation
    from environment import GRID, SIZE
    n = int(round(SIZE / res))
    sub = int(round(CELL / res))
    grid = lot.grid.copy()
    for c in range(GRID - 6, GRID):
        grid[c, 1] = True
    occ = np.kron(grid, np.ones((sub, sub), dtype=bool))
    k = int(round(inflate / res))
    occ = binary_dilation(occ, iterations=k)
    border = np.zeros_like(occ)                        # lot boundary counts too
    border[:k, :] = border[-k:, :] = border[:, :k] = border[:, -k:] = True
    occ |= border
    gx, gy = min(int(goal_xy[0] / res), n - 1), min(int(goal_xy[1] / res), n - 1)
    occ[gx, gy] = False
    dist = np.full((n, n), np.inf)
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
    # cells that became unreachable only because of inflation: fall back to plain map
    plain, _ = holonomic_heuristic(lot, goal_xy, res)
    dist = np.where(np.isfinite(dist), dist, plain + 50.0)
    return dist, res


@dataclass
class TruckGoal:
    wall_x: float            # x of the east wall face (front of the parking row)
    nose_gap: tuple = (0.15, 1.5)   # allowed gap between bumper and wall [m]
    y_band: tuple = (0.9, 2.1)      # cab and trailer axle y [m]
    ang_tol: float = math.radians(10.0)     # truck heading vs. East
    trailer_tol: float = math.radians(15.0)  # trailer heading vs. East


@dataclass(order=True)
class _Node:
    f: float
    g: float = field(compare=False)
    state: tuple = field(compare=False)
    parent: tuple = field(compare=False)
    traj: np.ndarray = field(compare=False)
    dir: int = field(compare=False)
    steer: float = field(compare=False)


class TruckHybridAStar:
    def __init__(self, lot, truck: TruckTrailer, cfg: PlannerConfig = None):
        self.lot = lot
        self.truck = truck
        # Tuned on 2-core runs: coarser grid + longer primitives + 15 deg hitch bins
        # roughly halve the time to a solution versus the car settings.
        self.cfg = cfg or PlannerConfig(timeout=120.0, reverse_cost=2.5, switch_cost=5.0,
                                        h_weight=2.0, step=2.2, xy_res=1.5)
        self.model = TruckModel(truck, self.cfg)
        self.coll = TruckCollision(lot, self.model)
        self.hitch_bin = math.radians(15.0)

    # --- helpers ---------------------------------------------------------
    def _key(self, s):
        c = self.cfg
        return (int(s[0] // c.xy_res), int(s[1] // c.xy_res),
                int((mod2pi(s[2]) + math.pi) // (2 * math.pi / c.yaw_bins)) % c.yaw_bins,
                int(math.floor(mod2pi(s[2] - s[3]) / self.hitch_bin)))

    def _h(self, s):
        i = min(int(s[0] / self.hres), self.hdist.shape[0] - 1)
        j = min(int(s[1] / self.hres), self.hdist.shape[1] - 1)
        # small extra term: straightening the trailer costs something
        return self.cfg.h_weight * (self.hdist[i, j] + 2.0 * abs(mod2pi(s[2] - s[3])))

    def at_goal(self, s):
        G = self.goal
        fx, _ = self.model.front(s)
        gap = G.wall_x - fx
        tx = s[0] - self.model.d1 * math.cos(s[3])
        ty = s[1] - self.model.d1 * math.sin(s[3])
        return (G.nose_gap[0] <= gap <= G.nose_gap[1]
                and G.y_band[0] <= s[1] <= G.y_band[1] and G.y_band[0] <= ty <= G.y_band[1]
                and abs(mod2pi(s[2])) < G.ang_tol and abs(mod2pi(s[3])) < G.trailer_tol
                and tx < s[0])

    def _expand(self, s, prim):
        truck = propagate(s[:3], prim, self.cfg.step, self.cfg.sample_ds)
        th1 = trailer_rollout(s[3], truck, s[:3], self.model.d1)
        return np.column_stack([truck, th1])

    def _prim_cost(self, prim, parent):
        c = self.cfg
        d, _, steer, _ = prim
        cost = c.step * (1.0 if d > 0 else c.reverse_cost)
        cost += c.step * c.steer_cost * abs(steer)
        cost += c.steer_change_cost * abs(steer - parent.steer)
        if parent.dir != 0 and d != parent.dir:
            cost += c.switch_cost
        return cost

    def _try_analytic(self, node):
        best = None
        s = node.state
        for g in self.rs_goals:
            for i, (_, L, T) in enumerate(rs_paths(s[:3], g, self.model.rs_radius)):
                if i >= self.cfg.rs_max_candidates:
                    break
                cost, prev = 0.0, node.dir
                for l in L:
                    d = 1 if l >= 0 else -1
                    cost += abs(l) * (1.0 if d > 0 else self.cfg.reverse_cost)
                    if prev != 0 and d != prev:
                        cost += self.cfg.switch_cost
                    prev = d
                if best and cost >= best[2]:
                    continue
                pts, dirs = sample_path(s[:3], L, T, self.model.rs_radius, self.cfg.sample_ds)
                th1 = trailer_rollout(s[3], pts[1:], s[:3], self.model.d1)
                traj = np.column_stack([pts[1:], th1])
                if not self.at_goal(traj[-1]) or self.coll.hits(traj):
                    continue
                best = (traj, dirs[1:], cost)
        return best

    def _reconstruct(self, nodes, key, tail=None):
        segs, dsegs = [], []
        if tail is not None:
            segs.append(tail[0])
            dsegs.append(tail[1])
        while key is not None:
            n = nodes[key]
            if n.traj is not None:
                segs.append(n.traj)
                dsegs.append(np.full(len(n.traj), n.dir))
            else:
                segs.append(np.array([n.state]))
                dsegs.append(np.array([1]))
            key = n.parent
        return np.concatenate(segs[::-1]), np.concatenate(dsegs[::-1])

    # --- main entry ------------------------------------------------------
    def plan(self, start, goal: TruckGoal):
        t0 = time.time()
        cfg = self.cfg
        self.goal = goal
        start = tuple(float(v) for v in start)
        if self.coll.hits(np.array([start])):
            return PlanResult(False, reason="start pose is in collision", time=time.time() - t0)
        if all(self.coll.hits(self._expand(start, p)) for p in self.model.prims):
            return PlanResult(False, reason="rig is boxed in at the start (no move is free)",
                              time=time.time() - t0)

        # Reeds-Shepp targets for the truck: a few nose gaps along the parking row
        f = self.model.cab.front
        yc = 0.5 * sum(goal.y_band)
        self.rs_goals = [(goal.wall_x - gap - f, yc, 0.0) for gap in (0.3, 0.7, 1.1)]
        self.rs_goals = [g for g in self.rs_goals
                         if not self.coll.hits(np.array([[*g, 0.0]]))]
        if not self.rs_goals:
            return PlanResult(False, reason="parked pose is in collision (row blocked)",
                              time=time.time() - t0)
        self.hdist, self.hres = approach_heuristic(self.lot, self.rs_goals[0][:2])
        if not np.isfinite(self._h(start)):
            return PlanResult(False, reason="goal unreachable: parking area is walled off",
                              time=time.time() - t0)

        root = _Node(self._h(start), 0.0, start, None, None, 0, 0.0)
        k0 = self._key(start)
        nodes = {k0: root}
        heap = [(root.f, 0, k0)]
        closed = set()
        counter, exp = 1, 0
        while heap:
            if time.time() - t0 > cfg.timeout:
                return PlanResult(False, reason=f"timeout after {cfg.timeout:.0f} s",
                                  expansions=exp, time=time.time() - t0)
            if exp >= cfg.max_expansions:
                return PlanResult(False, reason="expansion limit reached",
                                  expansions=exp, time=time.time() - t0)
            _, _, key = heapq.heappop(heap)
            if key in closed:
                continue
            closed.add(key)
            node = nodes[key]
            exp += 1

            if self.at_goal(node.state):
                p, d = self._reconstruct(nodes, key)
                return PlanResult(True, p, d, "reached goal region", exp, time.time() - t0, node.g)

            # RS shots are expensive with the trailer rollout: only from the parking row's
            # neighborhood, roughly facing East, and only every few expansions there.
            s = node.state
            near = (s[1] < 9.0 and abs(mod2pi(s[2])) < math.radians(70)
                    and math.hypot(s[0] - self.rs_goals[0][0], s[1] - self.rs_goals[0][1]) < 22.0)
            if (near and exp % 3 == 0) or exp % (5 * cfg.rs_every) == 0:
                shot = self._try_analytic(node)
                if shot is not None:
                    p, d = self._reconstruct(nodes, key, (shot[0], shot[1]))
                    return PlanResult(True, p, d, "analytic Reeds-Shepp expansion",
                                      exp, time.time() - t0, node.g + shot[2])

            for prim in self.model.prims:
                traj = self._expand(node.state, prim)
                end = tuple(float(v) for v in traj[-1])
                ck = self._key(end)
                if ck in closed or ck == key or self.coll.hits(traj):
                    continue
                g = node.g + self._prim_cost(prim, node)
                old = nodes.get(ck)
                if old is None or g < old.g:
                    h = self._h(end)
                    if not np.isfinite(h):
                        continue
                    nodes[ck] = _Node(g + h, g, end, key, traj, prim[0], prim[2])
                    heapq.heappush(heap, (g + h, counter, ck))
                    counter += 1
        return PlanResult(False, reason="no path: search space exhausted",
                          expansions=exp, time=time.time() - t0)


def plan_truck(lot, truck: TruckTrailer = None, cfg=None):
    """NW start (rig straight, facing East, along the north wall) -> SE parking row."""
    truck = truck or TruckTrailer()
    planner = TruckHybridAStar(lot, truck, cfg)
    _, sy, _ = lot.start_pose()
    tail = truck.d1 + truck.trailer_length / 2          # hitch -> trailer rear
    start = (tail + 0.25, sy, 0.0, 0.0)                 # trailer rear 0.25 m off the wall
    walls = [c for c in lot.spec.wall_cells]
    wall_x = min(ix for ix, _ in walls) * CELL          # west face of the east walls
    return planner.plan(start, TruckGoal(wall_x=wall_x)), planner


if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt
    from environment import generate_lot
    from collision import CollisionChecker
    from planner import plot_result

    truck = TruckTrailer()
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    for ax, seed in zip(axs, (0, 4, 6, 10)):   # 10 = example of a graceful failure
        lot = generate_lot("truck", seed=seed)
        res, _ = plan_truck(lot, truck)
        ok = res.success and not CollisionChecker(lot, truck).path_in_collision(res.path)
        print(f"truck seed {seed}: success={res.success} exact_check_ok={ok} "
              f"t={res.time:5.1f}s exp={res.expansions} reason='{res.reason}'")
        plot_result(lot, truck, res, ax=ax, every=25)
    plt.tight_layout()
    plt.savefig("truck_demo.png", dpi=70)
    print("saved truck_demo.png")
    plt.show()   # close the window to exit