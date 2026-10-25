"""reeds_shepp.py - Reeds-Shepp curves (shortest paths for a car that can drive both ways).

Used by the Hybrid A* planner for its analytic expansion: from a search node, try to
connect straight to the goal pose with a Reeds-Shepp curve and collision-check it.

Path families follow Reeds & Shepp (1990), "Optimal paths for a car that goes both
forwards and backwards", in the normalized form used by OMPL and PythonRobotics
(CSC, CCC, CCCC, CCSC, CCSCC), with the time-flip / reflect / backwards symmetries.
Every candidate is VERIFIED by integrating it and checking it lands on the goal pose,
so a mis-derived family can only cost an option, never return a wrong path.

AI-assisted code adapted from the published formulas above.
"""
import math
import numpy as np

PI = math.pi


def mod2pi(a):
    """Wrap to [-pi, pi)."""
    return (a + PI) % (2 * PI) - PI


def polar(x, y):
    return math.hypot(x, y), math.atan2(y, x)


# ---------------------------------------------------------------------------
# Base families in normalized coordinates (turning radius 1, start at origin, heading 0).
# Each returns a list of (segment lengths, segment types) or None.
# Negative length = driving backward.
# ---------------------------------------------------------------------------
def _LSL(x, y, phi):
    u, t = polar(x - math.sin(phi), y - 1.0 + math.cos(phi))
    if t >= 0.0:
        v = mod2pi(phi - t)
        if v >= 0.0:
            return [t, u, v], "LSL"
    return None


def _LSR(x, y, phi):
    u1, t1 = polar(x + math.sin(phi), y - 1.0 - math.cos(phi))
    u1 = u1 * u1
    if u1 >= 4.0:
        u = math.sqrt(u1 - 4.0)
        theta = math.atan2(2.0, u)
        t = mod2pi(t1 + theta)
        v = mod2pi(t - phi)
        if t >= 0.0 and v >= 0.0:
            return [t, u, v], "LSR"
    return None


def _LxRxL(x, y, phi):
    u1, theta = polar(x - math.sin(phi), y - 1.0 + math.cos(phi))
    if u1 <= 4.0:
        A = math.acos(0.25 * u1)
        t = mod2pi(A + theta + PI / 2)
        u = mod2pi(PI - 2 * A)
        v = mod2pi(phi - t - u)
        return [t, -u, v], "LRL"
    return None


def _LxRL(x, y, phi):
    u1, theta = polar(x - math.sin(phi), y - 1.0 + math.cos(phi))
    if u1 <= 4.0:
        A = math.acos(0.25 * u1)
        t = mod2pi(A + theta + PI / 2)
        u = mod2pi(PI - 2 * A)
        v = mod2pi(-phi + t + u)
        return [t, -u, -v], "LRL"
    return None


def _LRxL(x, y, phi):
    u1, theta = polar(x - math.sin(phi), y - 1.0 + math.cos(phi))
    if 0.0 < u1 <= 4.0:
        u = math.acos(1 - u1 * u1 * 0.125)
        s = 2 * math.sin(u) / u1
        if abs(s) > 1.0:
            return None
        A = math.asin(s)
        t = mod2pi(-A + theta + PI / 2)
        v = mod2pi(t - u - phi)
        return [t, u, -v], "LRL"
    return None


def _LRxLR(x, y, phi):
    u1, theta = polar(x + math.sin(phi), y - 1.0 - math.cos(phi))
    if u1 <= 2.0:
        A = math.acos((u1 + 2) * 0.25)
        t = mod2pi(theta + A + PI / 2)
        u = mod2pi(A)
        v = mod2pi(phi - t + 2 * u)
        if t >= 0 and u >= 0 and v >= 0:
            return [t, u, -u, -v], "LRLR"
    return None


def _LxRLxR(x, y, phi):
    u1, theta = polar(x + math.sin(phi), y - 1.0 - math.cos(phi))
    if u1 < 1e-9:
        return None
    u2 = (20 - u1 * u1) / 16
    if 0 <= u2 <= 1:
        u = math.acos(u2)
        s = 2 * math.sin(u) / u1
        if abs(s) > 1.0:
            return None
        A = math.asin(s)
        t = mod2pi(theta + A + PI / 2)
        v = mod2pi(t - phi)
        if t >= 0 and v >= 0:
            return [t, -u, -u, v], "LRLR"
    return None


def _LxR90SL(x, y, phi):
    u1, theta = polar(x - math.sin(phi), y - 1.0 + math.cos(phi))
    if u1 >= 2.0:
        r = math.sqrt(u1 * u1 - 4)
        u = r - 2
        A = math.atan2(2, r)
        t = mod2pi(theta + A + PI / 2)
        v = mod2pi(t - phi + PI / 2)
        if t >= 0 and v >= 0:
            return [t, -PI / 2, -u, -v], "LRSL"
    return None


def _LSR90xL(x, y, phi):
    u1, theta = polar(x - math.sin(phi), y - 1.0 + math.cos(phi))
    if u1 >= 2.0:
        r = math.sqrt(u1 * u1 - 4)
        u = r - 2
        A = math.atan2(r, 2)
        t = mod2pi(theta - A + PI / 2)
        v = mod2pi(t - phi - PI / 2)
        if t >= 0 and v >= 0:
            return [t, u, PI / 2, -v], "LSRL"
    return None


def _LxR90SR(x, y, phi):
    u1, theta = polar(x + math.sin(phi), y - 1.0 - math.cos(phi))
    if u1 >= 2.0:
        t = mod2pi(theta + PI / 2)
        u = u1 - 2
        v = mod2pi(phi - t - PI / 2)
        if t >= 0 and v >= 0:
            return [t, -PI / 2, -u, -v], "LRSR"
    return None


def _LSL90xR(x, y, phi):
    u1, theta = polar(x + math.sin(phi), y - 1.0 - math.cos(phi))
    if u1 >= 2.0:
        t = mod2pi(theta)
        u = u1 - 2
        v = mod2pi(phi - t - PI / 2)
        if t >= 0 and v >= 0:
            return [t, u, PI / 2, -v], "LSLR"
    return None


def _LxR90SL90xR(x, y, phi):
    u1, theta = polar(x + math.sin(phi), y - 1.0 - math.cos(phi))
    if u1 >= 4.0:
        r = math.sqrt(u1 * u1 - 4)
        u = r - 4
        A = math.atan2(2, r)
        t = mod2pi(theta + A + PI / 2)
        v = mod2pi(t - phi)
        if t >= 0 and v >= 0:
            return [t, -PI / 2, -u, -PI / 2, v], "LRSLR"
    return None


_FAMILIES = [_LSL, _LSR, _LxRxL, _LxRL, _LRxL, _LRxLR, _LxRLxR,
             _LxR90SL, _LSR90xL, _LxR90SR, _LSL90xR, _LxR90SL90xR]

_SWAP = str.maketrans("LR", "RL")


def _integrate(lengths, types, x=0.0, y=0.0, th=0.0):
    """Endpoint of a normalized path (radius 1)."""
    for d, c in zip(lengths, types):
        if c == "S":
            x += d * math.cos(th)
            y += d * math.sin(th)
        else:
            k = 1.0 if c == "L" else -1.0
            th2 = th + k * d
            x += (math.sin(th2) - math.sin(th)) / k
            y -= (math.cos(th2) - math.cos(th)) / k
            th = th2
    return x, y, th


def _candidates(x, y, phi):
    """All verified normalized paths from (0,0,0) to (x, y, phi)."""
    out = []
    xb = x * math.cos(phi) + y * math.sin(phi)
    yb = x * math.sin(phi) - y * math.cos(phi)
    for f in _FAMILIES:
        trials = []
        for (a, b, c), tf, rf in (((x, y, phi), 1, False), ((-x, y, -phi), -1, False),
                                  ((x, -y, -phi), 1, True), ((-x, -y, phi), -1, True)):
            r = f(a, b, c)
            if r:
                L, T = r
                trials.append(([tf * l for l in L], T.translate(_SWAP) if rf else T))
        # "backwards" variants: run the word in reverse
        for (a, b, c), tf, rf in (((xb, yb, phi), 1, False), ((-xb, yb, -phi), -1, False),
                                  ((xb, -yb, -phi), 1, True), ((-xb, -yb, phi), -1, True)):
            r = f(a, b, c)
            if r:
                L, T = r
                L = [tf * l for l in L][::-1]
                T = (T.translate(_SWAP) if rf else T)[::-1]
                trials.append((L, T))
        for L, T in trials:
            ex, ey, eth = _integrate(L, T)
            if abs(ex - x) < 1e-4 and abs(ey - y) < 1e-4 and abs(mod2pi(eth - phi)) < 1e-4:
                out.append((L, T))
    return out


def rs_paths(start, goal, radius):
    """All verified Reeds-Shepp paths, shortest first.

    Returns a list of (length_m, lengths_m, types). lengths_m are signed meters.
    """
    dx, dy = goal[0] - start[0], goal[1] - start[1]
    c, s = math.cos(start[2]), math.sin(start[2])
    x = (c * dx + s * dy) / radius
    y = (-s * dx + c * dy) / radius
    phi = mod2pi(goal[2] - start[2])
    res = []
    for L, T in _candidates(x, y, phi):
        Lm = [l * radius for l in L if abs(l) > 1e-10]
        Tm = "".join(t for l, t in zip(L, T) if abs(l) > 1e-10)
        res.append((sum(abs(l) for l in Lm), Lm, Tm))
    res.sort(key=lambda r: r[0])
    return res


def sample_path(start, lengths, types, radius, step=0.1):
    """Dense samples (N x 3) along a path, plus the driving direction (+1/-1) per sample."""
    x, y, th = start
    pts, dirs = [(x, y, th)], [1 if lengths and lengths[0] >= 0 else -1]
    for d, c in zip(lengths, types):
        n = max(1, int(math.ceil(abs(d) / step)))
        ds = d / n
        sgn = 1 if d >= 0 else -1
        for _ in range(n):
            if c == "S":
                x += ds * math.cos(th)
                y += ds * math.sin(th)
            else:
                k = (1.0 if c == "L" else -1.0) / radius
                th2 = th + k * ds
                x += (math.sin(th2) - math.sin(th)) / k
                y -= (math.cos(th2) - math.cos(th)) / k
                th = th2
            pts.append((x, y, mod2pi(th)))
            dirs.append(sgn)
    return np.array(pts), np.array(dirs)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    worst, n_ok = 0.0, 0
    for _ in range(2000):
        s = (0.0, 0.0, 0.0)
        g = (rng.uniform(-10, 10), rng.uniform(-10, 10), rng.uniform(-PI, PI))
        P = rs_paths(s, g, 2.0)
        if P:
            n_ok += 1
            _, L, T = P[0]
            pts, _ = sample_path(s, L, T, 2.0, 0.05)
            err = np.hypot(*(pts[-1, :2] - g[:2]))
            worst = max(worst, err)
    print(f"solved {n_ok}/2000 random queries, worst endpoint error {worst:.2e} m")