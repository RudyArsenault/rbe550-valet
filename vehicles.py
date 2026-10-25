"""vehicles.py

All vehicles share the same interface:
    state  : tuple/array, vehicle-specific (see each class)
    step(state, control, dt) -> new state     (kinematic integration, RK4)
    footprints(state) -> list of 4x2 arrays   (oriented bounding-box corners, for
                                               collision checking and plotting)

Models 
  DiffDriveRobot : state (x, y, th),            control (v, omega)
  Car            : state (x, y, th),            control (v, phi)   (Ackermann, bicycle model,
                                                 (x, y) is the REAR AXLE center)
  TruckTrailer   : state (x, y, th0, th1),      control (v, phi)   (hitch at the rear axle
                                                 of the truck; trailer is passive)
"""
from dataclasses import dataclass
import numpy as np


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def box_corners(x, y, th, length, width, back, front=None):
    #Corners of a rectangle whose reference point is (x, y), heading th

    if front is None:
        front = length - back
    c, s = np.cos(th), np.sin(th)
    local = np.array([[-back, width / 2], [-back, -width / 2],
                      [front, -width / 2], [front, width / 2]])
    R = np.array([[c, -s], [s, c]])
    return local @ R.T + np.array([x, y])


def rk4(f, state, control, dt):
    s = np.asarray(state, dtype=float)
    k1 = f(s, control)
    k2 = f(s + 0.5 * dt * k1, control)
    k3 = f(s + 0.5 * dt * k2, control)
    k4 = f(s + dt * k3, control)
    return s + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)



#robot: differential drive 

@dataclass
class DiffDriveRobot:
    length: float = 0.70
    width: float = 0.57
    v_max: float = 1.0        # m/s
    w_max: float = 1.5        # rad/s
    name: str = "robot"

    def deriv(self, s, u):
        v, w = u
        return np.array([v * np.cos(s[2]), v * np.sin(s[2]), w])

    def step(self, state, control, dt):
        v, w = control
        control = (np.clip(v, -self.v_max, self.v_max), np.clip(w, -self.w_max, self.w_max))
        out = rk4(self.deriv, state, control, dt)
        out[2] = wrap(out[2])
        return out

    def footprints(self, state):
        x, y, th = state[:3]
        # reference point = center of the wheel axis = box center
        return [box_corners(x, y, th, self.length, self.width, back=self.length / 2)]



#car: Ackermann steering, kinematic bicycle model referenced to the rear axle

@dataclass
class Car:
    length: float = 5.2
    width: float = 1.8
    wheelbase: float = 2.8
    phi_max: float = np.deg2rad(35.0)   # max steering angle (tunable)
    v_max: float = 2.0
    name: str = "car"

    @property
    def rear_overhang(self):
        return (self.length - self.wheelbase) / 2.0   # assumes symmetric overhangs

    @property
    def min_turn_radius(self):
        return self.wheelbase / np.tan(self.phi_max)

    def deriv(self, s, u):
        v, phi = u
        return np.array([v * np.cos(s[2]), v * np.sin(s[2]), v / self.wheelbase * np.tan(phi)])

    def step(self, state, control, dt):
        v, phi = control
        control = (np.clip(v, -self.v_max, self.v_max),
                   np.clip(phi, -self.phi_max, self.phi_max))
        out = rk4(self.deriv, state, control, dt)
        out[2] = wrap(out[2])
        return out

    def footprints(self, state):
        x, y, th = state[:3]
        return [box_corners(x, y, th, self.length, self.width, back=self.rear_overhang)]



#truck with one trailer, Hitch is at the truck's rear axle center (x, y)
#th0 = truck heading, th1 = trailer heading d1 = hitch -> trailer axle center.
#   th0' = v/L * tan(phi)
#   th1' = v/d1 * sin(th0 - th1)

@dataclass
class TruckTrailer:
    length: float = 5.4
    width: float = 2.0
    wheelbase: float = 3.4
    trailer_length: float = 4.5
    trailer_width: float = 2.0
    d1: float = 5.0
    phi_max: float = np.deg2rad(30.0)
    v_max: float = 2.0
    hitch_angle_max: float = np.deg2rad(70.0)   #jackknife limit |th0 - th1|
    name: str = "truck"

    @property
    def rear_overhang(self):
        return (self.length - self.wheelbase) / 2.0

    def deriv(self, s, u):
        v, phi = u
        th0, th1 = s[2], s[3]
        return np.array([v * np.cos(th0), v * np.sin(th0),
                         v / self.wheelbase * np.tan(phi),
                         v / self.d1 * np.sin(th0 - th1)])

    def step(self, state, control, dt):
        v, phi = control
        control = (np.clip(v, -self.v_max, self.v_max),
                   np.clip(phi, -self.phi_max, self.phi_max))
        out = rk4(self.deriv, state, control, dt)
        out[2] = wrap(out[2])
        out[3] = wrap(out[3])
        return out

    def jackknifed(self, state):
        return abs(wrap(state[2] - state[3])) > self.hitch_angle_max

    def trailer_axle(self, state):
        x, y, _, th1 = state
        return np.array([x - self.d1 * np.cos(th1), y - self.d1 * np.sin(th1)])

    def footprints(self, state):
  
        x, y, th0, th1 = state
        truck = box_corners(x, y, th0, self.length, self.width, back=self.rear_overhang)
        ax, ay = self.trailer_axle(state)
        trailer = box_corners(ax, ay, th1, self.trailer_length, self.trailer_width,
                              back=self.trailer_length / 2)
        return [truck, trailer]


def make_vehicle(kind):
    return {"robot": DiffDriveRobot, "car": Car, "truck": TruckTrailer}[kind]()


if __name__ == "__main__":
    # check: car drives a circle of radius L / tan(phi).
    car = Car()
    s = np.array([0.0, 0.0, 0.0])
    phi = np.deg2rad(20)
    R = car.wheelbase / np.tan(phi)
    dt = 0.01
    T = 2 * np.pi * R / 1.0
    for _ in range(int(round(T / dt))):
        s = car.step(s, (1.0, phi), dt)
    print(f"car after one lap: x={s[0]:.3f}, y={s[1]:.3f}, th={s[2]:.3f} (expect ~0, 0, 0)")