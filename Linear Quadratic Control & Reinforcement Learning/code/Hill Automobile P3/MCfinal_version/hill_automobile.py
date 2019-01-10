#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def truncate(x, a=-np.inf, b=np.inf):
  if x < a:
    x = a
  if x > b:
    x = b
  return x

class HillAutomobile:
  def __init__(self, mass=0.2, friction=0.3, dt=0.1): 
    self.pos_list = list()
    self.grav = 9.8
    self.friction = friction
    self.dt = dt  # seconds
    self.mass = mass
    self.pos = -0.5
    self.vel = 0.0

  def reset(self, exploring_starts=True, init_pos=-0.5, init_vel = 0.0):
    if exploring_starts:
        init_pos = np.random.uniform(-1.2,0.5)
        init_vel = np.random.uniform(-1.5,1.5)
    truncate(init_pos,-1.2,0.5)
    truncate(init_vel,-1.5,1.5)
    self.pos_list = [init_pos]
    self.pos = init_pos
    self.vel = init_vel
    return [self.pos, self.vel]

  def update(self, u):
    # update the position and velocity of the automobile
    # controls u are integers=[0,1,2]
    if(u >= 3):
        raise ValueError("Error: the control " + str(u) + " is out of range.")
    done = False
    cost = 0.01
    control_list = [-0.2, 0, +0.2]
    u_t = control_list[u]
    velocity_t1 = self.vel + (-self.grav * self.mass * np.cos(3*self.pos)
                   + (u_t/self.mass) - (self.friction*self.vel)) * self.dt
    position_t1 = self.pos + (velocity_t1 * self.dt)
    # Ensure the automobile stays within bounds
    if position_t1 < -1.2:
        position_t1 = -1.2
        velocity_t1 = 0
    # Assign the new position and velocity
    self.pos = position_t1
    self.vel= velocity_t1
    self.pos_list.append(position_t1)
    # Cost and done when the automobile reaches the goal
    if position_t1 >= 0.5:
        cost = -1.0
        done = True
    return [position_t1, velocity_t1], cost, done

  def render(self):
    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-1.2, 0.5), ylim=(-1.1, 1.1))
    ax.grid(False)
    x_sin = np.linspace(start=-1.2, stop=0.5, num=100)
    y_sin = np.sin(3 * x_sin)
    ax.plot(x_sin, y_sin)
    dot, = ax.plot([], [], 'ro')
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    _position_list = self.pos_list
    _dt = self.dt

    def _init():
        dot.set_data([], [])
        time_text.set_text('')
        return dot, time_text

    def _animate(i):
        x = _position_list[i]
        y = np.sin(3 * x)
        dot.set_data(x, y)
        time_text.set_text("Time: " + str(np.round(i*_dt, 1)) + "s" + '\n' + "Frame: " + str(i))
        return dot, time_text

    ani = animation.FuncAnimation(fig, _animate, np.arange(1, len(self.pos_list)),
                                  blit=True, init_func=_init, repeat=False)
    total_time = len(self.pos_list) * _dt

    print('Arriving Time:', total_time)
    plt.show()
      


