import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation
from src import HW5_P2
#import HW5_P2
from collections import namedtuple
import sympy


def render(x,dt):
  """ Acrobot problem animation
  Adapted from the double pendulum problem animation.
  https://matplotlib.org/examples/animation/double_pendulum_animated.html
  """
  x1 = np.cos(x[:,0]+np.pi/2)
  y1 = np.sin(x[:,0]+np.pi/2)

  x2 = np.cos(x[:,0]+x[:,1]+np.pi/2) + x1
  y2 = np.sin(x[:,0]+x[:,1]+np.pi/2) + y1

  fig = plt.figure()
  ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2.2, 2.2), ylim=(-2.2, 2.2))
  ax.grid()

  line, = ax.plot([], [], 'o-', lw=2)
  time_template = 'time = %.1fs'
  time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

  def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text


  def animate(i):
    thisx = [0, x1[i], x2[i]]
    thisy = [0, y1[i], y2[i]]

    line.set_data(thisx, thisy)
    time_text.set_text(time_template % (i*dt))
    return line, time_text

  ani = animation.FuncAnimation(fig, animate, np.arange(1, len(x)),
                                interval=25, blit=True, init_func=init)
  plt.show()


def main():
  # create a time array 
  #this is for linearized model Q3
  Consts = namedtuple('Consts', ['r_const', 'g_const', 'xi_const', 'k_const', 'X_linearpoint'])
  myconsts = Consts(r_const = 1, g_const = 9.8, xi_const = 1, k_const = 1, X_linearpoint = np.array([0,0,0,0]))
  a, b, A0, B0, X_dot = HW5_P2.Linearizing (myconsts.r_const, myconsts.g_const, myconsts.xi_const, myconsts.X_linearpoint)
  acrobot = lambda x, u : A0 * x.reshape((4,1)) + B0 * u
  Q, stage_cost0 = HW5_P2.Q_approximation (myconsts.r_const, myconsts.k_const)
  M = HW5_P2.LQR (Q, A0, B0, myconsts.X_linearpoint)
  my_pi = lambda X: - 1/myconsts.r_const * B0.T * M * X.reshape((4,1))

  def acrobot_ode(x, t):
    #for linearized model
    u = my_pi (x)
    dxdt = acrobot(x, u)
    dxdt = (sympy.matrix2numpy(dxdt).astype(float)).squeeze()
    return dxdt

  dt = 1/30
  t = np.arange(0.0, 10, dt)
  # initial state
  x_init = 0.1*np.random.randn(4)
  print(x_init)
  # integrate the ODE using scipy.integrate.
  x = integrate.odeint(acrobot_ode, x_init, t)
  # display results
  render(x,dt)


  ##################################
  #this is for nonlinearized model Q4
  # Consts = namedtuple('Consts', ['r_const', 'g_const', 'xi_const', 'k_const', 'X_linearpoint'])
  # myconsts = Consts(r_const = 1, g_const = 9.8, xi_const = 1, k_const = 1, X_linearpoint = np.array([0,0,0,0]))
  # a, b, A0, B0, X_dot = HW5_P2.Linearizing (myconsts.r_const, myconsts.g_const, myconsts.xi_const, myconsts.X_linearpoint)
  # Q, stage_cost0 = HW5_P2.Q_approximation (myconsts.r_const, myconsts.k_const)
  # M = HW5_P2.LQR (Q, A0, B0, myconsts.X_linearpoint)
  # my_pi = lambda X: - 1/myconsts.r_const * B0.T * M * X.reshape((4,1))

  # # construction on nonlinearized system
  # dt = 1/30
  # t = np.arange(0.0, 5, dt)
  # x_init = 0.5*np.random.randn(4)  #np.array([0,0,0,0]) #initial condition
  # print(x_init)
  # x = integrate.odeint(HW5_P2.nonlinear_system, x_init, t, args=(a, b, my_pi))
  # render(x,dt)

  plt.plot(t, x[:, 0], 'b', label='theta1(t)')
  plt.plot(t, x[:, 1], 'r', label='theta2(t)')
  plt.legend(loc='best')
  plt.xlabel('t')
  plt.grid()
  plt.show()

if __name__ == "__main__":
  main()




