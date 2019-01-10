"""
===========================
Inverted pendulum animation
===========================

Adapted from the double pendulum problem animation.
https://matplotlib.org/examples/animation/double_pendulum_animated.html
"""



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation



###############################################################
### Modify this section with your inverted pendulum results ###

# change with your time discretization size
dt = 0.05
t = np.arange(0.0, 20, dt)


# the current animation just shows a rotating pendulum
# change with your results
theta = np.linspace(-np.pi, np.pi, t.shape[0])
x1 = np.sin(theta)
y1 = np.cos(theta)
u = np.zeros(t.shape[0])
################################################################

fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
ax.grid()
ax.axis('equal')
plt.axis([-2, 2, -2, 2])

line, = ax.plot([], [], 'o-', lw=2)
time_template = 'time = %.1fs \nangle = %.2frad\ncontrol = %.2f'
time_text = ax.text(0.05, 0.8, '', transform=ax.transAxes)


def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text


def animate(i):
    thisx = [0, x1[i]]
    thisy = [0, y1[i]]

    line.set_data(thisx, thisy)
    time_text.set_text(time_template % (t[i], theta[i], u[i]))
    return line, time_text

ani = animation.FuncAnimation(fig, animate, np.arange(1, t.shape[0]),
                              interval=25, blit=True, init_func=init)
plt.show()
