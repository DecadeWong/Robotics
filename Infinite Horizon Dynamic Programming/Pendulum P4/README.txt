from line 176 to 241 is my discretiztion part

from line 244 to 267 is where I use the result of value iteration to balance the pendulum, and I save the result as myxuV.npy in src

from line 270 to 293 is where I use the result of policy iteration to balance the pendulum, and I save the result as myxuP.npy in src


Therefore, I have two files, and load them into the animation files.

in inverted_pendulum_animation, line 25 I load the PI result to balance the pendulum
in inverted_pendulum_animation, line 26 I load the VI result to balance the pendulum

it depends on which you want to see, you can comment either line to show the animation and light the line you want to implement.