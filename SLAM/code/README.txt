The top level module for this project is SLAM.py, run the code to get occupancy grid map, and texture map. The occupancy grid map show first called figure(1), after close the window, the figure(2) I plot the trajectoy of the robot, the figure(3) of the texture map will pop out.

basically, in this SLAM file it only run grid map, if want to run the texture map, please comment out the line 175 to 182, and modify the load data at top, which i already write, and just comment out the code for specific data set.

Also at the line 130, is the controller for the loop, num, is the stop point, and 100 is my timestamp


the mapping.py and texture.py are the files i write for my top level module, which contain the mapping construction function and texture map function respectively.

The localization.py file is my particle filter which contain the predict, update and resampling step.


the load_data.py and p3_utils.py are files from professor.