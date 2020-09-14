## exercise 0.5.2

import numpy as np
import matplotlib.pyplot as plt


# We simulate measurements every 100 ms for a period of 10 seconds 
t = np.arange(0, 10, 0.1)

# The data from the sensors are generated as either a sine or a cosine 
# with some Gaussian noise added.
sensor1 = 3*np.sin(t)+0.5*np.random.normal(size=len(t))
sensor2 = 3*np.cos(t)+0.5*np.random.normal(size=len(t))

# Change the font size to make axis and title readable
font_size = 15
plt.rcParams.update({'font.size': font_size})

# Define the name of the curves
legend_strings = ['Sensor 1', 'Sensor 2']

# Start plotting the simulated measurements
plt.figure(1)
# Plot the sensor 1 output as a function of time, and
# make the curve red and fully drawn
plt.plot(t, sensor1, 'r-')

# Plot the sensor 2 output as a function of time, and
# make the curve blue and dashed
plt.plot(t, sensor2, 'b--')

# Ensure that the limits on the axis fit the data 
plt.axis('tight')

# Add a grid in the background
plt.grid()

# Add a legend describing each curve, place it at the "best" location
# so as to minimize the amount of curve it covers
plt.legend(legend_strings,loc='best')

# Add labels to the axes
plt.xlabel('Time [s]')
plt.ylabel('Voltage [mV]')

# Add a title to the plot
plt.title('Sensor outputs')

# Export the figure
plt.savefig('ex1_5_2.png')

# Show the figure in the console
plt.show()
