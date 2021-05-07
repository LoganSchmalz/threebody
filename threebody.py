# An implementation of the three-body problem by Logan Schmalz
# https://github.com/LoganSchmalz/threebody/
# MIT License

import numpy as np
import scipy as sci
import scipy.integrate
import scipy.linalg
import matplotlib.pyplot as plt

# As astronomers, we like to normalize values to scales that make sense
# So that's what we'll do here
# Using these norm values, we will compute some constants that make multiplying easier
# This will also speed up calculations, since we will use floating-point numbers
# as opposed to fixed point, which are slower with big differences in exponents
G = 6.67408e-11 #Newton-meters^2/kg^2, Gravitational Constant
m_norm = 1.989e30 #kg, mass of Sun
r_norm = 1.496e11 #meters, 1 AU
v_norm = 29780 #meters/sec, speed of Earth around Sun
t_norm = 1*365*24*3600 #sec, orbital period of Earth
# And here are our new constants
K1 = G * t_norm * m_norm / (r_norm**2 * v_norm)
K2 = v_norm * t_norm / r_norm

# The body_calc function takes the current conditions as an array rvs
# It returns all of the drs and dvs together, to be added to rvs by our ODE
# function and plugged back in as rvs to solve again
def body_calc(rvs, t, m1, m2, m3):
    # Here we are extracting all our values from our current conditions array
    r1 = rvs[ :3]
    r2 = rvs[3:6]
    r3 = rvs[6:9]

    v1 = rvs[9:12]
    v2 = rvs[12:15]
    v3 = rvs[15:18]

    # Getting easy access to distance values between bodies
    r12 = sci.linalg.norm(r1-r2)
    r23 = sci.linalg.norm(r2-r3)
    r13 = sci.linalg.norm(r1-r3)

    # And doing our gravity calculations with our special constants
    dv1 = K1*(m2*(r2-r1)/r12**3 + m3*(r3-r1)/r13**3)
    dv2 = K1*(m1*(r1-r2)/r12**3 + m3*(r3-r2)/r23**3)
    dv3 = K1*(m1*(r1-r3)/r13**3 + m2*(r2-r3)/r23**3)
    # And finally determining our change in position
    dr1 = K2*v1
    dr2 = K2*v2
    dr3 = K2*v3

    # Then we want to send these back to our ODE function to reuse
    drs = np.concatenate((dr1, dr2, dr3)) # Takes in tuple to combine into
    dvs = np.concatenate((dv1, dv2, dv3)) # single array
    return np.concatenate((drs, dvs)) # Returns all the differences at once

# Sun
r1 = np.array([0,0,0])
v1 = np.array([0,0,0])
m1 = 1
# Venus
r2 = np.array([0,0.723332,0])
v2 = np.array([1.176,0,0])
#v2=np.array([2.352,0,0]) # twice Venus' normal velocity
m2 = 2.4472e-6
# Earth
r3 = np.array([1,0,0])
v3 = np.array([0,1,0])
m3 = 3.00269e-6

# Setup equal masses at the points of an equalateral triangle
# with velocities following the edges
#m1 = m2 = m3 = 1
#r1 = np.array([0,0,0])
#r2 = np.array([1,0,0])
#r3 = np.array([0.5,np.sqrt(1-0.5**2),0])
#v1 = np.array([.5,0,0])
#v2 = np.array([-0.25,np.sqrt(3)/4,0])
#v3 = np.array([-0.25,-np.sqrt(3)/4,0])

# combining all of our arrays into one to pass into our ODE solver
init_rvs = np.concatenate((r1, r2, r3, v1, v2, v3))

# generates a linear prograssion of times from 0 to 20
# the units are technically years, but the 0-20 years is divided
# into 10,000 intervals, so these are 0.002 years, or about 17.5 hours
times = np.linspace(0,20,10000)

# We use odeint as it is typically faster than integrate, due to underlying
# Python implementation, even though integrate is technically newer and more
# versatile
solution = sci.integrate.odeint(body_calc, init_rvs, times, args=(m1,m2,m3))

# Here we want to extract out position values at each time step
#
# Explanation:
#
# Solutions is a multidimensional array, we can think of it as
# a Cx6 matrix, where C is some constant for how many time steps we have
# the 6 comes from our 6 values (r1, r2, r3, v1, v2, and v3)
# these values themselves are 3-dimensional vectors
# In reality, the 6 dimensions and 3 dimensions are actually 'flattened' into
# one 18-dimensional vector.
#
# So for r1_sol for example:
# we want the first 3 values of our 18-dimensional vector
# which correspond to x1,y1,z1
# and these values at each timestep appear in all C
# so we use " : " to say that we want to be checking every C timestep
# and we use " :3" to say we want the first 3 values (again, x1,y1,z1).
#
# for r2_sol:
# we again want every value at each timestep, so we start with " : "
# and we use "3:6" to say we want the 4th, 5th, and 6th values (x2,y2,z2)
#
# similarly for r3_sol, we use "6:9" to get 7th, 8th, and 9th values
# if we wanted v1, we could use "9:12", but that's not very useful for us
#
# (note: in Python, arrays begin indexing at 0, thus for example the value
# in index 2 is the third value.
# in this sense, we can say " :3" is the same as writing "0:3", with the end
# being non-inclusive, so we get a0,a1,a2
# and for "3:6", we get a3,a4,a5)
# (extra note: the technical reason that it makes sense to allow a comma here
# is that numpy arrays can actually take a "tuple" of slice boundaries)
r1_sol = solution[ : ,  :3]
r2_sol = solution[ : , 3:6]
r3_sol = solution[ : , 6:9]

fig = plt.figure()
axs = fig.add_subplot(111)

# Plotting the objects' paths
# similarly here, we extract the first, second, third coordinates
# using " : " to go through every timestep, and then 0, 1, 2 as
# the index for which coordinate we want: 0=x, 1=y, 2=z
axs.plot(r1_sol[ : , 0], r1_sol[ : , 1], color="#ffa500")
axs.plot(r2_sol[ : , 0], r2_sol[ : , 1], color="#808080")
axs.plot(r3_sol[ : , 0], r3_sol[ : , 1], color="b")

# Plotting the objects' final locations
# and the -1 here means get final timestep
axs.scatter(r1_sol[-1,0], r1_sol[-1,1], color="#ffa500")
axs.scatter(r2_sol[-1,0], r2_sol[-1,1], color="#808080")
axs.scatter(r3_sol[-1,0], r3_sol[-1,1], color="b")

plt.show()
