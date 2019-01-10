import numpy as np
from numpy import random as rnd
from matplotlib import pyplot as plt

t_ind = 0

def F(state,t):
    # X = [z,s,f,v,q]
    # Neuronal Activity z
    A=-1.
    B=0.
    C=5.0
    t_on =  np.arange(5.0,T,10.) # np.array([1.])#
    t_off = np.arange(5.15,T,10.) # np.array([1.15])#

    def I(t):
        global t_ind
        inp = 0.0
        if t>=t_on[t_ind] and t<t_off[t_ind]:
            inp = 1.0
        if np.isclose(t,t_off[t_ind]) and t < t_off[-1]:
            t_ind += 1
        return inp

    z = state[0]
    inp = I(t)
    z_dot = (A + B*inp)*z + C*inp

    # Hemodynamic system: parameters for s 
    epsilon=0.8
    tau_s=1.54
    tau_f=2.44

    s = state[1]
    f = state[2]
    s_dot = epsilon*z - s/tau_s - (f - 1)/tau_f
    f_dot = s

    # Balloon model: v and q
    tau_0=1.02
    alpha=0.32
    alpha=1.0/alpha

    v = state[3]
    q = state[4]

    v_dot = (f - v**(alpha))/tau_0
    q_dot = ( f*(1.-(1.-E_0)**(1.0/f))/E_0 - (v**(alpha-1))*q )/tau_0

    Drift = np.zeros_like(state)
    Drift[0] = z_dot
    Drift[1] = s_dot
    Drift[2] = f_dot
    Drift[3] = v_dot
    Drift[4] = q_dot

    return Drift

def BOLD(state):
    state_obs = state[-2:] # this is the states on which the signal depends;  doesn't have to be the same dimensions as the obs_signal: for DCM the dynamic variables are v and q ( X = [z,s,f,v,q])

    v = state_obs[0]
    q = state_obs[1]

    Vo = 0.018
    k1=7.0*E_0
    k2=2.0
    k3=2.0*E_0-0.2

    return Vo*( k1*(1.-q) + k2*(1.-q/v) + k3*(1.-v) )

#model-specific parameters
dim = 5 # X = [z,s,f,v,q]
noise_covariance = np.zeros([dim,1])
noise_covariance[0] = 0.15**2

E_0=0.4
dt = 0.01
T = 16.0
steps = int(T/dt)

dim_obs = 1
var_obs = 0.003**2
obs_step = 40
obs_time = dt*np.arange(obs_step,steps+1,obs_step)

x = np.zeros([steps+1,dim])
x[0,-3:] = 1 
bold = np.zeros([steps+1])
observations = np.zeros([len(obs_time),dim_obs])

for t in range(steps):
    x[t+1,:] = x[t,:] + F(x[t,:],t*dt)*dt + np.sqrt(dt*noise_covariance.T)*rnd.randn()
    bold[t+1] = BOLD(x[t+1,:]) + np.sqrt(var_obs)*rnd.randn()
    if np.any(dt*(t+1) == obs_time):
        observations[dt*(t+1) == obs_time,:] = bold[t+1].reshape(np.shape(observations[dt*(t+1) == obs_time,:]))
        
#plt.figure()
#plt.plot(dt*np.arange(steps+1),x[0,:,:])
#plt.figure()
#plt.plot(dt*np.arange(steps+1),bold[0])
#plt.plot(obs_time,observations[0],'r*')
#plt.show()

print 'Data generated: '
print 'Data: ', observations.T
print 'Time points: ', obs_time
print '# of observations: ', len(obs_time)

np.savez('Data_SmoothingProblem/time_series_DCM_'+str(len(obs_time))+'obs.npz',obs_time = obs_time, observations = observations)
np.savez('Data_SmoothingProblem/Ground_truth_DCM_'+str(len(obs_time))+'obs.npz',x=x, bold=bold)
