import numpy as np
#BOLD Signal
Vo = 0.018 # 0.018 standard param values; 0.04 is used in Havlicek2015 
E_0 = 0.4
TE = 0.026 # from experiment
#Old Parameter values as in Friston2005(?) 
k1 = 7.0*E_0 
k2 = 2.0
k3 = 2.0*E_0-0.2
#Parameters as in Havlicek2015
#theta0 = 188.1 
#k1 = 4.3*theta0*E_0*TE  
#k2 = 0 
#k3 = 1 
#Neuronal activity Z
A = None #1. #50.
#nr_nodes = 2
#Adj = np.identity(nr_nodes)
#Adj[1,0] = np.random.uniform(-0.75,0.75) #For Adj[1,0]=Adj[0,1] = 0.5, 0.,-0.33 and -0.5 works well #the Adj-Matrix must be positive definite!!
#Adj[0,1] = np.random.uniform(-0.75,0.75)
#try:
#    assert np.linalg.det(self.Adj)>0., "Adj must be positive definite"
#except:
#    Adj[1,0] = np.random.uniform(-0.75,0.75)
#    Adj[0,1] = np.random.uniform(-0.75,0.75)
B = 0.
C = 2.5 #Without imput the problem becomes much more dificult and the solutions are bias towards uncoupled systems
t_on = None
t_off = None
noise_factor = 0.001 #0.01 gives results for connectivity estimation for A=1; 0.001 works for both A=1 and A=50

# Hemodynamic system: parameters for s 
epsilon= 0.8
tau_s = 1.54
tau_f = 2.44 #Learned tau_f=2.99  #2.44 is the standard
# Balloon model: v and q
tau_0 = 1.02 #Learned tau_0 = 1.84 # 1.02 is the standard value
alpha=0.32
