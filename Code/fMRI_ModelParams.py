Subject = '2'
ROI = 'Toy'
trial = 2 #Hemodyn parameters were learned with trail 1 for both subjects 1 and 2    
#BOLD Signal
Vo = 0.04 # 0.018 standard param values; 0.04 is used in Havlicek2015 
E_0 = 0.4
TE = 0.026 # from experiment
#Old Parameter values as in Friston2005(?) 
#k1 = 7.0*E_0 
#k2 = 2.0
#k3 = 2.0*E_0-0.2
#Parameters as in Havlicek2015
theta0 = 188.1 
k1 = 4.3*theta0*E_0*TE  
k2 = 0 
k3 = 1 
#Neuronal activity Z
A = 50.
B = 0.
C = 0.
t_on = 0.
t_off = 0.
noise_factor = 0.3

#Hemodynamic system: parameters for s 
epsilon= 0.8
tau_s = 1.54
tau_f = None #2.44 is the standard
# Balloon model: v and q
tau_0 = None # 1.02 is the standard value;
alpha=0.32
