#from __future__ import print_function
import sys
from time import time
import numpy as np
from mpi4py import MPI
from fMRI_Problem import fMRI_Problem as SP
from APIS import APIS as Smoother

comm = MPI.COMM_WORLD

Iters = 120
case_identifier = sys.argv[1]

meta_params = {}
meta_params["Iters"] = Iters
meta_params["steps_between_obs"] = 40
meta_params["N_particles"] = 1000    # particles per worker
meta_params["learning_rate"] = 0.005
meta_params["anneal_threshold"] = 0.001 # in fraction of particles
meta_params["anneal_factor"] = 1.15
ess_threshold = 0.1
params2update =  ['sigma_dyn'] # ['sigma_dyn','obs_signal']
#Instantiate objects
smproblem = SP(comm,meta_params,case_identifier)
apis = Smoother(smproblem)

if comm.Get_rank()==0: 
    print "Number of iterations:", Iters
    N_t = apis.timepoints
    array_pmeanZ = np.zeros([Iters,N_t])
    array_pmeanBold = np.zeros([Iters,N_t])
    array_meanZ = np.zeros([Iters,N_t])
    array_meanBold = np.zeros([Iters,N_t])
    array_OLC = np.zeros([Iters,smproblem.dim_control,N_t])
    array_feedbackMatrix = np.zeros([Iters,smproblem.dim_control,N_t])
    
    start_time = time()
for itr in np.arange(Iters):
    if comm.Get_rank()==0: print "Iteration ",itr
    apis.generate_particles()
    apis.get_statistics(itr)
    apis.adapt_initialization()
    apis.update_controller()
    apis.update_parameters(itr,ess_threshold,*params2update)
    apis.posterior_obssignal()
    
    local_meanZ = np.mean(apis.Particles[:,0,:],axis=0)
    global_meanZ = np.zeros(apis.Particles.shape[-1])
    comm.Reduce(local_meanZ,global_meanZ)  
    
    local_mean_obSignal = np.mean(apis.local_obsSignal,axis=0)
    global_mean_obsSignal = np.zeros(apis.local_obsSignal.shape[1:])
    comm.Reduce(local_mean_obSignal,global_mean_obsSignal)
    
    if comm.Get_rank()==0:
        array_pmeanZ[itr] = global_meanZ
        array_pmeanBold[itr] = global_mean_obsSignal
        
        array_meanZ[itr] = apis.mean_post[0]
        array_meanBold[itr] = apis.mean_postObsSignal
        
        array_OLC[itr] = smproblem.openloop_term
        array_feedbackMatrix[itr] = smproblem.feedback_term
     

    
if comm.Get_rank()==0:
    elapsed = (time() - start_time)
    print 'Elapsed time for ',Iters,' iterations: ', elapsed, 'sec' 
    print 'Per itaration is on average:',elapsed/Iters
    
apis.save_data() #Only saves if the flag -save is given in arg.sys
if comm.Get_rank()==0 and "-save" in sys.argv:
    apis.save_var("array_pmeanZ",array_pmeanZ)
    apis.save_var("array_pmeanBold",array_pmeanBold)
    
    apis.save_var("array_meanZ",array_meanZ)
    apis.save_var("array_meanBold",array_meanBold)
    
    apis.save_var("array_OLC",array_OLC)
    apis.save_var("array_feedbackMatrix",array_feedbackMatrix)
