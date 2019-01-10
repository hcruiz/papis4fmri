#from __future__ import print_function
import sys
from time import time
import numpy as np
from mpi4py import MPI
from Net_fMRI_Problem import fMRI_Problem as SP #from Test_Problem import Test_Problem as SP #
from APIS_Net import APIS_Net as Smoother

comm = MPI.COMM_WORLD

Iters = 200
case_identifier = None #sys.argv[1] #

meta_params = {}
meta_params["data_dir"] = "ToyData_3ROIS"
meta_params["Iters"] = Iters
meta_params["steps_between_obs"] = 40
meta_params["N_particles"] =  1000 # 500 #particles per worker
meta_params["learning_rate"] = 0.01 #0.003 for A=1 chain #for A=50 0.003 ok # for A=1 0.1 ok for noise_factor=0.01
meta_params["anneal_threshold"] = 0.03 # fraction of particles
meta_params["anneal_factor"] = 1.15
meta_params["graph_generator"] = [] #['random'] #[] # ['complete'] # #'random' or 'complete'
ess_threshold = 0.05
params2update =  ['None'] #['sigma_dyn'] #
regularizer= ['L1'] #[] #['bi-dir'] #['test_reg'] #
trainer =  ['momentum'] #[] #['adam'] #['test_update'] #['newton']# 

#Instantiate objects
smproblem = SP(comm,meta_params,case_identifier)
apis = Smoother(smproblem)

if comm.Get_rank()==0: 
    print "Number of iterations:", Iters
    print "Learning rate: ", meta_params["learning_rate"]
    N_t = apis.timepoints
    nr_rois = smproblem.nr_rois
    #array_pmeanZ = np.zeros([Iters,nr_rois,N_t])
    #array_pmeanBold = np.zeros([Iters,nr_rois,N_t])
    #array_meanZ = np.zeros([Iters,nr_rois,N_t])
    #array_meanBold = np.zeros([Iters,nr_rois,N_t])
    #array_OLC = np.zeros([Iters,smproblem.dim_control,N_t])
    #array_feedbackMatrix = np.zeros([Iters,smproblem.dim_control,smproblem.dim_control,N_t])
    nr_elements = nr_rois*(nr_rois-1)
    Adj_offdiag_Itrs = np.zeros([Iters,nr_elements])
    Adj_Eigenvals_Itrs = np.zeros([Iters,nr_rois])
    GradsAdj_Itrs = np.zeros([Iters,nr_elements])
    CostAdj_Itrs = np.zeros([Iters,1])
    MInfo_Itrs = np.zeros([Iters,1])
    
    start_time = time()
for itr in np.arange(Iters):
    if comm.Get_rank()==0: print "Iteration ",itr
    apis.generate_particles()
    apis.get_statistics(itr)
    apis.adapt_initialization()
    apis.update_controller()
    ####################################
    mask = ~np.eye(smproblem.feedback_term.shape[0],dtype=bool)
    smproblem.feedback_term[mask] = 0.  # A diagonal parametrization of the controller is important to avoid ESS decay
    ####################################
    
    apis.posterior_obssignal()
    
    local_meanZ = np.mean(apis.Particles[:,0,:],axis=0)
    global_meanZ = np.zeros(apis.Particles.shape[-1])
    comm.Reduce(local_meanZ,global_meanZ)  
    
    local_mean_obSignal = np.mean(apis.local_obsSignal,axis=0)
    global_mean_obsSignal = np.zeros(apis.local_obsSignal.shape[1:])
    comm.Reduce(local_mean_obSignal,global_mean_obsSignal)
    
    apis.update_parameters(itr,ess_threshold,*params2update)
    if comm.Get_rank()==0:
        mask = ~np.eye(smproblem.Adj.shape[0],dtype=bool)
        Adj_offdiag_Itrs[itr,:] = smproblem.Adj[mask]
    
    apis.update_connectivity(itr,ess_threshold,regularizer,trainer)
    
    if comm.Get_rank()==0:
        #print global_meanZ.shape, global_mean_obsSignal.shape
        #array_pmeanZ[itr] = global_meanZ
        #array_pmeanBold[itr] = global_mean_obsSignal
        
        #array_meanZ[itr] = apis.mean_post[0]
        #array_meanBold[itr] = apis.mean_postObsSignal
        
        #array_OLC[itr] = smproblem.openloop_term
        #array_feedbackMatrix[itr] = smproblem.feedback_term
        GradsAdj_Itrs[itr,:] = apis.last_gradient_step[mask] 
        Adj_Eigenvals_Itrs[itr,:] = np.real(np.linalg.eigvals(smproblem.Adj))
        CostAdj_Itrs[itr] = apis.log_pz
        MInfo_Itrs[itr] = np.log(apis.norm_psi/apis.total_number_particles) + apis.wCost_itrs[itr]
        print "Mutual information for single TS:", MInfo_Itrs[itr]
     
    
if comm.Get_rank()==0:
    elapsed = (time() - start_time)
    print 'Elapsed time for ',Iters,' iterations: ', elapsed, 'sec' 
    print 'Per itaration is on average:',elapsed/Iters
    
apis.save_data() #Only saves if the flag -save is given in arg.sys
if comm.Get_rank()==0 and "-save" in sys.argv:
    #apis.save_var("array_pmeanZ",array_pmeanZ)
    #apis.save_var("array_pmeanBold",array_pmeanBold)
    
    #apis.save_var("array_meanZ",array_meanZ)
    #apis.save_var("array_meanBold",array_meanBold)
    
    #apis.save_var("array_OLC",array_OLC)
    #apis.save_var("array_feedbackMatrix",array_feedbackMatrix)
    
    apis.save_var("Adj_offdiag_Itrs",Adj_offdiag_Itrs)
    apis.save_var("GradsAdj_Itrs",GradsAdj_Itrs)
    apis.save_var("Adj_Eigenvals_Itrs",Adj_Eigenvals_Itrs)
    apis.save_var("CostAdj_Itrs",CostAdj_Itrs)
    apis.save_var("MInfo_Itrs",MInfo_Itrs)
