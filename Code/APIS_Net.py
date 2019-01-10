import sys
import os
#sys.path.append("../../")
import numpy as np
from mpi4py import MPI
import scipy.io as sio
from APIS import APIS
"""
Created on 14 June 2017

The class APIS generates trajectories/particles integrating the stochastic system defined in the Smoothing_Problem class, it estimates its costs and unnorm_weights and using these, it estimates the control parameters defined in Smoothing_Problem. (In the future: It also trains the model of the system using the EM-algorithm)

This class:
    -Initializes all data variables, e.g. particles, costs (S=V+C_u,V=neg_logLikelihood,C_u), weights, ESS, ESS_raw, mean_post, var_post, norm_psi, the correlation matrices between the basis functions corr_basis=H and the correlations between noise realizations and basis_functions corr_basisnoise=dQ(h).
    -Estimates all above quantities from the particles.
    -Implements annealing
    -Implements a resampling funtion to get the smoothing particles according to the weights
    (-Implements the EM algorithm in update_parameters)

An istance of the class is initialized in the main program and gets as argument an instance of Smoothing_Problem class. particles are generated, weights and all necessary estimations for the updates are computed. A resampling function to get the smoothing particles is also defined.
NOTE: The functions with the prefix nn in their name, use only the weights exp(-S) that are NOT normalized!
@author: HCRuiz
"""

class APIS_Net(APIS):
    
    def __init__(self,smproblem):
        super(APIS_Net,self).__init__(smproblem)

    def _get_nncorrelations(self):
        if self.rank==0:
            print "PICE update rule for time-indep. feedback controller"
            self.hh_t = np.zeros([self.dim_control,self.dim_control])
            self.hdW_t = np.zeros([self.dim_control,self.dim_control])
            self.dW_t = np.zeros([self.dim_control,self.timepoints])
        
        for d in np.arange(self.dim_control): #loop over controlled dimensions (and dims of noise)
            dW_t_d = self.nnweighted_sum(self.Noise[d,:,:])
            if self.rank==0: self.dW_t[d,:] = dW_t_d
            for k in np.arange(self.dim_control): #loop over basis functions
                hh_t_dk = self.nnweighted_sum(np.sum(self.Basis_xt[:,k,:]*self.Basis_xt[:,d,:],axis=1))
                hdW_t_dk = self.nnweighted_sum(np.sum(self.Basis_xt[:,k,:]*self.Noise[d,:,:],axis=1))
                if self.rank==0:
                    self.hh_t[d,k] = hh_t_dk
                    self.hdW_t[d,k] = hdW_t_dk
                    
    def _update_control_params(self,*args):
        self._get_nncorrelations() # Only root=0 has the correlations!!
        if self.rank==0: 
            for t in np.arange(self.smp.openloop_term.shape[-1]):
                if t==0: print "Standard Update..."
                self.smp.openloop_term[:,t] += self.learning_rate*self.dW_t[:,t]/(self.dt*self.annealed_psi)
                #Although mathematically doesn't matter the normalization constant \psi, it is important for stability. Without normalization, the results vary much more!
            #print "Shape of hdW_t:",self.hdW_t.shape, "Shape of hh_t:",self.hh_t.shape
            H_inv = np.linalg.pinv(self.hh_t/self.annealed_psi)
            update_step = self.learning_rate*self.hdW_t*H_inv/(self.dt*self.annealed_psi)   
            self.smp.feedback_term[:,:] += update_step[:,:,np.newaxis]
        self.smp.openloop_term = self.comm.bcast(self.smp.openloop_term,root=0)
        self.smp.feedback_term = self.comm.bcast(self.smp.feedback_term,root=0)
        
        
    def update_connectivity(self,itr,ess_threshold,reg,trainer):
        eta= 0.0075 #0.01-0.07 works for A=1 chain # 0.00003 works stable for A=50 
        if self.rank==0 and eta==0: print "WARNING: NO UPDATE!"
        momentum_rate = 0.9 # 0.8 works fine for A=1 and high noise; 0.9 works ok for A=50 (no instabilities,etc but also no convergence to GT)
        if self.smp.case_identifier is not None: #\lambda_{L1}=0.001,0.01,0.05,0.1
            lambda_L1 = np.array([0.,0.001,0.01,0.05,0.1])
            reg_factor = lambda_L1[int(self.smp.case_identifier)] #0.05 + 0.005*float(self.smp.case_identifier)
        else:
            reg_factor = 0.01 #0.05 workes well for close_net; 0.01 is proven better for chain_net (all for A=1)
        
        roi_states = self.Particles[:,0:self.dim_control,:]
        self.Grad_Adj = np.zeros((self.dim_control,self.dim_control))
        self.Grad_Adj[:,:] = None
        if itr==0: self.log_pz = np.zeros(1)
        #print self.Noise.shape, self.Particles.shape, roi_states.shape
        if self.rank==0  and itr==0: 
            self.last_gradient_step = np.zeros_like(self.Grad_Adj)
            self.exp_decay_avg_grads = 0.
            self.secnd_moment = 0.
            self.count_itrs = 1.
        #if self.rank==0  and itr==0 and (ess_threshold>self.anneal_threshold or ('-testing' in sys.argv)):
            print "ESS threshold is ",ess_threshold
            print "Updating Connectivity if raw ESS >",ess_threshold
            print "Learning rate eta for connectivity is set at: ",eta 
            print "regularization strength:", reg_factor

        if ess_threshold<self.anneal_threshold or ('-testing' in sys.argv): 
            if '-testing' in sys.argv and self.rank==0:
                print "WARNING: Testing APIS; learning threshold might be smaller than annealing threshold"
            else:
                if itr==0 and self.rank==0:print "WARNING: Learning threshold is smaller than annealing threshold"
 
        self.ESS_raw = self.comm.bcast(self.ESS_raw,root=0)
        
        if self.ESS_raw>ess_threshold or ('-testing' in sys.argv):
            proxy_log_pz = 0.
            sum_t_uplusnoise = np.zeros((self.N_particles,self.dim_control,self.dim_control))
            sum_t_states = np.zeros((self.N_particles,self.dim_control,self.dim_control))
            hessian = np.zeros((self.dim_control,self.dim_control))
            mask = ~np.eye(self.smp.Adj.shape[0],dtype=bool) #np.ones(self.smp.Adj.shape,dtype=bool) #
            
            for d1 in np.arange(self.dim_control):
                upn = self.U_xt[:,d1,:]*self.dt + self.Noise[d1,:,:]
                upn2 = np.sum(upn**2,axis=1)/(self.sigma_dyn[d1,d1]**2)
                summand_log_pz = self.nnweighted_sum(upn2) #/(self.dt*self.sigma_dyn[d1,d1]**2)
                
                for d2 in np.arange(self.dim_control):
                    sum_t_uplusnoise[:,d1,d2] = np.sum(upn*roi_states[:,d2,:],axis=1)/self.sigma_dyn[d1,d1]  # DO NOT take sigma_dyn out!! (even if for A=50 works better) 
                    proxy_grad = self.nnweighted_sum(sum_t_uplusnoise[:,d1,d2])
                    sum_t_states[:,d1,d2] = np.sum(roi_states[:,d1,:]*roi_states[:,d2,:]*self.dt,axis=1)
                    hessian_proxy = self.nnweighted_sum(sum_t_states[:,d1,d2])
                    if self.rank==0: 
                        summand_log_pz /= self.annealed_psi
                        proxy_grad /= self.annealed_psi # DO NOT take out!! 
                        hessian[d1,d2] = hessian_proxy/self.annealed_psi
                        
                        if 'test_reg' in reg:
                            if d1==0 and d2==0: print "Testing regularization"
                            l1_reg = 0.
                            #out-degree sparsity regularization
                            out_Edges = np.sum(self.smp.Adj[:,d2]**2) - self.smp.Adj[d2,d2]**2
                            norm_outEdges = np.sqrt(out_Edges)                     
                            sq_outEdges = norm_outEdges**2 - self.smp.Adj[d1,d2]**2
                            sum_abs_outEdges = np.sum(np.absolute(self.smp.Adj[:,d2])) 
                            sum_abs_outEdges -= np.absolute(self.smp.Adj[d2,d2]) + np.absolute(self.smp.Adj[d1,d2])
                            gradReg = (sq_outEdges-np.absolute(self.smp.Adj[d1,d2])*sum_abs_outEdges)/norm_outEdges**3. 
                            
                            gradReg += np.absolute(self.smp.Adj[d2,d1]) + l1_reg # kills bi-directionality AND keeps the weights small
                            
                            self.Grad_Adj[d1,d2] = proxy_grad - reg_factor*np.sign(self.smp.Adj[d1,d2])*gradReg
                            
                        elif 'discounted_path' in reg:
                            if d1==0 and d2==0: print "Discounted path regularization"
                            discount_factor = 0.1
                            if d1>d2:
                                print "Discount factor in dims ", d1,d2, " for ",self.smp.Adj[d1,d2]," is ",discount_factor**d2
                                self.Grad_Adj[d1,d2] = proxy_grad-(discount_factor**d2)*reg_factor*np.sign(self.smp.Adj[d1,d2])
                            else:
                                self.Grad_Adj[d1,d2] = proxy_grad-reg_factor*np.sign(self.smp.Adj[d1,d2])
                        elif 'L1' in reg:
                            if d1==0 and d2==0: print "L1-regularization with regularization strength:", reg_factor
                            self.Grad_Adj[d1,d2] = proxy_grad - reg_factor*np.sign(self.smp.Adj[d1,d2])
                        elif 'bi-dir' in reg: # to kill bidirectionality
                            if d1==0 and d2==0: print "bi-dir regularization with regularization strength:", reg_factor
                            norm = np.sqrt(self.smp.Adj[d1,d2]**2 + self.smp.Adj[d2,d1]**2)
                            factor = np.sign(self.smp.Adj[d1,d2])/norm**3.
                            grad_bidir = factor*(self.smp.Adj[d2,d1]**2-np.absolute(self.smp.Adj[d1,d2])*np.absolute(self.smp.Adj[d2,d1]))
                            self.Grad_Adj[d1,d2] = proxy_grad - reg_factor*grad_bidir
                        elif 'L2' in reg:
                            if d1==0 and d2==0: print "L2-regularization with regularization strength:", reg_factor
                            self.Grad_Adj[d1,d2] = proxy_grad - reg_factor*self.smp.Adj[d1,d2]
                        else:
                            if d1==0 and d2==0: print "No regularization!" 
                            #finds a fully connected network with bidirectionality and negative weights
                            self.Grad_Adj[d1,d2] = proxy_grad 
                            
                        proxy_log_pz += summand_log_pz
            self.log_pz = proxy_log_pz

            if self.rank==0:
                eigvals = np.real(np.linalg.eigvals(self.smp.Adj))
                #mask = ~np.eye(self.smp.Adj.shape[0],dtype=bool)
                #print mask
                
                if 'momentum' in trainer:
                    self.smp.Adj[mask] += eta*self.Grad_Adj[mask] + momentum_rate*self.last_gradient_step[mask]
                    self.last_gradient_step = eta*self.Grad_Adj + momentum_rate*self.last_gradient_step
                    if self.rank==0: print "MOMENTUM trainer"
                elif 'adam' in trainer:
                    beta1 = 0.9
                    beta2 = 0.99
                    eps = 10**(-8)
                    eta_adam = 0.007 #0.001 seems to work fine, but I have to let it run longer (than 250 iters)
                    self.exp_decay_avg_grads = beta1*self.exp_decay_avg_grads + (1.-beta1)*self.Grad_Adj
                    self.secnd_moment = beta2*self.secnd_moment + (1.-beta2)*self.Grad_Adj**2.
                    
                    m_hat = self.exp_decay_avg_grads/(1.-beta1**self.count_itrs)
                    v_hat = self.secnd_moment/(1.-beta2**self.count_itrs)
                    if self.count_itrs==1:
                        self.last_gradient_step = eta*m_hat
                    else:
                        self.last_gradient_step = eta_adam*m_hat/(np.sqrt(v_hat)+eps)
                    
                    self.smp.Adj[mask] += self.last_gradient_step[mask]
                    
                    if self.rank==0: 
                        print "ADAM trainer"
                        #print m_hat, np.sqrt(v_hat)
                    self.count_itrs += 1.
                elif 'newton' in trainer:
                    eta = 0.05
                    alpha = 0.07 
                    # alpha=1. gives directions similar to the gradient where the directions in the first column dominate; 
                    #for <=10^-2 the direction of connection 2->3 and 3->2 dominate, but the connection 1->2 is learned too slowly!!
                    #alpha=0.1 the driection of connection 2->3 and 3->2 dominate but are less strong than 1->2 and 1->3... so some value between 0.01 and 0.1 should do the job
                    inv_hessian = np.linalg.pinv(hessian+alpha*np.eye(hessian.shape[0]))
                    newton_step = np.dot(self.Grad_Adj,inv_hessian)
                    self.smp.Adj[mask] += eta*newton_step[mask] + momentum_rate*self.last_gradient_step[mask]
                    self.last_gradient_step = eta*newton_step + momentum_rate*self.last_gradient_step
                    if self.rank==0: 
                        print "newton trainer with alpha = ",alpha, " eta = ",eta
                        print "Inverse Hessian = \n",inv_hessian                   
                elif 'test_update' in trainer:
                    #eta= 0.0001
                    C = self.log_pz
                    self.smp.Adj[mask] += eta*self.Grad_Adj[mask]/C + momentum_rate*self.last_gradient_step[mask]
                    self.last_gradient_step = eta*self.Grad_Adj/C + momentum_rate*self.last_gradient_step
                    if self.rank==0: 
                        print "TEST trainer log(C)"
                        print "C=",C
                        
                else:
                    self.smp.Adj[mask] += eta*self.Grad_Adj[mask]
                    self.last_gradient_step = eta*self.Grad_Adj    
                    if self.rank==0: print "STANDARD trainer"
                
                
                
                if any(eigvals>0.):
                    print "Some Eigvals of Adj are positive! Skip update"
                    self.smp.Adj[mask] -= self.last_gradient_step[mask]

                print "Update Step Matrix: \n", self.last_gradient_step
                print "Adjacency MAtrix: \n", self.smp.Adj
                print "Eigen values of Adj: ",eigvals
                print "Costs Adj: ", np.log(self.log_pz)

            self.smp.Adj = self.comm.bcast(self.smp.Adj,root=0)