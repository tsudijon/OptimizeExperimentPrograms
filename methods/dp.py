import numpy as np


#################################################################
############### Methods for Single Programs ###################
#################################################################


# this is an approximation: assume that the smallest unit of allocation is N_step.
def metaproduction_DP_small_I(N,I, N_step, production_function):
    N_range = np.arange(0,N,N_step)
    I_range = np.arange(1,I+1,1)


    opt_vals = np.zeros((len(I_range), len(N_range)))
    opt_vals[:,0] = 0 # base case: N = 0
    opt_vals[0,:] = production_function(N_range) # base case: I = 1

    allocation = np.zeros((len(I_range), len(N_range)))
    allocation[:,0] = 1
    allocation[0,:] = N_range

    ## DP iteration
    for i in range(1,len(I_range)):
        for n in range(1,len(N_range)):
            vals = opt_vals[i-1,n-np.arange(n+1)] + opt_vals[0,np.arange(n+1)]
            opt_vals[i,n] = np.max(vals)
            allocation[i,n] = np.argmax(vals)*N_step


    opt_alloc = np.zeros(allocation.shape[0])
    running_N = N - N_step
    N_range = np.arange(0,N,N_step)

    for i in range(allocation.shape[0])[::-1]:
        
        opt_alloc[i] = allocation[i, np.searchsorted(N_range, running_N)]
        running_N = running_N - int(opt_alloc[i])

    return opt_vals, allocation, opt_alloc



# this is an approximation: assume that the smallest unit of allocation is N_step, and chunks of ideas are allocated at once.
def optimal_production_DP(N,I, N_step, I_step, production_function):
    N_range = np.arange(1,N, N_step)
    I_range = np.arange(1,I, I_step)


    helper_vals = np.zeros((I_step+1, len(N_range)))
    one_test_values = production_function(np.arange(0,N,N_step))

    ## DP base case: N = 0.
    helper_vals[:,0] = 0
    
    ## DP base case: I = 0.
    helper_vals[0,:] = 0

    ## Precompute up to I_step 
    for i in range(1,I_step+1):
        for n in range(1,len(N_range)):       
            helper_vals[i,n] = np.max(helper_vals[i-1,n-np.arange(n+1)] + one_test_values[np.arange(n+1)]) 


    opt_vals = np.zeros((len(I_range), len(N_range)))
    opt_vals[:,0] = 0 #base case, N = 0
    opt_vals[0,:] = production_function(N_range)

    ## DP iteration
    for i in range(1,len(I_range)):
        for n in range(1,len(N_range)):       
            opt_vals[i,n] = np.max(opt_vals[i-1,n-np.arange(n+1)] + helper_vals[-1,np.arange(n+1)]) 

    return opt_vals


#################################################################
############### Methods for Multiple Programs ###################
#################################################################


def solve_multiprogram_allocation_dp(programs, mpfs, N, N_step):

    """
    programs: list 
        List of program names

    mpfs: dict 
        dict of metaproduction functions, keyed by program name,

    """

    N_range = np.arange(1,N, N_step)
    opt_vals = np.zeros((len(programs), len(N_range)))
    opt_indexes = np.zeros((len(programs), len(N_range)))

    # base case: 1st program
    opt_vals[0,:] = mpfs[programs[0]]
    opt_indexes[0,:] = np.arange(len(N_range))


    # base case: 1 person
    for i in range(1,len(programs)):
        opt_vals[i,0] = max(opt_vals[i-1,0], mpfs[programs[i]][0] + mus_p[:i].sum())
        opt_indexes[i,0] = 1 
    

    for i in range(1,len(programs)):
        for n in range(1,len(N_range)):
            new_mpf = mpfs[programs[i]]
            v = opt_vals[i-1,n - np.arange(n+1)] + new_mpf[np.arange(n+1)]

            idx = np.argmax(v)
            opt_vals[i,n] = v[idx]
            opt_indexes[i,n] = idx 

    ### find optimal allocation
    optimal_allocation = np.zeros(len(programs))

    N_val = np.searchsorted(np.arange(1,N,N_step), N) - 1
    for i in range(len(programs))[::-1]:
        
        optimal_allocation[i] = opt_indexes[i,N_val]
        N_val = int(N_val - optimal_allocation[i])

    return opt_vals, opt_indexes, optimal_allocation


def solve_multiprogram_idea_dp(programs, mpfs, I, I_step):

    """
    programs: list 
        List of program names

    mpfs: dict 
        dict of metaproduction functions, keyed by program name,

    """

    I_range = np.arange(1,I+1, I_step)
    opt_vals = np.zeros((len(programs), len(I_range)))
    opt_indexes = np.zeros((len(programs), len(I_range)))

    # base case: 1st program
    opt_vals[0,:] = mpfs[programs[0]]
    opt_indexes[0,:] = np.arange(len(I_range))


    # base case: 1 idea
    for i in range(1,len(programs)):
        opt_vals[i,0] = max(opt_vals[i-1,0], mpfs[programs[i]][0])
        opt_indexes[i,0] = 1 
    


    for n in range(1,len(programs)):
        for i in range(1,len(I_range)):
            new_mpf = mpfs[programs[n]]
            v = opt_vals[n-1,i - np.arange(i+1)] + new_mpf[np.arange(i+1)]

            idx = np.argmax(v)
            opt_vals[n,i] = v[idx]
            opt_indexes[n,i] = idx 


    opt_alloc = np.zeros(opt_indexes.shape[0]).astype(int)
    running_I = I 

    for t in range(opt_indexes.shape[0])[::-1]:
        
        opt_alloc[t] = opt_indexes[t, running_I-1]
        running_I = running_I - int(opt_alloc[t])


    opt_alloc = opt_alloc.tolist()
    print(opt_alloc)

    return opt_vals, opt_indexes, opt_alloc

