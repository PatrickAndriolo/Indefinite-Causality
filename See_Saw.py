# Code developed in collaboration with Davide Poderine

import numpy as np
import warnings
import cvxpy as cp
#import picos as pc
#from qutip import *
#from mosek import *

# dimension of the instruments:
d = 4  

def Composition(W, d, i):

    subsys_dim = int(np.sqrt(d)) 

    Z = np.reshape(np.copy(W),[2,2,2,2,2,2,2,2])

    # trace Ai
    if(i == 0):
        
        Z_traced = np.einsum('akmpalnq -> kmplnq', np.copy(Z)) 
        
        for u in range(subsys_dim):
            for v in range(subsys_dim):
                if u == v:
                    Z[u,:,:,:,v,:,:,:] = 0.5 * Z_traced
                else:
                    Z[u,:,:,:,v,:,:,:] = np.zeros((2,2))

    # trace Ao
    elif(i == 1):
        
        Z_traced = np.einsum('akmpbknq -> ampbnq', np.copy(Z)) 

        for u in range(subsys_dim):
            for v in range(subsys_dim):
                if u == v:
                    Z[:,u,:,:,:,v,:,:] = 0.5 * Z_traced
                else:
                    Z[:,u,:,:,:,v,:,:] = np.zeros((2,2))

    # trace Bi
    elif(i == 2):
        
        Z_traced = np.einsum('akmpblmq -> akpblq', np.copy(Z)) 

        for u in range(subsys_dim):
            for v in range(subsys_dim):
                if u == v:
                    Z[:,:,u,:,:,:,v,:] = 0.5 * Z_traced
                else:
                    Z[:,:,u,:,:,:,v,:] = np.zeros((2,2))

    # trace Bo
    elif(i == 3):
        
        Z_traced = np.einsum('akmpblnp -> akmbln', np.copy(Z)) 
        
        for u in range(subsys_dim):
            for v in range(subsys_dim):
                if u == v:
                    Z[:,:,:,u,:,:,:,v] = 0.5 * Z_traced
                else:
                    Z[:,:,:,u,:,:,:,v] = np.zeros((2,2))

    else:
        print('No subsystem corresponding to index ' + str(i))

    Z = np.reshape(Z, [16,16])
    
    return(Z)


# Currently the function below is not being used:

def Partial_Trace(A, d, i):

    subsys_dim = int(np.sqrt(d)) 

    A = np.reshape(A, [subsys_dim,subsys_dim,subsys_dim,subsys_dim])

    # trace input space
    if(i == 0):
        A_traced = np.einsum('ikjl -> kl', A) 

    # trace output space
    if(i == 1):
        A_traced = np.einsum('ikjl -> ij', A) 

    A_traced = np.reshape(A_traced,[subsys_dim,subsys_dim])
    
    return(A_traced)




def Random_Instruments(d):
    
    T = np.random.randn(d, d)
    T = T @ T.T  # Make sure T is positive semidefinite

    # Step 2: Create variable matrices M1 and M2
    M1 = cp.Variable((d, d), hermitian=True)
    M2 = cp.Variable((d, d), hermitian=True)

    # Step 3: Define the constraints
    constraints = [M1 >> 0, M2 >> 0]  # M1 and M2 are positive semidefinite
    
    # Partial trace constraint (assuming 4x4 matrices and partial trace over the first qubit)
    constraints += [cp.partial_trace(M1 + M2, [2, 2], 1) == np.eye(2)]

    # Step 4: Define the objective function
    objective = cp.Minimize(cp.norm(M1 + M2 - T, 2))

    # Step 5: Solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve()

    return M1.value, M2.value


def Initial_Scenario(d):
    
    # x = 0 and x = 1 lead to four values of "a", each has one associated instrument;
    instrument = [Random_Instruments(d), Random_Instruments(d)]
    # entries having respective indexes: a = 0 x = 0, a = 1 x = 0; a = 0 x = 1, a = 1 x = 1.
    instrument = [instrument[0][0], instrument[0][1], instrument[1][0], instrument[1][1]] 
    
    #print(A)
    
    return(instrument)


def Optimize_ProcessMatrix_GYNI(d, A, B, verbose): 

    subsys_dim = int(np.sqrt(d))

    Wfixed = cp.Variable([d**2, d**2], hermitian=True)
    
    W = np.full((d**2, d**2), None, dtype='object')
    
    for i in range(d**2):
        for j in range(d**2):
            W[i, j] = Wfixed[i, j]
    
    mom00 = np.kron(A[0], B[0]) #.reshape(d**2, d**2)
    mom10 = np.kron(A[2], B[1]) #.reshape(d**2, d**2)
    mom01 = np.kron(A[1], B[2]) #.reshape(d**2, d**2)
    mom11 = np.kron(A[3], B[3]) #.reshape(d**2, d**2)
    
    p0000 = np.trace(mom00 @ W)
    p0110 = np.trace(mom10 @ W)
    p1001 = np.trace(mom01 @ W)
    p1111 = np.trace(mom11 @ W)
    
    pGYNI = 0.25 * (p0000 + p0110 + p1001 + p1111)

    constraints = [cp.real(pGYNI) >= 0.5] 
    
    # positive semidefinite condition
    constraints += [Wfixed >> 0] 

    # trace condition Tr W = d_(A_o)*d_(B_o)
    constraints += [np.trace(W) == subsys_dim**2]  

    # model for constraints 10.(c - e)
    #constraints += [eW == eI for eW, eI in zip(MATRIZ1.flatten(), MATRIZ 2.flatten())]

    # Eq. (10.c) -- ok!
    constraints += [eW == eI for eW, eI in zip(Composition(Composition(W, d, 3), d, 2).flatten(), Composition(Composition(Composition(W, d, 3), d, 2), d, 1).flatten())]
    
    # Eq. (10.d) -- ok! 
    constraints += [eW == eI for eW, eI in zip(Composition(Composition(W, d, 1), d, 0).flatten(), Composition(Composition(Composition(W, d, 3), d, 1), d, 0).flatten())]

    # Eq. (10.e) -- ok!
    #constraints += [a == b + c - d for a, b, c, d in zip( W.flatten(), Composition(W, d, 3).flatten(), Composition(W, d, 1).flatten(), Composition(Composition(W, d, 3), d, 1).flatten())]  
    constraints += [eW == eI for eW, eI in zip( W.flatten(), ( Composition(W, d, 3) + Composition(W, d, 1) - Composition(Composition(W, d, 3), d, 1) ).flatten())]  
    
    prob = cp.Problem(cp.Maximize(cp.real(pGYNI)), constraints)
    prob.solve(verbose = verbose)

    print('Process matrix optimization - GYINI maximal value = ' + str(np.real(pGYNI.value)))

    #return(W)
    return(Wfixed.value)


def Optimize_Instrument_GYNI(d, W, variable_instrument, fixed_instrument, verbose):
    
    if variable_instrument == 'A':
        A = [cp.Variable((d, d), hermitian = True) for _ in range(4)]
        B = fixed_instrument
        
    else:
        A = fixed_instrument
        B = [cp.Variable((d, d), hermitian = True) for _ in range(4)]
    
    if variable_instrument == 'A':

        # every matrix for A should be positive semidefinite
        constraints = [A[i] >> 0 for i in range(len(A))]  

        # input subspace should be complete

        # using cp partial trace function:
        constraints += [cp.partial_trace(A[0] + A[1], [2,2], 1) == np.eye(2)]
        constraints += [cp.partial_trace(A[2] + A[3], [2,2], 1) == np.eye(2)]
        
    else:
        
        # every matrix for B should be positive semidefinite
        constraints = [B[i] >> 0 for i in range(len(B))]  
        
        # input subspace should be complete

        # using cp partial trace function:
        constraints += [cp.partial_trace(B[0] + B[1], [2,2], 1) == np.eye(2)]
        constraints += [cp.partial_trace(B[2] + B[3], [2,2], 1) == np.eye(2)]

    # Kronecker products resulting in (d^2, d^2) matrices
    mom0000 = cp.kron(A[0], B[0])
    mom0110 = cp.kron(A[2], B[1])
    mom1001 = cp.kron(A[1], B[2])
    mom1111 = cp.kron(A[3], B[3])
    
    # terms of the GYNI inequality
    p0000 = cp.trace(mom0000 @ W)
    p0110 = cp.trace(mom0110 @ W)
    p1001 = cp.trace(mom1001 @ W)
    p1111 = cp.trace(mom1111 @ W)
    
    pGYNI = 0.25 * (p0000 + p0110 + p1001 + p1111)

    constraints += [cp.real(pGYNI) >= 0.56] 
    
    prob = cp.Problem(cp.Maximize(cp.real(pGYNI)), constraints)
    prob.solve(verbose = verbose)
    
    if variable_instrument == 'A':
        print('Instrument A optimization - GYINI maximal value = ' + str(np.real(pGYNI.value)))
        return [a.value for a in A]
    else:
        print('Instrument B optimization - GYINI maximal value = ' + str(np.real(pGYNI.value)))
        return [b.value for b in B]

def SeeSaw(d, n, verbose):
    
    i = 0

    #def kron(*As):
        #res = As[0]
        #for A in As[1:]:
            #res = np.kron(res, A)
        #return res

    A = [None] * 4

    A[0] = np.array([[0., 0., 0., 0.],
                     [0., 0., 0., 0.],
                     [0., 0., 0., 0.],
                     [0., 0., 0., 0.]])

    A[1] = np.array([[1., 0., 0., 1.],
                     [0., 0., 0., 0.],
                     [0., 0., 0., 0.],
                     [1., 0., 0., 1.]])

    A[2] = np.array([[1., 0., 0., 0.],
                     [0., 0., 0., 0.],
                     [0., 0., 0., 0.],
                     [0., 0., 0., 0.]])

    A[3] = np.array([[0., 0., 0., 0.],
                     [0., 0., 0., 0.],
                     [0., 0., 1., 0.],
                     [0., 0., 0., 0.]])

    B = A

    # A and B begin with same random instruments
    #A = Initial_Scenario(d)
    #B = A
    
    while i < n:

        print('Printing during see-saw')
        print('\n\n\n')
        W = Optimize_ProcessMatrix_GYNI(d, A, B, verbose)
        print(np.real(np.where(np.abs(W) > 1e-4, W, 0)))
        print('\n\n\n')
        A = Optimize_Instrument_GYNI(d, W, 'A', B, verbose)
        print(np.real(np.where(np.abs(A) > 1e-4, A, 0)))
        print('\n\n\n')
        B = Optimize_Instrument_GYNI(d, W, 'B', A, verbose)
        print(np.real(np.where(np.abs(B) > 1e-4, B, 0)))
        print('\n\n\n')
        
        i += 1

    return A, B, W

warnings.filterwarnings("ignore", category=UserWarning)