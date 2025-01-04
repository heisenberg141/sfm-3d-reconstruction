import numpy as np
def EssentialFromFundamental(F, K):
    E_est = np.dot(K.T, np.dot(F,K))
    U, S, V = np.linalg.svd(E_est,full_matrices=True)
    
    S = np.diag(S)
    S[0,0], S[1,1], S[2,2] = 1, 1, 0
    E = np.dot(U,np.dot(S,V))

    return E
