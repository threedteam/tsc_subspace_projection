'''
Public sensors drift dataset
Code of 'Anti-Drift in Electronic Nose via Dimensionality Reduction: A Discriminative Subspace Projection Approach' Author: ZHENGKUN YI 
Code Author: Xi Chen
Created on : 2020/12/30
'''
import numpy as np
from Algorithm.core import within_class_scatter
from Algorithm.core import target_domain_scatter
from Algorithm.core import source_domain_scatter
from Algorithm.core import between_class_scatter

def DDRCA(Ms,Mt,batchS,batchT,Ys,alpha,lambdas,detla):

    dim = np.size(batchS, 1)
    Ns = np.size(batchS,0)
    Nt = np.size(batchT,0)
    M = np.zeros((dim, dim))
    Ss = source_domain_scatter(batchS)
    Sw = within_class_scatter(batchS, Ys)
    St = target_domain_scatter(batchT)
    Sb = between_class_scatter(batchS,Ys)
    Ms = Ms.reshape((1, dim))  # reshape
    Mt = Mt.reshape((1, dim))  # reshape
    MDD = np.dot((Ms - Mt).T, (Ms - Mt))  # mean distribution discrepancy minimization
    tem = np.linalg.pinv(MDD)
    tem1 = Ss/Ns + (alpha*St)/Nt - lambdas*Sw + detla*Sb
    M = M + np.dot(tem,tem1)
    eigenvalue, eigenvector = np.linalg.eig(M)
    eigenvalue = np.real(eigenvalue)  # Complex numbers appear during eigen decomposition, which makes it impossible to compare eigenvalues, so take the real part of the eigenvalue and the eigenvector
    Idex = np.argsort(-eigenvalue)
    eigenvector = np.real(eigenvector)# Same as above
    neweigenvector = eigenvector[:, Idex]
    return neweigenvector


