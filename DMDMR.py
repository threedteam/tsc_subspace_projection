
"""
Code of "Drift Compensation for an Electronic Nose by Adaptive Subspace Learning" Author: Dr.Liu
Code Author: Xi Chen
Created on  2020/12/02
"""
import numpy as np
from Algorithm.core import target_domain_scatter
from Algorithm.core import source_domain_scatter
from Algorithm.core import HSIC

def MyAlgorithm(Ms,Mt,batchS,batchT,Ys,alpha,lambdas,detla):
    '''
     calculate projection matrix P
    :param Ms: Source domain initial center
    :param Mt: Target domain initial center
    :param batchS: Source Data
    :param batchT: Target Data
    :param Ys: Source label
    :param alpha: Trade-off parameter
    :param lambdas: Trade-off parameter
    :param detla: Trade-off parameter
    :return: projection matrix P
    '''
    dim = np.size(batchS,1)                #  dimensional
    M = np.zeros((dim,dim))
    Ss = source_domain_scatter(batchS)     # source domain scatter matrix
    St = target_domain_scatter(batchT)     # target domain scatter matrix
    Ms = Ms.reshape((1,dim))
    Mt = Mt.reshape((1,dim))
    MDD = np.dot((Ms-Mt).T,(Ms-Mt))        # mean distribution discrepancy minimization
    hsic = HSIC(batchS,Ys)                 # Hilbert-Schmidt independence criterion
    M = M + Ss + alpha*St - lambdas*MDD + detla*hsic # M matrix
    eigenvalue, eigenvector = np.linalg.eig(M)
    index = np.argsort(-eigenvalue)        # max index of eigenvalue
    neweigenvector = eigenvector[:,index]  # reoder
    return neweigenvector