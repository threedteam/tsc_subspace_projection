"""
Code of "Drift Compensation for an Electronic Nose by Adaptive Subspace Learning" Author: Dr.Zhang
Code Author: Xi Chen
Created on  2020/11/30
"""
import numpy as np
from Algorithm.core import source_domain_scatter
from Algorithm.core import target_domain_scatter

def DRCA(Ms,Mt,batchS,batchT,Lambda):
    """
    calculate projection matrix P
    :param Ms: Source domain initial center
    :param Mt: Target domain initial center
    :param batchS: Source Data
    :param batchT: Target Data
    :param Lambda:Trade-off parameter
    :return: projection matrix P
    """
    dim = np.size(batchS,1)
    A = np.zeros((dim,dim))
    Ms = Ms.reshape((1, dim))  # reshape
    Mt = Mt.reshape((1, dim))  # reshape
    tem = np.linalg.pinv(np.dot((Ms-Mt).T,(Ms-Mt)))
    Ss = source_domain_scatter(batchS)  # source domain scatter matrix
    St = target_domain_scatter(batchT)  # target domain scatter matrix
    tem1 = Ss + Lambda*St
    A = A + np.dot(tem,tem1)
    eigenvalue,eigenvector = np.linalg.eig(A)
    eigenvalue = np.real(eigenvalue) # Complex numbers appear during eigen decomposition, which makes it impossible to compare eigenvalues, so take the real part of the eigenvalue and the eigenvector
    Idex = np.argsort(-eigenvalue)
    eigenvector = np.real(eigenvector) # Same as above
    neweigenvector = eigenvector[:,Idex]
    return neweigenvector













