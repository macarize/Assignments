import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import time
import math
import warnings


def mfa_em(X,K,M):
    itr_num = 30
    X = np.array(X)
    D, N = X.shape

    pi = np.array([1]*M)/M  # (M,) array

    mu = np.array(np.random.normal(loc=0, scale=0.1, size=M*D)).reshape([M,D,1])
    W = np.array(np.random.normal(loc=0, scale=0.1, size=M*D*K)).reshape([M,D,K])
    psi = np.array([0.1]*D).reshape((1,-1))

    for itr in range(itr_num):
        # ===== Expectation Stage (Calculating the posterior statistics, based on old parameters)
        # H is of size MxN, where hij is the prob of xj been in mixture j
        H = np.zeros((M,N))
        for j in range(M):
            covMat = np.matmul(W[j,...],W[j,...].T) + np.diag(psi[0])
            # print('covMat={}'.format(covMat))
            # multiGaussian = multivariate_normal(mu[j,...].reshape(-1), covMat)
            # tmp = multiGaussian.pdf(X.T)
            # Avoiding a bad mixture that has prob=0 for all samples
            badMixture = True
            while badMixture:
                tmp2 = calGassuianProb(X,mu[j,...].reshape(-1), covMat)  # (N,) array
                # print('tmp={}'.format(tmp))
                # print('tmp2={}'.format(tmp2))
                # print('pdf.shape={}'.format(tmp.shape))
                if np.sum(tmp2)<=1e-15:
                    new_mixture_mean = X[:,np.random.randint(N)]
                    print('BAD {}-th Gaussian component!\n Try to Re-initialize with new mean = {}'.format(j,new_mixture_mean))
                    mu[j,:,0] = new_mixture_mean
                else:
                    badMixture = False

            H[j,:] = tmp2
            # print('H[{},:].shape={}'.format(j,H[j,:].shape))
        # print('np.sum(H,axis=0)={}'.format(np.sum(H,axis=0)))
        # Avoiding bad sample that has prob=0 for all mixtures, if it happens, set equal prob.
        sumH = np.sum(H,axis=0)  # (N,)
        zero_idx = [ i for (i,v) in enumerate(sumH) if v<1e-15]
        H[:,zero_idx] += np.array([1.0]*D).reshape((-1,1))/D
        sumH[zero_idx] = 1
        H = H/sumH.reshape((1,-1))  # Should avoid divid by zero

        X_mu = []  # MxDxN matrix
        for j in range(M):
            X_mu.append(X - mu[j,...])
        X_mu = np.array(X_mu)

        # we use j to represent the index of which mixture (mixture_j)
        # E_(j,z|x)[j,z] = Gj*Wj'*inv(psi)*(x_n - muj) * Hj, where Gj = inv(I + Wj'*inv(psi)*Wj)
        # sumE_(j,z|x)[j,zz'] = N*Gj + sum(Ez[j,...]*Ez[j,...]')
        Ez = np.zeros((M,K,N))  # MxKxN
        sumEzz = np.zeros((M,K,K))  # MxKxK
        for j in range(M):
            # print('psi.shape={}'.format(psi.shape))
            # print('W[j,...].shape={}'.format(W[j,...].shape))
            # print('shape of np.eye(K)+matmul(W[j,...].T/psi,W[j,...]) = {}'.format( (np.eye(K)+matmul(W[j,...].T/psi,W[j,...])).shape) )
            Gj = inv( np.eye(K) + matmul(W[j,...].T/psi,W[j,...]) )  # KxK matrix
            Hj = H[j,:].reshape((1,N))  # (1,N) matrix
            Eztmp = matmul(matmul(Gj,W[j,...].T)/psi,X_mu[j,...])  # KxN
            Ez[j,...] = Eztmp * Hj  # KxN

            sumEzz[j,...] = np.sum(Hj)*Gj + matmul(Eztmp*Hj,Eztmp.T)  # KxK

        # ===== Maximization Stage
        # Update pi
        pi_new = np.sum(H,axis=1)/N  # (M,) array
        # print('pi_new={}'.format(pi_new))
        # print('H.shape={}'.format(H.shape))
        # print(H)
        # print(np.sum(H,axis=0))
        # Update mu, MxDx1 matrix
        mu_new = np.zeros_like(mu)
        for j in range(M):
            Hj = H[j,:].reshape((1,N))  # (1,N) matrix
            # X is of size DxN
            mu_new[j,...] = (np.sum(X*Hj,axis=1)/np.sum(Hj)).reshape((-1,1))
        # Update W, MxDxK matrix
        W_new = np.zeros_like(W)
        for j in range(M):
            # Hj = H[j,:].reshape((1,N))  # (1,N) matrix
            # X_mu[j,...] DxN matrix
            W_new[j,...] = matmul(matmul(X_mu[j,...],Ez[j,...].T),inv(sumEzz[j,...]))  # DxK matrix
        # Update psi, 1xD matrix
        psi_new = np.zeros_like(psi)  # 1xD matrix
        for j in range(M):
            Hj = H[j,:].reshape((1,N))  # (1,N) matrix
            # X_mu[j,...] DxN matrix
            Sj = matmul(X_mu[j,...]*Hj,X_mu[j,...].T)
            psi_new += np.diag( Sj - matmul(W_new[j,...], matmul(Ez[j,...],X_mu[j,...].T) ) ).reshape((1,-1))
        psi_new /= N

        # Update all parameters for next itr
        pi = pi_new  # (M,) array
        mu = mu_new  # MxDx1 matrix
        # print('W_new.shape={}'.format(W_new.shape))
        W = W_new  # MxDxK matrix
        psi = psi_new  # 1xD matrix

    return pi, mu.reshape(M,D), W, psi[0]

    # X is of dim (D,N), where N is number of data points
    # W is of dim (M,D,K)
    # mu is of dim (M,D)
    # psi is of dim (D,)
    # Return:
    # 	Z, (M,K,N)
    #	ZinXSpace, (M,D,N)
def mfa_inference(X,mu,W,psi):
    # Compute the latent variables Z for X
    X = np.array(X)
    assert(X.ndim==2)
    D, N = X.shape
    assert(W.ndim==3)
    M, WD, K = W.shape
    assert(WD==D)
    assert(mu.ndim==2)
    muM, muD = mu.shape
    assert(muM==M)
    assert(muD==D)
    assert(psi.ndim==1)
    psi = psi.reshape((1,D))
    assert(K<=D and K>0)

    Z = np.ones((M,K,N))
    ZinXSpace = np.ones((M,D,N))

    for midx in range(M):
        G = inv( np.eye(K) + matmul(W[midx,...].T/psi,W[midx,...]) )  # KxK matrix
        tmpM = mu[midx,...].reshape((D,1))
        X_mu = X - tmpM  # DxN matrix
        tmpZ = matmul(matmul(G,W[midx,...].T)/psi,X_mu)  # KxN
        tmpZinXSpace = matmul(W[midx,...],tmpZ)+tmpM
        Z[midx,...] = tmpZ
        ZinXSpace[midx,...] = tmpZinXSpace

    return Z, ZinXSpace

leukemia = np.loadtxt("leukemia_small.csv", delimiter=',')
print(leukemia)