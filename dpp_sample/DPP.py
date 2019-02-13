"""
Copyright (c) 2016 Ling Chun Kai


Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

"""
Code snippet provides rudimentary (brute force) sampling of a L-ensemble [1]. Also provides approximate MAP inference up to a 1/4 approximation [2]. Requires the usual numpy, scipy, matplotlib stack.

[1] Kulesza, Alex, and Ben Taskar. "Determinantal point processes for machine learning." arXiv preprint arXiv:1207.6083 (2012).
[2] Gillenwater, Jennifer, Alex Kulesza, and Ben Taskar. "Near-optimal MAP inference for determinantal point processes." Advances in Neural Information Processing Systems. 2012.
"""

import math
import numpy as np
import scipy.linalg
import scipy.optimize
import matplotlib as mpl
import matplotlib.pyplot as plt

class DPP:
    
    def __init__(self, L):
        self.L = L
        self.e_val = None
        self.e_vec = None

    def compute_eigen_decomposition(self):
        self.e_val, self.e_vec=scipy.linalg.eigh(self.L)
        idx=self.e_val.argsort()[::-1]
        self.e_val=self.e_val[idx]
        self.e_vec=self.e_vec[:,idx]

    def sample(self):
        # Sorted eigendecomposition
        if self.e_val == None or self.e_vec == None:
            print "eigendecomposition not cached, computing..."
            self.compute_eigen_decomposition()

        e_val = self.e_val
        e_vec = self.e_vec
            
        ## Sample from binomial distributions 
        peV=[eV/(eV+1) if eV>0 else 0 for eV in e_val]
        drawneVals=np.random.binomial(1,p=peV)

        eBasis = e_vec[:, [x for x in xrange(len(drawneVals)) if drawneVals[x]==1]]

        print "E[|V|]=", sum(peV)
        print "|V|=", eBasis.shape[1]
    
        ## Iteratively draw vectors
        chosen = []
        while eBasis.shape[1] > 0:
            probs = np.sum(np.square(eBasis.T), 0)/eBasis.shape[1]
            
            # elementary vector chosen
            elem_chosen=np.random.multinomial(1, probs, size=1)
            chosen.append(np.squeeze(elem_chosen))
            
            if eBasis.shape[1] == 1: break
            
            # Projection of chosen element onto existing subspace
            proj = np.sum(np.diag(np.squeeze(eBasis.T.dot(elem_chosen.T))).dot(eBasis.T), 0)
            proj = np.expand_dims(proj/np.linalg.norm(proj), 1)  
            
            # Find orthogonal basis of subspace - projection
            residual = np.diag(np.squeeze(eBasis.T.dot(proj))).dot(np.tile(proj.T, (eBasis.shape[1],1)))
            eBasis = scipy.linalg.orth(eBasis-residual.T)
        
        return sum(chosen) 

    def expected_cardinality(self):
        if self.e_val == None or self.e_vec == None:
            self.compute_eigen_decomposition()

        return sum([eV/(eV+1) for eV in self.e_val])

    def get_map(self):
        """
        Find MAP in O(N^3), up to convergence rate
        Polytope assumed to be [0,1]^N
        """        
        x = self.local_opt(np.ones([self.L.shape[0],1]))
        y = self.local_opt(optConstraints=np.reshape(np.rint(np.ones(x.shape)-x),-1)[...,np.newaxis])
        if self.soft_max(x) > self.soft_max(y):
            return x
        else:
            return y

    def local_opt(self, optConstraints, epsilon = 10**-5, max_it = 150):
        """
        @param optConstraints: additonal optional inequality constraint (upper bound on x). Set to ones if we want restriction to hypercube
        """
        x = 0*np.ones([self.L.shape[0], 1])
        x = np.squeeze(x)
        for n in xrange(max_it):
            y = scipy.optimize.linprog(-np.squeeze(self.grad_soft_max(x)), np.concatenate((-np.eye(self.L.shape[0]), np.eye(self.L.shape[0])), axis=0), np.concatenate((np.zeros([self.L.shape[0], 1]), optConstraints )))
            y = y.x
            alpha, val, d = scipy.optimize.fmin_l_bfgs_b(lambda q: -self.soft_max(q * x + (1.-q) * y), 0.5, approx_grad=True, fprime = lambda r: np.array(self.grad_soft_max(r*x+(1-r)*y).T.dot(x-y)), bounds = [(0.,1.)]) # TODO: Use proper gradient for optimization with l_bfgs
            x = alpha * x + (1-alpha) * y
            if np.all(np.abs(x-np.rint(x)) < epsilon): # Convergence
                break
        return x

    def soft_max(self, x):
        """
        Compute F-tilde
        """
        return math.log(scipy.linalg.det(np.diag(np.squeeze(x)).dot(self.L-np.eye(x.size)) + np.eye(x.size)))

    def grad_soft_max(self, x):
        inv = scipy.linalg.inv(np.diag(np.squeeze(x)).dot(self.L-np.eye(x.size)) + np.eye(x.size))
        LmI = self.L - np.eye(self.L.shape[0])
        ret = np.zeros(x.shape)
        for k in xrange(x.size):
            ret[k] = LmI[k, :].dot(inv[:, k])
        return ret
 

if __name__ == "__main__":
    SIZE_GRID = 15
    NUMELS = SIZE_GRID*SIZE_GRID
    SIGMA = 25
    # Kernel defined to be non-circular, SE (RBF) 
    # SE kernel
    L = lambda a, b, sigma: 10*math.exp(-0.5*(np.linalg.norm(a-b)/sigma)**2)
    # Explicitly construct L matrix by iterating pairwise
    Ygrid, Xgrid = np.mgrid[0:SIZE_GRID, 0:SIZE_GRID]
    points = np.concatenate((np.reshape(Ygrid, -1)[..., np.newaxis], np.reshape(Xgrid, -1)[..., np.newaxis]), 1)
    l_mat = np.zeros((points.shape[0], points.shape[0]))
    for i in xrange(points.shape[0]):
        for j in xrange(points.shape[0]):
            l_mat[i, j] = L(i, j, SIGMA)

    print l_mat
    dpp = DPP(l_mat)

    dpp_sample_plot = plt.figure('Sampled from DPP')
    for k in xrange(9):
        scatterplot = dpp.sample()


