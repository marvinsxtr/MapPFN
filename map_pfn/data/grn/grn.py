#!/usr/bin/env python
import numpy as np
import pandas as pd
import scipy.special
import networkx as nx
from map_pfn.data.grn.smallworld import grouped_scale_free_graph


_README="""
A script to generate synthetic data from gene regulatory networks with grouped small-world 
 network structure. 

Author: Matthew Aguirre (SUNET: magu)
"""


def main():
    # get graph generation parameters and whether we want to make plots
    import argparse
    parser = argparse.ArgumentParser(description=_README)
    parser.add_argument('--out',        type=str, required=True, metavar='my_grn', default='my_grn')
    parser.add_argument('--num-genes',  type=int, required=False, metavar=2000, default=2000)
    parser.add_argument('--num-groups', type=int, required=False, metavar=1, default=1)
    parser.add_argument('--r'    ,      type=float, required=False, metavar=4, default=4)
    parser.add_argument('--delta-in',   type=float, required=False, metavar=100, default=100)
    parser.add_argument('--delta-out',  type=float, required=False, metavar=1, default=1)
    parser.add_argument('--w',          type=float, required=False, metavar=10, default=10)
    parser.add_argument('--kos',        action='store_true')
    parser.add_argument('--cores',      type=int, required=False, metavar=1, default=1)
    args = parser.parse_args()
    
    # dump log
    pd.DataFrame(vars(args), index=['value']).T.to_csv(args.out+'.log', sep='\t')
     
    # generate gene regulatory network
    G = grn().add_structure(n = args.num_genes, 
                            k = args.num_groups, 
                            alpha = 1e-99, 
                            beta  = 1 - 1./args.r, 
                            gamma = 1./args.r, 
                            kappa = args.w,
                            delta_in  = args.delta_in, 
                            delta_out = args.delta_out
                           )
    # save these useful matrixes: all pairs path distances and module affinity map
    G.dist = pd.DataFrame(dict(nx.all_pairs_shortest_path_length(G))).values
    G.module = np.array([[G.groups[i] == G.groups[j] for j in G.nodes()] for i in G.nodes()])
     
    # make perturbations
    if args.kos:
        G.ko = G.ko_all_nodes(n_jobs = args.cores)

    # save to file
    nx.write_gpickle(G, args.out + '.gpickle')
    


class grn(nx.DiGraph):
    def __init__(self, adjacency_matrix=None, groups=None):
        super(grn, self).__init__()
        # load provided weights if given, otherwise this is an empty graph
        if adjacency_matrix is not None:
            nx.to_networkx_graph(adjacency_matrix != 0, create_using=self)
            self.n = self.number_of_nodes()
            self.groups = groups
            self.beta = adjacency_matrix
            self.add_expression_parameters()

    
    def add_structure(self, n, k, alpha, beta, gamma, delta_in, delta_out, kappa, expression_params=True):
        # wrapper for grouped_scale_free_graph
        G = grouped_scale_free_graph(n=n, k=k, alpha=alpha, beta=beta, gamma=gamma, 
                                     delta_in=delta_in, delta_out=delta_out, kappa=kappa)
        nx.to_networkx_graph(G, create_using=self)
        
        # store relevant info in self
        self.n = self.number_of_nodes()
        self.groups = nx.get_node_attributes(self, 'group').copy()
        
        # add parameters for RNA model
        if expression_params:
            self.add_expression_parameters(G)
        
        return self

    
    def add_expression_parameters(self, G=None, inflate_edges=True):
        # these are for the gene expression function
        self.link = scipy.special.expit
        self.observe_rna = self.observation_model
        
        # set edge weights in self.beta if we weren't given them to start
        if G is not None: 
            S = np.random.normal(0, 1, size=(self.n, self.n))
            E = nx.convert_matrix.to_numpy_array(self, multigraph_weight=sum)
            # remove self loops
            E -= np.diag(np.diag(E))
            if inflate_edges:
                S += np.sign(S) 
            self.beta = np.array(np.multiply(S, E))
        
        # generate gene attributes: production rate alpha, degradation rate l
        self.alpha = scipy.special.logit(np.random.beta(2, 8, self.n)).reshape(-1,1)
        self.l = np.maximum(self.link(-self.alpha), np.random.beta(8, 2, self.n).reshape(-1,1))
        
        return self
    

    def simulate_rna(self, x0=None, alpha=None, beta=None, l=None, link=None,
                     s=1e-4, dt=1e-2, tmax=20000, n=1, tol=1e-3, step=1000, burnin=5000, save=True):
        # n is n_samples and self.n is n_genes
        # simulate gene expression according to an SDE regulatory model:
        #     X(t+dt) = X(t) + dt * [link(alpha + beta.T X) - l*X + N(0, s^2 X / dt)]
        
        # setup
        if alpha is None:
            alpha = self.alpha
        if beta is None:
            beta = self.beta
        if l is None:
            l = self.l
        if link is None:
            link = self.link
        
        # traceline goes here (this could be made more efficient)
        X = np.zeros((n, tmax, self.n)) 
         
        # set initial condition
        if x0 is None:
            X[:,0,:] = np.zeros((n,self.n))  #np.random.random((n,self.n))
        elif x0.shape == (n, self.n):
            X[:,0,:] = x0
        elif x0.shape == (self.n,n):
            X[:,0,:] = x0.T
        elif x0.shape == (self.n,):
            X[:,0,:] = np.vstack([x0 for _ in range(n)])
       
        # run simulations: x.shape=(tmax,self.n)
        converged = False
        for i in range(tmax-1):
            dpos = link(alpha.T + X[:,i,:] @ beta)
            dneg = l.T * X[:,i,:]
            X[:,i+1,:] = X[:,i,:] + dt*(dpos - dneg)
            X[:,i+1,:] += s * np.sqrt(dt * X[:,i,:]) * np.random.normal(0, 1, size=(n, self.n))
            X[:,i+1,:] = np.maximum(0, X[:,i+1,:]) # clip negative values

            # check convergence
            if i % step == 0 and i - step > burnin:
                now  = X[0, burnin:i, :].mean(axis=0)
                then = X[0, burnin:(i-step), :].mean(axis=0)
                if np.max(np.abs(np.log2(now/then))[now > s]) < tol:
                    X = X[:, :i, :]
                    converged = True
                    break        
        
        # pass into observation model
        X1 = self.observe_rna(X[:, burnin:i, :])
        # done!
        if save:
            self.converged = converged
            self.s = s
            self.dt = dt
            self.tol = tol
            self.rna = X1[0]
        return X1

    
    def observation_model(self, X, t=100000):
        return np.mean(X[:,-t:,:], axis=1)

    
    def set_rna_observation_model(self, f):
        self.observe_rna=lambda X: f(X)
        return self
    
    
    def perturb(self, new_alpha=None, new_beta=None, new_l=None, **kwargs):
        # self.simulate_rna will automatically populate old parameters if new ones are empty 
        #  and additional keyword arguments are passed on
        return self.simulate_rna(save=False, alpha=new_alpha, beta=new_beta, l=new_l, **kwargs)

    
    def ko_all_nodes(self, n_jobs=1, **kwargs):
        # use parallel processing and keep a fun little progress bar
        from joblib import Parallel, delayed
        from tqdm import tqdm
        
        # compute baseline rna if we need it
        if not hasattr(self, 'rna'):
            self.simulate_rna(save=True, **kwargs)
        
        # do each ko in turn, using the helper function below
        return np.array(Parallel(n_jobs = n_jobs, pre_dispatch = 'n_jobs', prefer='processes'
                                )(delayed(self.ko_one_node)(i) for i in tqdm(range(self.n))))
    

    def ko_one_node(self, gene, stat='logfc', **kwargs):
        # gene is an index from 0, ..., self.n - 1
        new_beta = self.beta.copy()
        new_beta[gene,:] = 0
        
        # kwargs get passed directly into self.simulate_rna by way of self.perturb
        new_rna = self.perturb(new_beta = new_beta, x0 = self.rna, **kwargs)
       
        # meh 
        if stat == 'logfc':
            return np.log2(new_rna.flatten()) - np.log2(self.rna.flatten())




if __name__=="__main__":
    main()