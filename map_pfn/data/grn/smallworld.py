#!/usr/bin/env python
import networkx as nx
import numpy as np


def grouped_scale_free_graph(
    n,
    alpha=0.05,
    beta=0.54,
    gamma=0.41,
    delta_in=2,
    delta_out=0.0,
    k=None,
    kappa=1,
    create_using=None,
    seed=None,
):
    """Returns a scale-free directed graph with group structure.

    Parameters
    ----------
    n : integer
        Number of nodes in graph
    alpha : float
        Probability for adding a new node connected to an existing node
        chosen randomly according to the in-degree distribution.
    beta : float
        Probability for adding an edge between two existing nodes.
    gamma : float
        Probability for adding a new node connected to an existing node
        chosen randomly according to the out-degree distribution.
    delta : float
        Bias for choosing nodes from out-degree distribution.
    k : integer, or None (default, becomes 0.1*n)
        Number of subclasses to be formed in graph.
    kappa : float, default 1
        Preferential attachment within groups for edge formation.

    Notes
    -----
    The sum of `alpha`, `beta`, and `gamma` must be 1.

    References: see also networkx.scale_free_graph (similar README)
    ----------
    .. [1] B. Bollobás, C. Borgs, J. Chayes, and O. Riordan,
           Directed scale-free graphs,
           Proceedings of the fourteenth annual ACM-SIAM Symposium on
           Discrete Algorithms, 132--139, 2003.
    """
    # Check values
    if alpha < 0:
        raise ValueError("alpha must be >= 0.")
    if beta < 0:
        raise ValueError("beta must be >= 0.")
    if gamma < 0:
        raise ValueError("gamma must be >= 0.")
    if abs(alpha + beta + gamma - 1.0) >= 1e-9:
        raise ValueError("alpha+beta+gamma must equal 1.")
    if delta_in < 0:
        raise ValueError("delta_in must be >= 0.")
    if delta_out < 0:
        raise ValueError("delta_out must be >= 0.")
    if kappa < 0:
        raise ValueError("kapppa must be >= 0")
    if seed is None:
        seed = None
    if k is None:
        k = 1
    # todo: check that k is a non-negative integer

    # Start with 3-cycle: (k-cycle massively drops modularity)
    V = {i for i in range(3)}  # vertexes
    if k < 3:  # classes
        if k == 2:
            K = {0: {0, 2}, 1: {1}}
        else:
            K = {0: {0, 1, 2}}
    else:
        K = {i: {i} for i in range(len(V))}
    E_s = [i for i in V]  # edge sources
    E_t = [(i + 1) % len(V) for i in V]  # edge targets
    D = {
        "in": {i: E_t.count(i) for i in V},  # in-degree
        "out": {i: E_s.count(i) for i in V},
    }  # out-degree

    while len(V) < n:
        r = np.random.random()
        # random choice in alpha,beta,gamma ranges
        if r < gamma:
            # gamma: add new node v with random class k_v
            v = len(V)
            V.add(v)
            D["out"][v] = 0
            D["in"][v] = 0
            k_v = np.random.choice(range(k))
            if k_v not in K:
                K[k_v] = {v}
            else:
                K[k_v].add(v)
            # pick w by out degree biased towards type k_v
            p = [(kappa if n in K[k_v] else 1) * (D["out"][n] + delta_out) for n in range(len(V))]
            w = np.random.choice(list(range(len(V))), p=p / np.sum(p))
        elif r < gamma + alpha:
            # alpha: add new node w with random class k_w
            w = len(V)
            V.add(w)
            D["out"][w] = 0
            D["in"][w] = 0
            k_w = np.random.choice(range(k))
            if k_w not in K:
                K[k_w] = {w}
            else:
                K[k_w].add(w)
            # pick v by out degree biased towards type k_w
            q = [(kappa if n in K[k_w] else 1) * (D["in"][n] + delta_in) for n in range(len(V))]
            v = np.random.choice(list(V), p=q / np.sum(q))
        else:
            # beta: pick w by out degree with bias delta_out and v as in alpha regime
            p = [(D["out"][n] + delta_out) for n in V]
            w = np.random.choice(list(range(len(V))), p=p / np.sum(p))
            k_w = [group for group, nodes in K.items() if w in nodes][0]
            q = [(kappa if n in K[k_w] else 1) * (D["in"][n] + delta_in) for n in range(len(V))]
            v = np.random.choice(list(range(len(V))), p=q / np.sum(q))
        # add edge (w,v)
        E_s.append(w)
        E_t.append(v)
        D["out"][w] += 1
        D["in"][v] += 1

    G = nx.MultiDiGraph(zip(E_s, E_t))
    nx.set_node_attributes(G, {n: {"group": k} for k, v in K.items() for n in v})
    return G
