%load_ext Cython
%%cython 
import numpy as np
import itertools as it
from scipy.special import gammaln,gamma,loggamma
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram as dd
from functools import lru_cache

class Node:
    def __init__(self,data,alpha,beta=None,left=None,right=None):
        """
        Initialize a bayesian hierarchical clustering with the following parameters.
        Data: NArray
        Alpha: Hyperparameter
        Beta: Hyperparameter
        Left: Left child node
        Right: Right child node
        """  
        self.data = data
        self.alpha = alpha
        self.beta = beta
        self.left = left
        self.right = right
        self.n_k = data.shape[0]
        """The prior probability on the merged hypothesis"""
        if left:
            self.d_k = alpha * gamma(self.n_k)+self.left.d_k * self.right.d_k
            self.pi_k = alpha * gamma(self.n_k)/self.d_k
        else:
            self.d_k = alpha
            self.pi_k = 1 
 
 def bhc(data,alpha,beta=None):
    """ 
    This function does a bayesian clustering. 
    Alpha: Hyperparameter
    Beta: Hyperparameter
    
    If beta is not given, it uses the Multinomial-Dirichlet.
    Otherwise it uses Bernoulli-Beta.
    """
    n_cluster = data.shape[0]
    list_clusters = list(range(n_cluster))
    clusters = {"n_cluster":n_cluster}
    clusters[n_cluster] = (1,[str(i+1) for i in range(n_cluster)])
    tree = {str(i+1):Node(data=np.array([data[i,:]]),alpha=alpha,beta=beta,left=None,right=None) 
                 for i in range(n_cluster)}
    while n_cluster > 1:
        "Find the pair with the highest probability of the merged hypothesis"
        r_k_max = float('-Inf')
        for left,right in list(it.combinations(tree.keys(), 2)):
            aux_data = np.vstack((tree[left].data,tree[right].data))
            aux_node = Node(aux_data,alpha,beta=beta,left=tree[left],right=tree[right])
            r_k = posterior(aux_node)
            if r_k > r_k_max:
                r_k_max = r_k
                merged_left = left
                merged_right = right
                merged_node = aux_node

        merged_node.r_k = r_k_max

        newkey = merged_left+','+ merged_right
        del tree[merged_left]
        del tree[merged_right]
        tree[newkey] = merged_node  
        n_cluster -= 1
        clusters[n_cluster] = (r_k_max,list(tree.keys()))
    return clusters,merged_node
 
 @lru_cache(maxsize=32)
def posterior(node):
    """
    Calculates the posterior proabability of the merged hypothesis
    """
    return node.pi_k * prob_dH1_k(node) / prob_dT_k(node)

@lru_cache(maxsize=32)
def prob_dT_k(node):
    """ 
    Calculates the marginal probability of the data in tree Tk
    """
    if node.left:
        return node.pi_k * prob_dH1_k(node) + (1-node.pi_k) * prob_dT_k(node.left) * prob_dT_k(node.right)
    else: 
        return node.pi_k * prob_dH1_k(node)
    
@lru_cache(maxsize=32)        
def prob_dH1_k(node):
    """
    Calculates the marginal likelihood using the following model:
        Bernoulli-Beta if beta is given.
        Multinomial-Dirichlet if only alpha is given.
    """
    N = node.data.shape[0]
    k = node.data.shape[1] 
    prob = 0
    if node.beta:
        m_d = np.sum(node.data,axis=0) 
        term1 = loggamma(node.alpha+node.beta) + loggamma(node.alpha+m_d) + loggamma(node.beta+N-m_d)
        term2 = loggamma(node.alpha) + loggamma(node.beta) + loggamma(node.alpha+node.beta+N)      
        prob = np.exp(np.sum(term1-term2))
    else:
        alpha = np.repeat(node.alpha, k)
        coefficient = np.sum(loggamma(node.data.sum(axis=1)+1) - loggamma(node.data+1).sum(axis=1))
        loggamma_alpha = loggamma(node.alpha)
        term1 = np.sum(loggamma(node.alpha + node.data.sum(axis=0)) - loggamma_alpha)
        term2 = loggamma(np.sum(alpha))
        term3 = loggamma(np.sum(node.data) + np.sum(alpha))
        prob = np.exp(coefficient + term1 + term2 -term3)
    return prob
def cut_tree_n(n_clusters,clusters):
    """
    Gives the clusters number by cutting the tree with n clusters
    n_clusters: Number of clusters chosen by the user
    clusters: Dictionary with all clusters (output from bhc function)
    """
    aux_cluster = clusters[n_clusters][1]
    n = clusters["n_cluster"]
    assigments=np.zeros(n)
    for i,c in enumerate(aux_cluster):
        if len(c)>1:
            for j in c.split(","):
                assigments[int(j)-1]=i
        else:
            assigments[int(c)-1] = i 
    return assigments
def dendogram(clusters):
    """
    Builds the dendrogram matrix needed to plot it with scipy
    clusters : Dictionary with all clusters (output from bhc function)
    """
    new_cluster = {}
    obs = clusters["n_cluster"]+1
    n = clusters["n_cluster"]
    Z = np.zeros(shape=(n-1,4))
    for i in range(1,obs-1):
        old_set = set(clusters[obs-i][1])
        new_set = set(clusters[obs-(i+1)][1])
        new_node = new_set.difference(old_set)
        old_nodes = old_set.difference(new_set)
        node_a = old_nodes.pop()
        node_b = old_nodes.pop()
        node_c = new_node.pop()
        new_cluster[node_c] = n
        if "," in node_b:
            num_b = new_cluster[node_b]
        else:
            num_b = int(node_b)-1
        if "," in node_a:
            num_a = new_cluster[node_a]
        else:
            num_a = int(node_a)-1
        rk = clusters[obs-(i+1)][0]
        Z[i-1,0] = num_a
        Z[i-1,1] = num_b
        Z[i-1,2] = np.abs(np.log(rk/(1-rk)))
        Z[i-1,3] = len(node_c.split(","))
        n+=1
    return Z
