import numpy as np
import itertools as it
from scipy.special import gammaln,gamma,loggamma
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram as dd
from functools import lru_cache

class Node:
    def __init__(self,key,data,alpha,beta=None,left=None,right=None,parent=None):
        """
        Initialize a bayesian hierarchical clustering with the followin parameters.
        Key: Identifier
        Data: NArray
        Alpha: Hyperparameter
        Beta: Hyperparameter
        Left: Left child node
        Right: Right child node
        Parent: Parent node
        """  
        self.key = key
        self.data = data
        self.alpha = alpha
        self.beta = beta
        self.left = left
        self.right = right
        self.parent = None
        self.parent_rk = None
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
    nodekey = n_cluster
    list_clusters = [i for i in range(n_cluster)]
    clusters = dict()
    clusters["n_cluster"] = n_cluster
    clusters[n_cluster] = (1,[str(i+1) for i in range(n_cluster)])
    tree = {str(i+1):Node(key=i+1,data=np.array([data[i,:]]),alpha=alpha,beta=beta,left=None,right=None,parent=None) 
                 for i in range(n_cluster)}
    while n_cluster > 1:
        "Find the pair with the highest probability of the merged hypothesis"
        r_k_max = -1000000
        for left,right in list(it.combinations(tree.keys(), 2)):
            nodekey += 1
            aux_data = np.vstack((tree[left].data,tree[right].data))
            aux_node = Node(nodekey,aux_data,alpha,beta=beta,left=tree[left],right=tree[right])
            r_k = posterior(aux_node)
            #print(r_k)
            if r_k > r_k_max:
                r_k_max = r_k
                merged_left = left
                merged_right = right
                merged_node = aux_node

        merged_node.r_k = r_k_max
        merged_node.left.parent = merged_node
        merged_node.right.parent = merged_node

        newkey = merged_left+','+ merged_right
        del tree[merged_left]
        del tree[merged_right]
        tree[newkey] = merged_node  
        n_cluster -= 1
        clusters[n_cluster] = (r_k_max,list(tree.keys()))
        nodekey +=1
    return clusters,merged_node
@lru_cache(maxsize=32)
def posterior(node):
    """
    Calculates the posterior probability of the merged hypothesis
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
    k= node.data.shape[1] 
    if node.beta:
        m_d = np.sum(node.data,axis=0)
        prob = 0
        for i in range(k):
            term1 = loggamma(node.alpha+node.beta)
            term2 = loggamma(node.alpha+m_d[i])
            term3 = loggamma(node.beta+N-m_d[i])
        
            term4 = loggamma(node.alpha) 
            term5 = loggamma(node.beta)
            term6 = loggamma(node.alpha+node.beta+N)
            
            probi = term1+term2+term3-term4-term5-term6
            prob = prob + probi
      
        prob = np.exp(prob)
    else:
        alpha = np.repeat(node.alpha, k)
        coefficient = [loggamma(np.sum(node.data[i,:])+1)- np.sum(loggamma(node.data[i, :]+1)) for i in range(N)]
        term1 = np.sum(coefficient)
        sumterm = [loggamma(alpha[j] + np.sum(node.data[:,j])) - loggamma(alpha[j]) for j in range(k)]
        term2 = np.sum(sumterm)
        term3 = loggamma(np.sum(alpha))
        term4 = loggamma(np.sum(node.data) + np.sum(alpha))
        prob = np.exp(term1 + term2 + term3 -term4)
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
        Z[i-1,2] = abs(np.log(rk/(1-rk)))
        Z[i-1,3] = len(node_c.split(","))
        n+=1
    return Z
