import networkx as nx
import numpy as np
import pickle as pkl
import scipy.sparse as sp
import sys
import zipfile as zf


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_data(dataset):
    """ Load datasets
    :param dataset: name of the input graph dataset
    :return: n*n sparse adjacency matrix and n*f node features matrix
    """
    if dataset == 'cora-large':
        adj = nx.adjacency_matrix(nx.read_edgelist("../data/coralarge", delimiter = ' '))
        features = sp.identity(adj.shape[0])

    elif dataset == 'sbm':
        adj = nx.adjacency_matrix(nx.read_edgelist("../data/sbm.txt"))
        features = sp.identity(adj.shape[0])

    elif dataset == 'blogs':
        adj = nx.adjacency_matrix(nx.read_edgelist("../data/blogs",
                                                   nodetype = int,
                                                   data = (('weight', int),),
                                                   delimiter = ' '))
        features = sp.identity(adj.shape[0])

    elif dataset == 'google':
         zf.ZipFile("../data/google.txt.zip").extract("google.txt", "../data/")
         adj = nx.adjacency_matrix(nx.read_edgelist("../data/google.txt"))
         features = sp.identity(adj.shape[0])
         
    elif dataset == 'webkd':
        adj = nx.adjacency_matrix(nx.read_edgelist("../data/webkd", delimiter=' '))
        adj = adj.T # Needed due to WebKD data format
        features = sp.csr_matrix(np.loadtxt("../data/webkd-content", usecols=range(1,1704), delimiter = '\t'))

    elif dataset == 'hamster':
        adj = nx.adjacency_matrix(nx.read_edgelist("../data/hamster", delimiter=' '))
        features = sp.identity(adj.shape[0])

    elif dataset == 'google-medium':
        adj = nx.adjacency_matrix(nx.read_edgelist("../data/GoogleNw.txt", delimiter='\t'))
        features = sp.identity(adj.shape[0])
        
    elif dataset == 'arxiv-hep':
        adj = nx.adjacency_matrix(nx.read_edgelist("../data/arxiv-hep.txt", delimiter='\t'))
        features = sp.identity(adj.shape[0])
    
    elif dataset == 'artists':
        adj = nx.adjacency_matrix(nx.read_weighted_edgelist("../data/deezer_graph.csv",
                                                           delimiter=',',
                                                           nodetype =float))
        adj = adj - sp.eye(adj.shape[0])
        features = sp.identity(adj.shape[0])

    elif dataset in ('cora', 'citeseer', 'pubmed'):
        # Load the data: x, tx, allx, graph
        names = ['x', 'tx', 'allx', 'graph']
        objects = []
        for i in range(len(names)):
            with open("../data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))
        x, tx, allx, graph = tuple(objects)
        test_idx_reorder = parse_index_file("../data/ind.{}.test.index".format(dataset))
        test_idx_range = np.sort(test_idx_reorder)
        if dataset == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range-min(test_idx_range), :] = tx
            tx = tx_extended
        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        graph = nx.from_dict_of_lists(graph)
        adj = nx.adjacency_matrix(graph)
    else:
        raise ValueError('Undefined dataset!')
    return adj, features

def load_label(dataset):
    """ Load node-level labels
    :param dataset: name of the input graph dataset
    :return: n-dim array of node labels, used for community detection
    """
    if dataset == 'cora-large':
        labels = np.loadtxt("../data/coralarge-cluster", delimiter = ' ', dtype = str)

    elif dataset == 'sbm':
        labels = np.repeat(range(100), 1000)

    elif dataset == 'blogs':
        labels = np.loadtxt("../data/blogs-cluster", delimiter = ' ', dtype = str)
        
    elif dataset == 'artists':
        labels = np.genfromtxt("../data/deezer_features.csv", delimiter=",")[:,33:53]
        labels = [np.nonzero(row)[0][0] for row in labels]
        
    elif dataset in ('cora', 'citeseer', 'pubmed'):
        names = ['ty', 'ally']
        objects = []
        for i in range(len(names)):
            with open("../data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))
        ty, ally = tuple(objects)
        test_idx_reorder = parse_index_file("../data/ind.{}.test.index".format(dataset))
        test_idx_range = np.sort(test_idx_reorder)
        if dataset == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
            ty_extended = np.zeros((len(test_idx_range_full), ty.shape[1]))
            ty_extended[test_idx_range-min(test_idx_range), :] = ty
            ty = ty_extended
        labels = sp.vstack((ally, ty)).tolil()
        labels[test_idx_reorder, :] = labels[test_idx_range, :]
        # One-hot to integers
        labels = np.argmax(labels.toarray(), axis = 1)
    else:
        raise ValueError('Undefined dataset!')
    return labels
