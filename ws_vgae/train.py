from fastgae.evaluation import *
from fastgae.input_data import *
from fastgae.model import *
from fastgae.optimizer import *
from fastgae.preprocessing import *
from fastgae.sampling import *
import numpy as np
import os
import scipy.sparse as sp
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
flags = tf.app.flags
FLAGS = flags.FLAGS

# Select graph dataset
flags.DEFINE_string('dataset', 'cora', 'Name of the graph dataset')
''' Available datasets:
- cora: with and without node features
- citeseer: with and without node features
- pubmed: with and without node features
- webkd: with and without node features
- google: the "Google Large" dataset from the paper
- google-medium
- blogs
- cora-large
- sbm
- hamster
- arxiv-hep
- artists
'''

# Select downstream task to evaluate the model
flags.DEFINE_string('task', 'link_prediction', 'Name of the task')
''' See Section 3.1 of the paper for details about tasks:
- link_prediction
- community_detection
'''

# Model
flags.DEFINE_string('model', 'gcn_ae', 'Name of the model')
''' Available models:
- gcn_ae: deterministic GAE (not used in the paper)
- gcn_vae: VGAE with weight sharing (WS)
- gcn_vae_nows: VGAE without WS
- deep_gcn_vae: Deep VGAE with WS
- deep_gcn_vae_nows: Deep VGAE without WS
'''

# Model parameters
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability)')
flags.DEFINE_integer('iterations', 200, 'Number of iterations in training')
flags.DEFINE_boolean('features', False, 'Include node features or not in encoder')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate (with Adam)')
flags.DEFINE_integer('hidden', 32, 'Number of units in GCN hidden layer')
flags.DEFINE_integer('dimension', 16, 'Dimension of encoder output, i.e. \
                                       embedding dimension')

# FastGAE parameters
flags.DEFINE_boolean('fastgae', False, 'Whether to use the FastGAE framework')
flags.DEFINE_integer('nb_node_samples', 1000, 'Number of nodes to sample at each \
                                        iteration, i.e. sampled subgraph size')
flags.DEFINE_string('measure', 'degree', 'Node importance measure used in \
                                          sampling: degree, core or uniform')
flags.DEFINE_float('alpha', 2.0, 'alpha hyperparameter of p_i distribution')
flags.DEFINE_boolean('replace', False, 'Whether to sample nodes with (True) \
                                        or without (False) replacement')

# Experimental setup parameters
flags.DEFINE_integer('nb_run', 1, 'Number of model run + test')
flags.DEFINE_float('prop_val', 5., 'Proportion of edges in validation set \
                                   (for Link Prediction task)')
flags.DEFINE_float('prop_test', 10., 'Proportion of edges in test set \
                                      (for Link Prediction task)')
flags.DEFINE_boolean('validation', False, 'Whether to report validation \
                                           results at each iteration (for \
                                           Link Prediction task)')
flags.DEFINE_boolean('verbose', True, 'Whether to print comments details')

# Lists to average final results
if FLAGS.task == 'link_prediction':
    mean_roc = []
    mean_ap = []
elif FLAGS.task == 'community_detection':
    mean_ami = []
    mean_ari = []
else:
    raise ValueError('Undefined task!')
mean_time = []

# Load graph dataset
if FLAGS.verbose:
    print("Loading data...")
adj_init, features_init = load_data(FLAGS.dataset)

# Load community labels for the community detection task
if FLAGS.task == 'community_detection':
    labels = load_label(FLAGS.dataset)

# The entire training+test process is repeated FLAGS.nb_run times
for i in range(FLAGS.nb_run):

    # Preprocessing and initialization steps
    if FLAGS.verbose:
        print("Preprocessing data...")

    # Edge Masking for Link Prediction:
    if FLAGS.task == 'link_prediction' :
        # Compute Train/Validation/Test sets
        adj, val_edges, val_edges_false, test_edges, test_edges_false = \
        mask_test_edges(adj_init, FLAGS.prop_test, FLAGS.prop_val)
    else:
        adj = adj_init

    # Compute number of nodes
    num_nodes = adj.shape[0]

    # Preprocessing on node features
    if FLAGS.features:
        features = features_init
    else:
        # If features are not used, replace feature matrix by identity matrix
        features = sp.identity(num_nodes)
    features = sparse_to_tuple(features)
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]

    # Start computation of running times
    t_start = time.time()

    # FastGAE: node sampling for stochastic subgraph decoding
    if FLAGS.fastgae:
        if FLAGS.verbose:
            print("Computing p_i distribution for", FLAGS.measure, "sampling")
        t_proba = time.time()
        # Node-level p_i degree-based, core-based, or uniform distribution
        node_distribution = get_distribution(FLAGS.measure, FLAGS.alpha, adj)
        # Node sampling
        sampled_nodes, adj_label, adj_sampled_sparse = node_sampling(adj, node_distribution,
                                                       FLAGS.nb_node_samples, FLAGS.replace)
    else:
        sampled_nodes = np.array(range(FLAGS.nb_node_samples))

    # Define placeholders
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape = ()),
        'sampled_nodes': tf.placeholder_with_default(sampled_nodes, shape = [FLAGS.nb_node_samples])
    }

    # Create model
    if FLAGS.model == 'gcn_ae':
        # Deterministic Graph Autoencoder (GAE)
        model = GCNModelAE(placeholders, num_features, features_nonzero)
    elif FLAGS.model == 'gcn_vae':
        # Variational Graph Autoencoder (VGAE) with Weight Sharing (WS)
        model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero)
    elif FLAGS.model == 'gcn_vae_nows':
        # VGAE without WS
        model = GCNModelVAENoWS(placeholders, num_features, num_nodes, features_nonzero)
    elif FLAGS.model == 'deep_gcn_vae':
        # Deep VGAE with WS
        model = DeepGCNModelVAE(placeholders, num_features, num_nodes, features_nonzero)
    elif FLAGS.model == 'deep_gcn_vae_nows':
        # Deep VGAE without WS
        model = DeepGCNModelVAENoWS(placeholders, num_features, num_nodes, features_nonzero)
    else:
        raise ValueError('Undefined model!')

    # Optimizer
    if FLAGS.fastgae:
        num_sampled = adj_sampled_sparse.shape[0]
        sum_sampled = adj_sampled_sparse.sum()
        pos_weight = float(num_sampled * num_sampled - sum_sampled) / sum_sampled
        norm = num_sampled * num_sampled / float((num_sampled * num_sampled
                                                    - sum_sampled) * 2)
    else:
        pos_weight = float(num_nodes * num_nodes - adj.sum()) / adj.sum()
        norm = num_nodes * num_nodes / float((num_nodes * num_nodes
                                                    - adj.sum()) * 2)
    if FLAGS.model == 'gcn_ae':
        opt = OptimizerAE(preds = model.reconstructions,
                          labels = tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                        validate_indices = False), [-1]),
                          pos_weight = pos_weight,
                          norm = norm)
    else:
        opt = OptimizerVAE(preds = model.reconstructions,
                           labels = tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                         validate_indices = False), [-1]),
                           model = model,
                           num_nodes = num_nodes,
                           pos_weight = pos_weight,
                           norm = norm)

    # Normalization and preprocessing on the adjacency matrix
    adj_norm = preprocess_graph(adj)
    if not FLAGS.fastgae:
        adj_label = sparse_to_tuple(adj + sp.eye(num_nodes))

    # Initialize TF session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Model training
    if FLAGS.verbose:
        print("Training...")

    for iter in range(FLAGS.iterations):
        # Flag to compute running time for each iteration
        t = time.time()
        
        # Construct feed dictionary
        feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)

        if FLAGS.fastgae:
            # Update sampled subgraph
            feed_dict.update({placeholders['sampled_nodes']: sampled_nodes})
            # New node sampling
            sampled_nodes, adj_label, _ = node_sampling(adj, node_distribution,
                                          FLAGS.nb_node_samples, FLAGS.replace)

        # Weights update
        outs = sess.run([opt.opt_op, opt.cost, opt.accuracy],
                        feed_dict = feed_dict)

        # Compute average loss
        avg_cost = outs[1]

        if FLAGS.verbose:
            # Display iteration information
            print("Iter:", '%04d' % (iter + 1), "train_loss=", "{:.5f}".format(avg_cost),
                  "time=", "{:.5f}".format(time.time() - t))
            # Validation, for link prediction
            if FLAGS.validation and FLAGS.task == 'link_prediction':
                feed_dict.update({placeholders['dropout']: 0})
                emb = sess.run(model.z_mean, feed_dict = feed_dict)
                feed_dict.update({placeholders['dropout']: FLAGS.dropout})
                val_roc, val_ap = get_roc_score(val_edges, val_edges_false, emb)
                print("val_roc=", "{:.5f}".format(val_roc), "val_ap=", "{:.5f}".format(val_ap))

    # Get the final node embedding vectors
    emb = sess.run(model.z_mean, feed_dict = feed_dict)

    # Compute total running time
    mean_time.append(time.time() - t_start)

    # Test model
    if FLAGS.verbose:
        print("Testing model...")

    # Link prediction: classify edges/non-edges
    if FLAGS.task == 'link_prediction':
        # Link prediction on test set
        roc_score, ap_score = get_roc_score(test_edges, test_edges_false, emb)
        # Report AUC and AP scores
        mean_roc.append(roc_score)
        mean_ap.append(ap_score)

    # Community detection: k-means clustering in embedding space
    else:
        # Clustering in the node embedding space
        mi_score, ari_score = clustering_latent_space(emb, labels)
        # Report AMI and ARI scores
        mean_ami.append(mi_score)
        mean_ari.append(ari_score)

# Report final results
print("\nTest results for", FLAGS.model,
      " - Model: ", FLAGS.dataset, " - Task: ", FLAGS.task, "\n",
      "___________________________________________________\n")

if FLAGS.task == 'link_prediction':
    print("AUC scores\n", mean_roc)
    print("Mean AUC score: ", np.mean(mean_roc),
          "\nStd of AUC scores: ", np.std(mean_roc), "\n \n")

    print("AP scores\n", mean_ap)
    print("Mean AP score: ", np.mean(mean_ap),
          "\nStd of AP scores: ", np.std(mean_ap), "\n \n")

else:
    print("AMI scores\n", mean_ami)
    print("Mean AMI score: ", np.mean(mean_ami),
          "\nStd of AMI scores: ", np.std(mean_ami), "\n \n")

    print("ARI scores\n", mean_ari)
    print("Mean  ARI score: ", np.mean(mean_ari),
          "\nStd of ARI scores: ", np.std(mean_ari), "\n \n")

print("Total Running times\n", mean_time)
print("Mean total running time: ", np.mean(mean_time),
      "\nStd of total running time: ", np.std(mean_time), "\n")
