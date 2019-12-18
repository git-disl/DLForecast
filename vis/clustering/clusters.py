from sklearn.cluster import DBSCAN, OPTICS, SpectralClustering, AgglomerativeClustering
from sklearn import metrics
import numpy as np
from clustering.union_find import get_true_clusters
from verse.python.embedding import Embedding

node_vecs = list()
# file_name = '../nerd/auth.txt'
file_name = '../verse/src/struct_vec.bin'


def get_nodes():
    nodes = set()
    with open('struct_edgelist.dat', 'r') as inf:
        for line in inf:
            if line.startswith('#') or line.startswith('%'):
                continue
            line = line.strip()
            splt = line.split(' ')
            if not splt: continue
            splt = splt[:-1]
            for node in splt:
                nodes.add(node)
    number_of_nodes = len(nodes)
    nodes = sorted(map(int, nodes))
    node2id = dict(zip(nodes, range(number_of_nodes)))
    print(nodes)
    return nodes


def get_verse_embeddings():
    nodes = get_nodes()
    em = Embedding(file_name, 128)
    return em.embeddings, nodes


def get_node_vectors():
    nodes = list()
    read_flag = 'r'
    # if file_name.endswith('bin'):
    #     read_flag = 'rb'
    #     X = np.fromfile(file_name, dtype=float)
    #     print(X)
    #     print(X.shape)
    with open(file_name, read_flag) as f:
        meta = f.readline()
        print(meta)
        num_nodes, vec_len = meta.strip().split(' ')

        for i in range(int(num_nodes)):
            line = f.readline()
            node = line.strip().split(' ')[0]
            if int(node) > 50000:
                node_vec = line.strip().split(' ')[1:]
                nodes.append(node)
                node_vec = [float(n) for n in node_vec]
                node_vecs.append(node_vec)

    X = np.array(node_vecs)
    return X, nodes


def get_clusters():
    X, nodes = get_verse_embeddings()
    # X, nodes = get_node_vectors()
    #cs = OPTICS(min_samples=2, p=1).fit(X)
    # cs = SpectralClustering(n_clusters=642).fit(X)

    cs = AgglomerativeClustering(n_clusters=642, linkage="single").fit(X)

    labels = {}
    for l in range(len(cs.labels_)):
        if cs.labels_[l] in labels:
            labels[cs.labels_[l]].append(nodes[l])
        else:
            labels[cs.labels_[l]] = [nodes[l]]

    for key, value in labels.items():
        print(key)
        print(value)

    return cs, nodes


def main():
    cs, nodes = get_clusters()
    #X = get_node_vectors()
    true_clusters = get_true_clusters(nodes)
    acc = metrics.adjusted_rand_score(true_clusters, cs.labels_)
    print("Accuracy: " + str(acc))


main()
