from python_algorithms.basic.union_find import UF

cst = 50000


def get_true_clusters(nodes):
    uf = UF(len(nodes))
    print(len(nodes))
    n_map = {}
    for n in range(len(nodes)):
        n_map[nodes[n]] = n

    prev_in = '0'
    prev_out = '0'
    prev_tx = '0'
    with open('../nerd/temp.dat', 'r') as f:
        edge = f.readline()
        while edge is not None:
            tokens = edge.strip().split('\t')
            if len(tokens) < 4:
                break
            txid = tokens[0]
            in_id = int(tokens[1]) + cst
            out_id = int(tokens[2]) + cst

            if in_id == out_id:
                edge = f.readline()
                continue

            elif txid == prev_tx:
                if uf.find(n_map[in_id]) != uf.find(n_map[prev_in]):
                    uf.union(n_map[in_id], n_map[prev_in])

                # if uf.find(n_map[out_id]) != uf.find(n_map[prev_out]):
                #    uf.union(n_map[out_id], n_map[prev_out])

            prev_tx = txid
            prev_in = in_id
            prev_out = out_id

            edge = f.readline()

    c_map = {}
    for id in range(len(uf._id)):
        if nodes[uf._id[id]] in c_map:
            c_map[nodes[uf._id[id]]].append(nodes[id])
        else:
            c_map[nodes[uf._id[id]]] = [nodes[id]]
    print(c_map)
    print(len(c_map.keys()))

    return uf._id