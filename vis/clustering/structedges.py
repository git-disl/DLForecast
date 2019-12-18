prev_txid = '0'
in_map = {}
out_map = {}

cst = 50000

with open('struct_edgelist.dat', 'w') as wf:
    with open('../nerd/temp.dat', 'r') as f:
        edge = f.readline()
        while edge is not None:
            tokens = edge.strip().split('\t')
            if len(tokens) < 4:
                break
            txid = tokens[0]
            in_id = str(int(tokens[1]) + cst)
            out_id = str(int(tokens[2]) + cst)
            weight = float(tokens[3])
            print(str(txid) + ' ' + str(in_id))

            if in_id == out_id:
                edge = f.readline()
                continue

            if txid == prev_txid:
                if in_id in in_map:
                    in_map[in_id] = in_map[in_id] + weight
                else:
                    in_map[in_id] = weight

                if out_id in out_map:
                    out_map[out_id] = out_map[out_id] + weight
                else:
                    out_map[out_id] = weight

            elif prev_txid != '0':
                for key, value in in_map.items():
                    wf.write(" ".join([key, prev_txid, str(value)]) + '\n')
                    print("Write row: "+ " ".join([key, prev_txid, str(value)]))
                for key, value in out_map.items():
                    wf.write(" ".join([prev_txid, key, str(value)]) + '\n')
                    print("Write row: " + " ".join([prev_txid, key, str(value)]))

                in_map = {}
                out_map = {}
                prev_txid = txid
                in_map[in_id] = weight
                out_map[out_id] = weight

            else:
                prev_txid = txid
                in_map[in_id] = weight
                out_map[out_id] = weight

            edge = f.readline()

        for key, value in in_map.items():
            wf.write(" ".join([key, prev_txid, str(value)]) + '\n')
            print("Write row: " + " ".join([key, prev_txid, str(value)]))
        for key, value in out_map.items():
            wf.write(" ".join([prev_txid, key, str(value)]) + '\n')
            print("Write row: " + " ".join([prev_txid, key, str(value)]))