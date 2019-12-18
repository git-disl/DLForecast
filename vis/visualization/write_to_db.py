from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra import ConsistencyLevel
import time
from datetime import datetime

auth_provider = PlainTextAuthProvider(username='admin', password='admin123')
cluster = Cluster(['35.238.43.195'], port=9042, auth_provider=auth_provider)
session = cluster.connect('bitcoin')
session.default_consistency_level = ConsistencyLevel.ALL
tx_insert = "insert into transactions (add_id, block_id, nbr_id, txid, role, amount, timestamp) values (?,?,?,?,?,?,?)"
info_insert = "insert into node_info (add_id, num_txs, num_rows, in_deg, out_deg, first_active, last_active) values (?,?,?,?,?,?,?)"
tx_insert_stmt = session.prepare(tx_insert)
info_insert_stmt = session.prepare(info_insert)

block_map = {}
tx_map = {}

with open('data/bh.dat') as bf:
    # for line in bf:
    #    print(line)
    for i in range(1000000):
        line = bf.readline()
        tokens = line.split('\t')
        if len(tokens) > 2:
            block_map[tokens[0]] = tokens[2]


with open('data/tx_temp.dat') as tf:
    for line in tf:
        print(line)
        tokens = line.split('\t')
        if len(tokens) > 2:
            tx_map[tokens[0]] = block_map[tokens[1]]

del block_map
futs = []
ad_map = {}
in_map = {}
out_map = {}
fa_map = {}
ntx_map = {}

with open('data/txedges.dat') as f:
    for line in f:
    #for i in range(10):
        data = []
        tokens = line.strip().split('\t')
        if len(tokens) > 2:
            txid = tokens[0]
            in_id = int(tokens[1])
            out_id = int(tokens[2])
            amount = float(tokens[3])
            if txid not in tx_map:
                continue

            if in_id in ad_map:
                ad_map[in_id] = ad_map[in_id] + 1
                ntx_map[in_id].add(txid)
            else:
                fa_map[in_id] = tx_map[txid]
                ad_map[in_id] = 1
                in_map[in_id] = 0
                out_map[in_id] = 0
                ntx_map[in_id] = set([txid])

            if out_id in ad_map:
                ad_map[out_id]=ad_map[out_id]+1
                ntx_map[out_id].add(txid)
            else:
                fa_map[out_id] = tx_map[txid]
                ad_map[out_id] = 1
                in_map[out_id] = 0
                out_map[out_id] = 0
                ntx_map[out_id] = set([txid])

            if in_id != out_id:
                in_map[out_id] = in_map[out_id] + 1
                out_map[in_id] = out_map[in_id] + 1
                in_block_id = int(ad_map[in_id]/100)
                out_block_id = int(ad_map[out_id] / 100)
                data.append([in_id, in_block_id, out_id, int(txid), "source", amount, datetime.fromtimestamp(int(tx_map[txid]))])
                data.append([out_id, out_block_id, in_id, int(txid), "target", amount, datetime.fromtimestamp(int(tx_map[txid]))])
                futs.append(session.execute_async(tx_insert_stmt, data[0]))
                futs.append(session.execute_async(tx_insert_stmt, data[1]))
            futs.append(session.execute_async(info_insert_stmt, [in_id, len(ntx_map[in_id]), ad_map[in_id], in_map[in_id], out_map[in_id], datetime.fromtimestamp(int(fa_map[in_id])), datetime.fromtimestamp(int(tx_map[txid]))]))
            futs.append(session.execute_async(info_insert_stmt, [out_id, len(ntx_map[out_id]), ad_map[out_id], in_map[out_id], out_map[out_id], datetime.fromtimestamp(int(fa_map[out_id])), datetime.fromtimestamp(int(tx_map[txid]))]))

for fu in futs:
    print(fu.result())
