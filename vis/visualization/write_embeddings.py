from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra import ConsistencyLevel
import time
from datetime import datetime

auth_provider = PlainTextAuthProvider(username='admin', password='admin123')
cluster = Cluster(['35.238.43.195'], port=9042, auth_provider=auth_provider)
session = cluster.connect('bitcoin')
session.default_consistency_level = ConsistencyLevel.ALL

insert = "insert into vector_embeddings (add_id, vector) values (?,?)"
insert_stmt = session.prepare(insert)

futs = []
with open ('data/auth.txt')as f:
    meta = f.readline()
    # num_nodes = meta.strip().split(' ')
    for line in f:
        tokens = line.strip().split(' ')
        node = int(tokens[0])-50000
        if node > 0:
            vec = [float(i) for i in tokens[1:]]
            futs.append(session.execute_async(insert_stmt, [node, vec]))

for fut in futs:
    print(fut.result())
