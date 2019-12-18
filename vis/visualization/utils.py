from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
import pandas as pd

auth_provider = PlainTextAuthProvider(username='admin', password='admin123')
cluster = Cluster(['35.238.43.195'], port=9042, auth_provider=auth_provider)
session = cluster.connect('bitcoin')


def pre_processed_data(rows):
    time = []
    amount = []
    nbr_id = []
    txn_id = []
    role = []
    for r in rows:
        time.append(r[6])
        amount.append(r[4])
        nbr_id.append(r[3])
        txn_id.append(r[2])
        role.append(r[5])
    print(len(time))
    print(len(txn_id))
    return time,amount,nbr_id,txn_id,role


def pre_process_info(rows):
    node_info = {}
    for r in rows:
        print(r)
        node_info['first_active'] = r[2]
        node_info['in_deg'] = r[3]
        node_info['last_active'] = r[4]
        node_info['num_rows'] = r[5]
        node_info['num_txs'] = r[6]
        node_info['out_deg'] = r[7]
    return node_info


def get_data_txns(add_id, block_id):
    print("-------------------")
    print(add_id, block_id)
    query = 'SELECT * from transactions where add_id={} and block_id={}'.format(add_id, block_id)
    rows = session.execute(query)
    #return pre_processed_data(rows)
    df = pd.DataFrame(list(rows))
    df = df.sort_values(by=['timestamp'], ascending=False)
    df['amount'] = round(df['amount'] * 0.00000001, 2)
    return df


def get_node_info(node_id):
    query = 'SELECT * from node_info where add_id={}'.format(node_id)
    rows = session.execute(query)
    return pre_process_info(rows)