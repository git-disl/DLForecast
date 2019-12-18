from flask import Flask, render_template, request
from utils import *
import matplotlib
import matplotlib.dates
import matplotlib.pyplot as plt

app = Flask(__name__, template_folder="/Users/shubhi/Bitcoin/gtcrypto/visualization/views/")


@app.route("/")
def index():
    return "Hello World!"

@app.route("/bitcoin/datasearch/", methods=['GET', 'POST'])
def hello():
    if request.method == 'GET':
        return render_template("landing.html", **locals())
    else:
        node_id = request.form['node_id']
        node_info = get_node_info(node_id)
        time_list = []
        amount_list = []
        nbr_id_list = []
        txn_id_list =[]
        role_list =[]

        i = int(node_info['num_rows']/100)
        # for i in range(int(node_info['num_txs']/100)+1):
        df = get_data_txns(node_id, i)
        time_list.extend(df['timestamp'])
        amount_list.extend(df['amount'])
        nbr_id_list.extend(df['nbr_id'])
        txn_id_list.extend(df['txid'])
        role_list.extend(df['role'])

        # print("------------}}}")
        # print(len(time_list))
        # print(len(txn_id_list))
        # dates = matplotlib.dates.date2num(time)
        fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
        ax.plot_date(time_list, amount_list)
        plt.xticks(rotation=45)
        plt.tight_layout()
        image_name = 'amount{}.png'.format(node_id)
        image_path = 'static/images/'
        fig.savefig(image_path+image_name)   # save the figure to file
        plt.close(fig)
        return render_template("results.html", **locals(), url ='/static/images/{}'.format(image_name),
            len = min(20,len(time_list)))


if __name__ == "__main__":
    app.run()