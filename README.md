## DLForecast
Bitcoin transaction forecast model.This model can be used to forecast the if two account will have a transaction in the future.

Data visualization is also provided to explain statistical characterization of bitcoin transaction data.

Built for explaining results of the paper :  
"Bitcoin Transaction Forecasting with Deep Network Representation Learning", Wenqi Wei(Georgia Tech), Qi Zhang(IBM Research), Ling Liu(Georgia Tech). under submission


#to run the transaction prediction, first download data from https://senseable2015-6.mit.edu/bitcoin/

#to generate embedding, go to online-node2vec(a three page paper) to download their code

#then, go to evalne to download their code and set it up

#When everything set up, you can run prediction models.






# Requires:  
Cassandra database setup - Table formats are present in cqls/create_tables. Data can be written using write_to_db.py. The sample files required are stored in data/ folder.

main.py - Starts a Flask web app, to allow searching for node data

graph_visuals.ipynb - Code for generating temporal graph and k-step directed graph representation.


# Citation
If you use our code, please cite:
```
W. Wei, Q. Zhang and L. Liu, "Bitcoin Transaction Forecasting with Deep Network Representation Learning" in IEEE Transactions on Emerging Topics in Computing, vol. , no. 01, pp. 1-1, 5555. doi: 10.1109/TETC.2020.3010464
```



# Acknowledgement
We thank the effort and contribution made by Shubhi Agarwal, a master student in computer science from Georgia Tech for the project.

