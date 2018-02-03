from data.datacontainer import DataContainer, PortfolioVectorMemory
from policynetwork import PolicyNetwork

import tensorflow as tf 

hdf_file_name = './data/hdfs/poloniex_30m.hf'
csv_file_name = './data/csvs/output.csv'
dc = DataContainer(hdf_file_name=hdf_file_name,
                   key='train')
# dc = DataContainer(csv_file_name=csv_file_name)
pvm = PortfolioVectorMemory(num_periods=dc.num_periods,
                            num_assets=dc.num_assets)

sess = tf.Session()
pn = PolicyNetwork(sess=sess,
                   num_assets=dc.num_assets,
                   history_length=50,
                   batch_size=50,
                   commission_rate=0,
                   mode='EIEE',
                   num_epochs=100,
                   use_batch_norm=False,
                   data_container=dc,
                   PVM=pvm)
pn.train()
pn.infer(test_batch_size=150)   