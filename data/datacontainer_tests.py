from datacontainer import DataContainer

file_name = './hdfs/poloniex_30m.hf'

dc = DataContainer(hdf_file_name=file_name,
                   key='train')
dc_gen = dc.yield_data(batch_size=5)
dic = next(dc_gen)
print(dic['batch_current_prices'].shape)
print(dic['batch_future_prices'].shape)