import csv
import datetime
import gdax
import numpy as np
import pandas as pd
import time

class DataGatherer():
    def __init__(self, granularity=900, output_file_name='./csvs/output.csv'):
        assert granularity in set([60, 300, 900, 3600, 21600, 86400])
        self.granularity = granularity
        self.output_file_name = output_file_name

    def gather_data(self, num_periods):
        assert num_periods <= 350

        output_file = open(self.output_file_name, 'w+')
        output_csv = csv.writer(output_file)
        output_csv.writerow(['time', 'low', 'high', 'open', 'close', 'volume'])

        public_client = gdax.PublicClient()
        now = datetime.datetime.now()
        start = datetime.datetime(year=2014,
                                  month=1,
                                  day=1)
        time_delta = datetime.timedelta(minutes=15)

        while start+num_periods*time_delta < now:
            items = public_client.get_product_historic_rates('BTC-USD',
                                                             granularity=900,
                                                             start=start,
                                                             end=start+num_periods*time_delta)
            output_csv.writerows(reversed(list(items)))
            start = start+num_periods*time_delta
            time.sleep(1)

dc = DataGatherer()
dc.gather_data(350)