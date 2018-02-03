from data.datacontainer import BitcoinTestContainer
from tradingstatemodel import TradingStateModel, QApproximator, ReplayBuffer

csv_file_name = './data/csvs/output.csv'

max_coins = 4

btc = BitcoinTestContainer(csv_file_name=csv_file_name)
rpb = ReplayBuffer()
q_approximator = QApproximator(num_features=btc.num_features+max_coins+1,
                               num_actions=max_coins+1)
tsm = TradingStateModel(bitcoin_container=btc,
                        model=q_approximator,
                        episode_length=2000,
                        gamma=0.95,
                        starting_coins=0,
                        max_coins=max_coins,
                        epochs=100,
                        replay_buffer=rpb,
                        batch_size=10)
tsm.train()
