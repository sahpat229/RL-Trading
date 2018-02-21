from collections import deque
import random
import numpy as np

class ReplayBufferException(Exception):
    pass

class ReplayBuffer():
    def __init__(self, buffer_size=200):
        self.buffer = deque()
        self.buffer_size = buffer_size

    def store(self, old_state, action, reward, new_state):
        experience = [old_state, action, reward, new_state]
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(experience)
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def sample(self, batch_size, recurrent=False, time_steps=None):
        """
        Sample the replay buffer.  Should only be sampled when the replay buffer is full.

        If recurrent=False, then we return [batch_size, num_items_stored] samples
        If recurrent=True, then we return [batch_size, time_steps, num_items_stored] samples
        """
        if not self.ready(batch_size=batch_size):
            raise ReplayBufferException('Replay Buffer should only be sampled when having batch_size amount of samples.') 

        if not recurrent:
            return random.sample(self.buffer, batch_size)
        else:
            if time_steps is None:
                raise ReplayBufferException('Replay Buffer needs have time_steps set if recurrent=True')

            start_inds = np.random.randint(low=0,
                                           high=self.buffer_size - time_steps,
                                           size=batch_size)
            batch_samples = [] # [batch_size, time_samples, num_items_stored]
            for start_ind in start_inds:
                time_samples = self.buffer[start_ind:start_ind+time_steps] # [time_samples, num_items_stored]
                batch_samples.append(time_samples)
                
            return batch_samples

    def clear(self):
        self.buffer.clear()

    def ready(self, batch_size):
        return len(self.buffer) >= batch_size

    @property
    def full(self):
        return len(self.buffer) == self.buffer_size
    