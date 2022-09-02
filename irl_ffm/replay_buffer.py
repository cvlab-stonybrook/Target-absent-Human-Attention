from collections import deque
import numpy as np
import random
import torch


class Memory(object):
    def __init__(self, memory_size: int, seed: int = 0) -> None:
        random.seed(seed)
        self.memory_size = memory_size
        self.buffer = deque(maxlen=self.memory_size)

    def add(self, experience) -> None:
        self.buffer.append(experience)

    def add_batch(self, experiences):
        states, new_states, act, curr_status, aux = experiences
        for i in range(act.size(0)):
            state_entry = [x[i].cpu() for x in states]
            next_state_entry = [x[i].cpu() for x in new_states]
            self.add((state_entry, next_state_entry, act[i].cpu(),
                      curr_status[i].cpu(), aux[i]))

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size: int, continuous: bool = False):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if continuous:
            rand = random.randint(0, len(self.buffer) - batch_size)
            return [self.buffer[i] for i in range(rand, rand + batch_size)]
        else:
            indexes = np.random.choice(np.arange(len(self.buffer)),
                                       size=batch_size,
                                       replace=False)
            return [self.buffer[i] for i in indexes]

    def clear(self):
        self.buffer.clear()

    def save(self, path):
        b = np.asarray(self.buffer)
        print(b.shape)
        np.save(path, b)

    def load(self, path, num_trajs, sample_freq, seed):
        # If path has no extension add npy
        if not path.endswith("pkl"):
            path += '.npy'
        pass

    def get_samples(self, batch_size):
        batch = self.sample(batch_size, False)

        batch_state, batch_next_state, batch_action, batch_done, batch_aux = zip(
            *batch)

        n_state_comps = len(batch_state[0])
        batch_state = [
            torch.stack([x[i] for x in batch_state])
            for i in range(n_state_comps)
        ]
        batch_next_state = [
            torch.stack([x[i] for x in batch_next_state])
            for i in range(n_state_comps)
        ]
        batch_action = torch.stack(batch_action)
        batch_aux = torch.stack(batch_aux)
        batch_done = torch.stack(batch_done) > 1

        return batch_state, batch_next_state, batch_action, batch_done, batch_aux
