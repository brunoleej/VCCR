import numpy as np
import torch as T


class ReplayBuffer:
    def __init__(self, obs_dim, action_dim, buffer_size, device="cpu"):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._obses = T.zeros((buffer_size, obs_dim), dtype=T.float32, device=device)
        self._actions = T.zeros((buffer_size, action_dim), dtype=T.float32, device=device)
        self._rewards = T.zeros((buffer_size, 1), dtype=T.float32, device=device)
        self._next_obses = T.zeros((buffer_size, obs_dim), dtype=T.float32, device=device)
        self._dones = T.zeros((buffer_size, 1), dtype=T.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> T.Tensor:
        return T.tensor(data, dtype=T.float32, device=self._device)

    def load_dataset(self, dataset):
        n_transitions = dataset["observations"].shape[0]
        self._obses[:n_transitions] = self._to_tensor(dataset["observations"])
        self._actions[:n_transitions] = self._to_tensor(dataset["actions"])
        self._rewards[:n_transitions] = self._to_tensor(dataset["rewards"][..., None])
        self._next_obses[:n_transitions] = self._to_tensor(dataset["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(dataset["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)
        print(f"Size of Dataset: {n_transitions}")

    def add_batch(self, observations, next_observations, actions, rewards, terminals):
        batch_size = len(terminals)
        if self._pointer + batch_size > self._buffer_size:
            begin = self._pointer
            end = self._buffer_size
            first_add_size = end - begin
            self._obses[begin:end] = self._to_tensor(observations[:first_add_size].copy())
            self._next_obses[begin:end] = self._to_tensor(next_observations[:first_add_size].copy())
            self._actions[begin:end] = self._to_tensor(actions[:first_add_size].copy())
            self._rewards[begin:end] = self._to_tensor(rewards[:first_add_size].copy())
            self._dones[begin:end] = self._to_tensor(terminals[:first_add_size].copy())

            begin = 0
            end = batch_size - first_add_size
            self._obses[begin:end] = self._to_tensor(observations[first_add_size:].copy())
            self._next_obses[begin:end] = self._to_tensor(next_observations[first_add_size:].copy())
            self._actions[begin:end] = self._to_tensor(actions[first_add_size:].copy())
            self._rewards[begin:end] = self._to_tensor(rewards[first_add_size:].copy())
            self._dones[begin:end] = self._to_tensor(terminals[first_add_size:].copy())

            self._pointer = end
            self._size = min(self._size + batch_size, self._buffer_size)

        else:
            begin = self._pointer
            end = self._pointer + batch_size
            self._obses[begin:end] = self._to_tensor(observations.copy())
            self._next_obses[begin:end] = self._to_tensor(next_observations.copy())
            self._actions[begin:end] = self._to_tensor(actions.copy())
            self._rewards[begin:end] = self._to_tensor(rewards.copy())
            self._dones[begin:end] = self._to_tensor(terminals.copy())

            self._pointer = end
            self._size = min(self._size + batch_size, self._buffer_size)

    def sample(self, batch_size):
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = self._obses[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_obses[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]

    def sample_all(self, batch_size):
        num_batches = int((self._pointer+1)/batch_size)
        indices = np.arange(self._pointer)
        np.random.shuffle(indices)
        for batch_id in range(num_batches):
            batch_start = batch_id * batch_size
            batch_end = min(self._pointer, (batch_id + 1) * batch_size)

            states = self._obses[batch_start:batch_end]
            actions = self._actions[batch_start:batch_end]
            rewards = self._rewards[batch_start:batch_end]
            next_states = self._next_obses[batch_start:batch_end]
            dones = self._dones[batch_start:batch_end]
            yield [states, actions, rewards, next_states, dones]

    def normalize_states(self, eps=1e-3, mean=None, std=None):
        mean = self._obses.mean(0, keepdims=True)
        std = self._obses.std(0, keepdims=True) + eps
        self._obses = (self._obses - mean) / std
        self._next_obses = (self._next_obses - mean) / std
        return mean.cpu().data.numpy().flatten(), std.cpu().data.numpy().flatten()
