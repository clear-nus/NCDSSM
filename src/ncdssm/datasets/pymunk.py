import torch
import numpy as np

from ..utils import listofdict2dictoflist


class PymunkDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        file_path: str,
        missing_p=0.0,
        train=True,
        dt=0.1,
        ctx_len=20,
        pred_len=40,
    ):
        self._data, self._ground_truth = self._load_dataset(
            file_path, num_timesteps=ctx_len + pred_len
        )
        self.missing_p = missing_p
        self.train = train
        self.dt = dt
        self.ctx_len = ctx_len
        self.observed_mask = np.random.choice(
            [True, False],
            p=[1 - missing_p, missing_p],
            size=(self._data["y"].shape[0], ctx_len),
        )
        self.observed_mask[:, 0] = True

    def __len__(self):
        idx_batch = 0
        sizes = [val.shape[idx_batch] for val in self._data.values()]
        assert all(size == sizes[0] for size in sizes)
        size = sizes[0]
        return size

    def __getitem__(self, idx):
        target = self._data["y"][idx]
        past_target = target[: self.ctx_len]
        past_times = np.arange(self.ctx_len) * self.dt
        past_mask = self.observed_mask[idx].astype(np.float32)
        # past_target = past_target * past_mask[:, None]
        if not self.train:
            future_target = target[self.ctx_len :]
            future_times = np.arange(target.shape[0]) * self.dt
            future_times = future_times[self.ctx_len :]
        return dict(
            past_target=torch.as_tensor(past_target),
            future_target=None if self.train else torch.as_tensor(future_target),
            past_times=torch.as_tensor(past_times),
            future_times=None if self.train else torch.as_tensor(future_times),
            past_mask=torch.as_tensor(past_mask),
        )

    def collate_fn(self, list_of_samples):
        dict_of_samples = listofdict2dictoflist(list_of_samples)
        comb_past_target = torch.stack(dict_of_samples["past_target"])
        comb_past_times = dict_of_samples["past_times"][0]
        comb_past_mask = torch.stack(dict_of_samples["past_mask"])
        comb_future_target = None
        comb_future_times = None
        if dict_of_samples["future_target"][0] is not None:
            comb_future_target = torch.stack(dict_of_samples["future_target"])
            comb_future_times = dict_of_samples["future_times"][0]
        return dict(
            past_target=comb_past_target,
            past_times=comb_past_times,
            past_mask=comb_past_mask,
            future_target=comb_future_target,
            future_times=comb_future_times,
        )

    def _load_dataset(self, file_path, num_timesteps=100):
        npzfile = np.load(file_path)
        images = npzfile["images"].astype(np.float32)
        # The datasets in KVAE are binarized images
        images = (images > 0).astype(np.float32)
        assert images.ndim == 4
        images = images.reshape(
            images.shape[0], images.shape[1], images.shape[2] * images.shape[3]
        )
        data = {"y": images[:, :num_timesteps]}

        if "state" in npzfile:  # all except Pong have state.
            position = npzfile["state"].astype(np.float32)[:, :, :2]
            velocity = npzfile["state"].astype(np.float32)[:, :, 2:]
            ground_truth_state = {"position": position, "velocity": velocity}
        else:
            ground_truth_state = None
        return data, ground_truth_state
