import torch
import numpy as np

from torch.utils.data import Dataset

from ..utils import listofdict2dictoflist


class BouncingBallDataset(Dataset):
    def __init__(
        self,
        path: str = "./data/bouncing_ball.npz",
        dt: float = 0.1,
        ctx_len: int = 200,
        pred_len: int = 100,
        missing_p: float = 0.0,
        num_always_obs: float = 0,
    ):
        self._data = np.load(path)["target"]
        self.dt = dt
        self.ctx_len = ctx_len
        self.pred_len = pred_len
        self.missing_p = missing_p
        self.num_always_obs = num_always_obs
        self._set_mask()

    def _set_mask(self):
        self.observed_mask = np.random.choice(
            [True, False],
            p=[1 - self.missing_p, self.missing_p],
            size=(self._data.shape[0], self.ctx_len),
        )
        self.observed_mask[:, : self.num_always_obs] = True

    def __len__(self):
        return self._data.shape[0]

    def __getitem__(self, idx):
        target = self._data[idx]
        past_target = target[: self.ctx_len]
        past_times = np.arange(self.ctx_len) * self.dt
        past_mask = self.observed_mask[idx]
        # past_target = past_target * past_mask[:, None]
        future_target = target[self.ctx_len :]
        future_times = np.arange(target.shape[0]) * self.dt
        future_times = future_times[self.ctx_len :]
        return dict(
            past_target=torch.as_tensor(past_target.astype(np.float32)),
            future_target=torch.as_tensor(future_target.astype(np.float32)),
            past_times=torch.as_tensor(past_times),
            future_times=torch.as_tensor(future_times),
            past_mask=torch.as_tensor(past_mask.astype(np.float32)),
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


class DampedPendulumDataset(BouncingBallDataset):
    def __init__(
        self,
        path: str = "./data/damped_pendulum/train.npz",
        dt: float = 0.1,
        ctx_len: int = 50,
        pred_len: int = 100,
        missing_p: float = 0,
        num_always_obs: float = 0,
    ):
        self._data = np.load(path)["obs"]
        self.dt = dt
        self.ctx_len = ctx_len
        self.pred_len = pred_len
        self.missing_p = missing_p
        self.num_always_obs = num_always_obs
        self._set_mask()
