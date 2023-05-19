import torch
import numpy as np
import pandas as pd

from ..type import Dict, NumpyArray
from ..utils import listofdict2dictoflist


class ClimateDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        csv_path: str,
        time_scale: float = 1.0,
        train: bool = True,
        ids: NumpyArray = None,
        val_options: Dict = None,
    ) -> None:
        super().__init__()
        self.train = train
        self.df = pd.read_csv(csv_path)
        assert self.df.columns[0] == "ID"

        if not train:
            assert val_options is not None
            ids_before_tval = self.df.loc[
                self.df["Time"] <= val_options["T_val"], "ID"
            ].unique()
            ids_after_tval = self.df.loc[
                self.df["Time"] > val_options["T_val"], "ID"
            ].unique()
            filtered_ids = np.intersect1d(ids_before_tval, ids_after_tval)
            self.df = self.df.loc[self.df["ID"].isin(filtered_ids)]

        if ids is not None:
            self.df = self.df.loc[self.df["ID"].isin(ids)].copy()
            map_dict = dict(
                zip(self.df["ID"].unique(), np.arange(self.df["ID"].nunique()))
            )
            self.df["ID"] = self.df["ID"].map(map_dict)

        self.df.Time = self.df.Time * time_scale

        if not train:
            self.df_before_tval = self.df.loc[
                self.df["Time"] <= val_options["T_val"]
            ].copy()
            self.df_after_tval = (
                self.df.loc[self.df["Time"] > val_options["T_val"]]
                .sort_values("Time")
                .copy()
            )
            self.df_after_tval = (
                self.df_after_tval.groupby("ID")
                .head(val_options["forecast_steps"])
                .copy()
            )

            self.df = self.df_before_tval

            self.df_after_tval = self.df_after_tval.astype(np.float32)
            self.df_after_tval.ID = self.df_after_tval.ID.astype(int)
            self.df_after_tval.sort_values("Time", inplace=True)

        self.df = self.df.astype(np.float32)
        self.df.ID = self.df.ID.astype(int)
        self.size = self.df["ID"].nunique()
        # self.df.set_index("ID", inplace=True)
        self.df.sort_values("Time", inplace=True)
        self.data_columns = list(
            filter(lambda c: c.startswith("Value"), self.df.columns)
        )
        self.mask_columns = list(
            filter(lambda c: c.startswith("Mask"), self.df.columns)
        )

        self.data = []

        for id, sub_df in self.df.groupby("ID"):
            sub_df = sub_df.reset_index(drop=True).sort_values("Time")
            item = {
                "Time": sub_df["Time"].to_numpy(),
                "Value": sub_df[self.data_columns].to_numpy(),
                "Mask": sub_df[self.mask_columns].to_numpy(),
            }
            if not self.train:
                val_sub_df = self.df_after_tval.loc[self.df_after_tval["ID"] == id]
                item["Future_Time"] = val_sub_df["Time"].to_numpy()
                item["Future_Value"] = val_sub_df[self.data_columns].to_numpy()
                item["Future_Mask"] = val_sub_df[self.mask_columns].to_numpy()
            self.data.append(item)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        subset = self.data[idx]
        past_target = subset["Value"]
        past_mask = subset["Mask"]
        past_times = subset["Time"]
        if not self.train:
            future_target = subset["Future_Value"]
            future_mask = subset["Future_Mask"]
            future_times = subset["Future_Time"]
        return dict(
            past_target=torch.as_tensor(past_target),
            past_mask=torch.as_tensor(past_mask),
            past_times=torch.as_tensor(past_times),
            future_target=None if self.train else torch.as_tensor(future_target),
            future_mask=None if self.train else torch.as_tensor(future_mask),
            future_times=None if self.train else torch.as_tensor(future_times),
        )

    def collate_fn(self, list_of_samples):
        dict_of_samples = listofdict2dictoflist(list_of_samples)
        batch_size = len(list_of_samples)
        target_dim = dict_of_samples["past_target"][0].shape[-1]
        # Collate past
        comb_past_times, past_inverse_indices = torch.unique(
            torch.cat(dict_of_samples["past_times"]), sorted=True, return_inverse=True
        )
        comb_past_target = torch.zeros(
            [batch_size, comb_past_times.shape[0], target_dim]
        )
        comb_past_mask = torch.zeros_like(comb_past_target)
        past_offset = 0
        for i, (tgt, mask, time) in enumerate(
            zip(
                dict_of_samples["past_target"],
                dict_of_samples["past_mask"],
                dict_of_samples["past_times"],
            )
        ):
            past_indices = past_inverse_indices[
                past_offset : past_offset + time.shape[0]
            ]
            past_offset += time.shape[0]
            comb_past_target[i, past_indices] = tgt
            comb_past_mask[i, past_indices] = mask
        # Collate future
        comb_future_target = None
        comb_future_times = None
        comb_future_mask = None
        if dict_of_samples["future_target"][0] is not None:
            comb_future_times, future_inverse_indices = torch.unique(
                torch.cat(dict_of_samples["future_times"]),
                sorted=True,
                return_inverse=True,
            )
            comb_future_target = torch.zeros(
                [batch_size, comb_future_times.shape[0], target_dim]
            )
            comb_future_mask = torch.zeros_like(comb_future_target)
            future_offset = 0
            for i, (tgt, mask, time) in enumerate(
                zip(
                    dict_of_samples["future_target"],
                    dict_of_samples["future_mask"],
                    dict_of_samples["future_times"],
                )
            ):
                future_indices = future_inverse_indices[
                    future_offset : future_offset + time.shape[0]
                ]
                future_offset += time.shape[0]
                comb_future_target[i, future_indices] = tgt
                comb_future_mask[i, future_indices] = mask
        return dict(
            past_target=comb_past_target,
            past_times=comb_past_times,
            past_mask=comb_past_mask,
            future_target=comb_future_target,
            future_times=comb_future_times,
            future_mask=comb_future_mask,
        )
