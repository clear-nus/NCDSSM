from pathlib import Path
import numpy as np

from ncdssm.type import Dict

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data"


def get_model(config):
    if config["dataset"] == "mocap" or config["dataset"] == "mocap2":
        from .mocap import build_model
    elif config["dataset"] == "bouncing_ball":
        from .bouncing_ball import build_model
    elif config["dataset"] == "damped_pendulum":
        from .pendulum import build_model
    elif config["dataset"] in {"box", "pong"}:
        from .pymunk import build_model
    elif config["dataset"] == "climate":
        from .climate import build_model
    else:
        raise ValueError(f"Unknown dataset {config['dataset']}")
    return build_model(config=config)


def get_dataset(config: Dict):
    from ncdssm.datasets import (
        PymunkDataset,
        MocapDataset,
        BouncingBallDataset,
        DampedPendulumDataset,
        ClimateDataset,
    )

    dataset_name = config["dataset"]
    if dataset_name == "mocap":
        train_dataset = MocapDataset(
            file_path=DATA_ROOT / "mocap/mocap35.mat",
            mode="train",
            ctx_len=300,
            pred_len=0,
            missing_p=config["train_missing_p"],
        )
        val_dataset = MocapDataset(
            file_path=DATA_ROOT / "mocap/mocap35.mat",
            mode="val",
            ctx_len=3,
            pred_len=297,
        )
        test_dataset = MocapDataset(
            file_path=DATA_ROOT / "mocap/mocap35.mat",
            mode="test",
            ctx_len=3,
            pred_len=297,
        )
    elif dataset_name == "mocap2":
        train_dataset = MocapDataset(
            file_path=DATA_ROOT / "mocap/mocap35.mat",
            mode="train",
            ctx_len=200,
            pred_len=100,
            missing_p=config["train_missing_p"],
        )
        val_dataset = MocapDataset(
            file_path=DATA_ROOT / "mocap/mocap35.mat",
            mode="val",
            ctx_len=100,
            pred_len=200,
        )
        test_dataset = MocapDataset(
            file_path=DATA_ROOT / "mocap/mocap35.mat",
            mode="test",
            ctx_len=100,
            pred_len=200,
        )
    elif dataset_name == "bouncing_ball":
        train_dataset = BouncingBallDataset(
            path=DATA_ROOT / "bouncing_ball/rv_train.npz",
            ctx_len=100,
            pred_len=200,
            missing_p=config["train_missing_p"],
        )
        val_dataset = BouncingBallDataset(
            path=DATA_ROOT / "bouncing_ball/rv_val.npz",
            ctx_len=100,
            pred_len=200,
            missing_p=config["train_missing_p"],
        )
        test_dataset = BouncingBallDataset(
            path=DATA_ROOT / "bouncing_ball/rv_test.npz",
            ctx_len=100,
            pred_len=200,
            missing_p=config["train_missing_p"],
        )
    elif dataset_name == "damped_pendulum":
        train_dataset = DampedPendulumDataset(
            path=DATA_ROOT / "damped_pendulum/train.npz",
            ctx_len=50,
            pred_len=100,
            missing_p=config["train_missing_p"],
        )
        val_dataset = DampedPendulumDataset(
            path=DATA_ROOT / "damped_pendulum/val.npz",
            ctx_len=50,
            pred_len=100,
            missing_p=config["train_missing_p"],
        )
        test_dataset = DampedPendulumDataset(
            path=DATA_ROOT / "damped_pendulum/test.npz",
            ctx_len=50,
            pred_len=100,
            missing_p=config["train_missing_p"],
        )
    elif dataset_name == "climate":
        csv_path = DATA_ROOT / "climate/climate-data-preproc.csv"
        fold_idx = config["data_fold"]
        train_idx = np.load(DATA_ROOT / f"climate/fold_idx_{fold_idx}/train_idx.npy")
        val_idx = np.load(DATA_ROOT / f"climate/fold_idx_{fold_idx}/val_idx.npy")
        test_idx = np.load(DATA_ROOT / f"climate/fold_idx_{fold_idx}/test_idx.npy")
        train_dataset = ClimateDataset(
            csv_path=csv_path,
            train=True,
            ids=train_idx,
        )
        val_dataset = ClimateDataset(
            csv_path=csv_path,
            train=False,
            ids=val_idx,
            val_options=dict(T_val=150, forecast_steps=3),
        )
        test_dataset = ClimateDataset(
            csv_path=csv_path,
            train=False,
            ids=test_idx,
            val_options=dict(T_val=150, forecast_steps=3),
        )
    elif dataset_name in {"pong", "box"}:
        data_root = DATA_ROOT / f"pymunk/{dataset_name}"
        train_dataset = PymunkDataset(
            file_path=data_root / "train.npz",
            missing_p=config["train_missing_p"],
            train=True,
        )
        val_dataset = PymunkDataset(
            file_path=data_root / "val.npz",
            missing_p=config["train_missing_p"],
            train=False,
        )
        test_dataset = PymunkDataset(
            file_path=data_root / "test.npz",
            missing_p=config["train_missing_p"],
            train=False,
        )
    else:
        raise ValueError(f"Unknown dataset {dataset_name}!")
    return train_dataset, val_dataset, test_dataset
