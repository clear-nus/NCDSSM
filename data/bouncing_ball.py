import os
import numpy as np
import argparse
from tqdm.auto import tqdm
from pathlib import Path


def generate_sequence(low=-1, high=1.0, vel=None, num_steps=300):
    y = np.random.uniform(low=low, high=high)
    if vel is None:
        vel = np.random.uniform(low=0.05, high=0.5) * np.random.choice([-1, 1])
    noise_scale = 0.05
    points = [y + noise_scale * np.random.randn(1)]
    step_size = 0.1
    for i in range(num_steps - 1):
        y = y + vel * step_size
        points.append(y + noise_scale * np.random.randn(1))
        if y <= low or y >= high:
            vel = -vel
    return np.stack(points)


def generate_sequences(num_samples, vels=None, num_steps=300):
    all_target = []
    for _ in tqdm(range(num_samples)):
        chosen_v = np.random.choice(vels) if vels is not None else None
        y = generate_sequence(vel=chosen_v, num_steps=num_steps)
        all_target.append(y)
    all_target = np.stack(all_target)
    return all_target


def generate_dataset(
    seed=42,
    vels=None,
    dataset_path=None,
    n_train=5000,
    n_val=500,
    n_test=500,
    n_timesteps=300,
    file_prefix="",
):
    if dataset_path is None:
        dataset_path = "./bouncing_ball/"
    os.makedirs(dataset_path, exist_ok=True)

    np.random.seed(seed=seed)

    obs_train = generate_sequences(n_train, vels=vels, num_steps=n_timesteps)
    obs_val = generate_sequences(n_val, vels=vels, num_steps=n_timesteps)
    obs_test = generate_sequences(n_test, vels=vels, num_steps=n_timesteps)

    np.savez(os.path.join(dataset_path, f"{file_prefix}train.npz"), target=obs_train)
    np.savez(os.path.join(dataset_path, f"{file_prefix}val.npz"), target=obs_val)
    np.savez(os.path.join(dataset_path, f"{file_prefix}test.npz"), target=obs_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_vels",
        type=int,
        default=0,
        help="Number of fixed velocities, 0 indicates ranodm velocity for every sample",
        choices=[0, 1, 2, 5],
    )
    args = parser.parse_args()

    data_path = str(Path(__file__).resolve().parent / "bouncing_ball")
    print(f"Saving dataset to: {data_path}.")
    vels = None
    if args.num_vels == 0:
        vels = None
        print("Generating dataset with random velocties...")
        generate_dataset(file_prefix="rv_", vels=vels, dataset_path=data_path)
    elif args.num_vels == 1:
        vels = [0.2]
        print("Generating dataset with 1 veloctity...")
        generate_dataset(file_prefix="1fv_", vels=vels, dataset_path=data_path)
    elif args.num_vels == 2:
        vels = [0.2, 0.4]
        print("Generating dataset with 2 velocties...")
        generate_dataset(file_prefix="2fv_", vels=vels, dataset_path=data_path)
    elif args.num_vels == 5:
        vels = [0.1, 0.2, 0.3, 0.4, 0.5]
        print("Generating dataset with 5 velocties...")
        generate_dataset(file_prefix="5fv_", vels=vels, dataset_path=data_path)
