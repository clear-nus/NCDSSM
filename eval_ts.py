import os
import yaml
import torch
import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from ncdssm.evaluation import evaluate_simple_ts, evaluate_sporadic
from ncdssm.plotting import show_time_series_forecast, show_latents
from ncdssm.torch_utils import torch2numpy, prepend_time_zero
from experiments.setups import get_model, get_dataset


def main():
    matplotlib.use("Agg")

    # COMMAND-LINE ARGS
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt", required=True, type=str, help="Path to checkpoint file."
    )
    parser.add_argument(
        "--sporadic",
        action="store_true",
        help="Whether sporadic dataset (e.g., climate) is used.",
    )
    parser.add_argument("--seed", type=int, help="Random seed.")
    parser.add_argument(
        "--max_size",
        type=int,
        default=np.inf,
        help="Max number of time series to test.",
    )
    parser.add_argument("--device", type=str, help="Device to eval on")
    parser.add_argument(
        "--no_state_sampling",
        action="store_true",
        help="Use only the means of the predicted state distributions without sampling",
    )
    parser.add_argument(
        "--smooth",
        action="store_true",
        help="Use smoothing for imputation",
    )
    parser.add_argument(
        "--num_plots", type=int, default=0, help="The number of plots to save"
    )

    args, _ = parser.parse_known_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    if args.sporadic:
        evaluate_fn = evaluate_sporadic
    else:
        evaluate_fn = evaluate_simple_ts
    # CONFIG
    ckpt = torch.load(args.ckpt, map_location="cpu")
    config = ckpt["config"]
    config["device"] = args.device or config["device"]
    # DATA
    _, _, test_dataset = get_dataset(config)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config["test_batch_size"],
        collate_fn=test_dataset.collate_fn,
    )

    # MODEL
    device = torch.device(config["device"])
    model = get_model(config=config)
    model.load_state_dict(ckpt["model"], strict=True)
    step = ckpt["step"]
    model = model.to(device)
    num_params = 0
    for name, param in model.named_parameters():
        num_params += np.prod(param.size())
        print(name, param.size())
    print(f"Total Paramaters: {num_params.item()}")
    # print(model.A, model.C)

    # REEVALUATE
    log_dir = config["log_dir"]
    folder = os.path.join(log_dir, "test_plots", f"step{step}")
    os.makedirs(folder, exist_ok=True)

    results = {"config": config}

    if args.max_size > 0:
        metrics = evaluate_fn(
            test_loader,
            model,
            device,
            num_samples=config["num_forecast"],
            no_state_sampling=args.no_state_sampling,
            use_smooth=args.smooth,
        )

        results["test"] = metrics

    plot_count = 0
    plot_data = []
    while plot_count < args.num_plots:
        for test_batch in test_loader:
            past_target = test_batch["past_target"].to(device)
            B, T, D = past_target.shape
            mask = test_batch["past_mask"].to(device)
            future_target = test_batch["future_target"].to(device)
            past_times = test_batch["past_times"].to(device)
            future_times = test_batch["future_times"].to(device)

            if past_times[0] > 0:
                past_times, past_target, mask = prepend_time_zero(
                    past_times, past_target, mask
                )
            predict_result = model.forecast(
                past_target,
                mask,
                past_times.view(-1),
                future_times.view(-1),
                num_samples=config["num_forecast"],
                no_state_sampling=args.no_state_sampling,
                use_smooth=args.smooth,
            )
            reconstruction = predict_result["reconstruction"]
            forecast = predict_result["forecast"]
            full_times = torch.cat([past_times, future_times], 0)
            latent_variables = dict()
            if "z_reconstruction" in predict_result:
                full_z = torch.cat(
                    [predict_result["z_reconstruction"], predict_result["z_forecast"]],
                    dim=-2,
                )
                latent_variables["z"] = full_z
            if "alpha_reconstruction" in predict_result:
                full_alpha = torch.cat(
                    [
                        predict_result["alpha_reconstruction"],
                        predict_result["alpha_forecast"],
                    ],
                    dim=-2,
                )
                latent_variables["alpha"] = full_alpha

            for j in range(B):
                print(f"Plotting {plot_count + 1}/{config['num_plots']}")
                samples_dir = os.path.join(folder, f"series_{j}")
                os.makedirs(samples_dir, exist_ok=True)
                masked_past_target = past_target.clone()
                masked_past_target[mask == 0.0] = float("nan")
                plot_data_j = dict(
                    fig_size=(12, 5),
                    past_times=torch2numpy(past_times),
                    future_times=torch2numpy(future_times),
                    inputs=torch2numpy(torch.cat([past_target, future_target], 1))[j],
                    masked_inputs=torch2numpy(
                        torch.cat([masked_past_target, future_target], 1)
                    )[j],
                    reconstruction=torch2numpy(reconstruction)[:, j],
                    forecast=torch2numpy(forecast)[:, j],
                )
                plot_data.append(plot_data_j)
                fig = show_time_series_forecast(
                    **plot_data_j,
                    file_path=os.path.join(samples_dir, f"series_{plot_count}.png"),
                )
                plt.close(fig)
                if len(latent_variables) > 0:
                    latent_variables_j = {
                        k: torch2numpy(v[:, j]) for k, v in latent_variables.items()
                    }
                    for m in range(5):
                        latent_variables_jm = {
                            k: v[m] for k, v in latent_variables_j.items()
                        }
                        plot_path = os.path.join(samples_dir, f"lat_{m}.png")
                        show_latents(
                            (15, 8),
                            time=torch2numpy(full_times),
                            latents=latent_variables_jm,
                            fig_title="Latents",
                            file_path=plot_path,
                        )
                plot_count += 1

                if plot_count == args.num_plots:
                    break
            if plot_count == args.num_plots:
                break
    if args.max_size > 0:
        with open(os.path.join(log_dir, "metrics.yaml"), "w") as fp:
            yaml.dump(results, fp, default_flow_style=False)


if __name__ == "__main__":
    main()
