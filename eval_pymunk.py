import os
import yaml
import torch
import argparse
import matplotlib
import numpy as np


from ncdssm.torch_utils import torch2numpy
from ncdssm.plotting import (
    show_latents,
    show_pymunk_forecast,
    show_wasserstein_distance,
)
from ncdssm.evaluation import evaluate_pymunk_dataset
from experiments.setups import get_model, get_dataset


def main():
    matplotlib.use("Agg")

    # COMMAND-LINE ARGS
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt", required=True, type=str, help="Path to checkpoint file."
    )

    parser.add_argument("--seed", type=int, help="Random seed.")

    parser.add_argument(
        "--wass",
        action="store_true",
        help="Whether to compute the Wasserstein distance.",
    )
    parser.add_argument("--device", type=str, help="Device to eval on")
    parser.add_argument(
        "--max_size",
        type=int,
        default=np.inf,
        help="Maximum number of time series to evaluate on. Only for debugging.",
    )
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
    # CONFIG
    ckpt = torch.load(args.ckpt, map_location="cpu")
    config = ckpt["config"]
    config["device"] = args.device or config["device"]
    # DATA
    train_dataset, _, test_dataset = get_dataset(config)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config["test_batch_size"],
        collate_fn=train_dataset.collate_fn,
    )

    # MODEL
    device = torch.device(config["device"])
    model = get_model(config=config)
    model.load_state_dict(ckpt["model"])
    step = ckpt["step"]
    model = model.to(device)
    num_params = 0
    for name, param in model.named_parameters():
        num_params += np.prod(param.size())
        print(name, param.size())
    print(f"Total Paramaters: {num_params.item()}")

    # REEVALUATE
    log_dir = config["log_dir"]
    folder = os.path.join(log_dir, "test_plots", f"step{step}")
    os.makedirs(folder, exist_ok=True)

    results = {"config": config}
    save_dict = dict(predictions=[])

    if args.wass:
        (
            wt_mean,
            wt_conf_interval,
            future_w_mean,
            future_w_conf_interval,
        ) = evaluate_pymunk_dataset(
            test_loader,
            model,
            device=device,
            num_samples=config["num_forecast"],
            max_size=args.max_size,
            no_state_sampling=args.no_state_sampling,
            use_smooth=args.smooth,
        ).values()
        save_dict["wass_dist"] = wt_mean

        show_wasserstein_distance(
            (15, 2),
            wt_mean,
            conf_intervals=wt_conf_interval,
            fig_title="Wasserstein Distance",
            file_path=os.path.join(folder, "wass.png"),
        )
        print(
            f"Forecast W: {future_w_mean.item():.3f} "
            f"+/- {future_w_conf_interval.item():.3f}"
        )
        results["future_w_mean"] = future_w_mean.item()
        results["future_w_conf_interval"] = future_w_conf_interval.item()

    plot_count = 0
    while plot_count < args.num_plots:
        for test_batch in test_loader:
            past_target = test_batch["past_target"].to(device)
            B, T, _ = past_target.shape
            mask = test_batch["past_mask"].to(device)
            future_target = test_batch["future_target"].to(device)
            past_times = test_batch["past_times"].to(device)
            future_times = test_batch["future_times"].to(device)
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
            full_prediction = torch.cat([reconstruction, forecast], dim=-2)
            full_target = torch.cat([past_target, future_target], dim=-2)
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
            if "aux_reconstruction" in predict_result:
                full_aux = torch.cat(
                    [
                        predict_result["aux_reconstruction"],
                        predict_result["aux_forecast"],
                    ],
                    dim=-2,
                )
                latent_variables["aux"] = full_aux
            for j in range(B):
                full_prediction_j = full_prediction[:, j].view(
                    full_prediction.shape[0],
                    full_prediction.shape[-2],
                    1,
                    config["img_size"],
                    config["img_size"],
                )
                full_target_j = full_target[j].view(
                    full_target.shape[1], 1, config["img_size"], config["img_size"]
                )
                # Plot first five samples
                samples_dir = os.path.join(folder, f"series_{j}")
                os.makedirs(samples_dir, exist_ok=True)
                full_target_j[:T][mask[j] == 0.0] = 0.0
                save_dict["predictions"].append(
                    dict(
                        target=torch2numpy(full_target_j),
                        pred=torch2numpy(full_prediction_j[0]),
                    )
                )
                show_pymunk_forecast(
                    torch2numpy(full_target_j),
                    torch2numpy(full_prediction_j[:5]),
                    os.path.join(samples_dir, "prediction.png"),
                )
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

    with open(os.path.join(log_dir, "metrics.yaml"), "w") as fp:
        yaml.dump(results, fp, default_flow_style=False)

    np.savez(os.path.join(log_dir, "plot_data.npz"), **save_dict)


if __name__ == "__main__":
    main()
