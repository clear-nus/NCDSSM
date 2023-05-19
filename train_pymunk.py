import os
import copy
import yaml
import torch
import argparse
import matplotlib
import numpy as np

from tensorboardX import SummaryWriter

from ncdssm.torch_utils import grad_norm, torch2numpy
from ncdssm.plotting import show_pymunk_forecast, show_wasserstein_distance
from ncdssm.evaluation import evaluate_pymunk_dataset
import experiments.utils
from experiments.setups import get_model, get_dataset


def train_step(train_batch, model, optimizer, reg_scheduler, step, device, config):
    batch_target = train_batch["past_target"]
    batch_times = train_batch["past_times"]
    batch_mask = train_batch["past_mask"]
    batch_target = batch_target.to(device)
    batch_times = batch_times.to(device)
    batch_mask = batch_mask.to(device)
    optimizer.zero_grad()
    out = model(
        batch_target,
        batch_mask,
        batch_times,
        num_samples=config.get("num_samples", 1),
    )
    cond_ll = out["likelihood"]
    reg = out["regularizer"]
    loss = -(cond_ll + reg_scheduler.val * reg).mean(0)
    loss.backward()

    if step <= config.get("ssm_params_warmup_steps", 0):
        ctkf_lr = optimizer.param_groups[0]["lr"]
        optimizer.param_groups[0]["lr"] = 0
    total_grad_norm = grad_norm(model.parameters())
    if float(config["max_grad_norm"]) != float("inf"):
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=config["max_grad_norm"]
        )
    optimizer.step()
    if step <= config.get("ssm_params_warmup_steps", 0):
        optimizer.param_groups[0]["lr"] = ctkf_lr
    print(
        f"Step {step}: Loss={loss.item():.4f}, Grad Norm: {total_grad_norm.item():.2f},"
        f" Reg-Coeff: {reg_scheduler.val:.2f}"
    )
    return dict(
        loss=loss.item(), cond_ll=cond_ll.mean(0).item(), reg=reg.mean(0).item()
    )


def main():
    matplotlib.use("Agg")
    # SET SEED
    # seed = 111
    # print(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # random.seed(seed)

    # COMMAND-LINE ARGS
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=str, help="Path to config file.")
    group.add_argument("--ckpt", type=str, help="Path to checkpoint file.")

    args, _ = parser.parse_known_args()
    # CONFIG
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location="cpu")
        config = ckpt["config"]
    else:
        config = experiments.utils.get_config_and_setup_dirs(args.config)
        parser = experiments.utils.add_config_to_argparser(config=config, parser=parser)
        args = parser.parse_args()
        # Update config from command line args, if any.
        updated_config_dict = vars(args)
        for k in config.keys() & updated_config_dict.keys():
            o_v = config[k]
            u_v = updated_config_dict[k]
            if u_v != o_v:
                print(f"{k}: {o_v} -> {u_v}")
        config.update(updated_config_dict)
    # DATA
    train_dataset, val_dataset, _ = get_dataset(config)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["train_batch_size"],
        num_workers=4,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config["test_batch_size"],
        collate_fn=train_dataset.collate_fn,
    )
    train_gen = iter(train_loader)
    # test_gen = iter(test_loader)

    # MODEL
    device = torch.device(config["device"])
    model = get_model(config=config)

    kf_param_names = {
        name for name, _ in model.named_parameters() if "base_ssm" in name
    }
    kf_params = [
        param for name, param in model.named_parameters() if name in kf_param_names
    ]
    non_kf_params = [
        param for name, param in model.named_parameters() if name not in kf_param_names
    ]
    print(kf_param_names)
    optim = torch.optim.Adam(
        params=[
            {"params": kf_params},
            {"params": non_kf_params},
        ],
        lr=config["learning_rate"],
    )
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optim, gamma=config["lr_decay_rate"]
    )
    reg_scheduler = experiments.utils.LinearScheduler(
        iters=config.get("reg_anneal_iters", 0),
        maxval=config.get("reg_coeff_maxval", 1.0),
    )
    start_step = 1
    if args.ckpt:
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optimizer"])
        # Hack to move optim states from CPU to GPU.
        for state in optim.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
        lr_scheduler.load_state_dict(ckpt["scheduler"])
        start_step = ckpt["step"] + 1
    model = model.to(device)
    num_params = 0
    for name, param in model.named_parameters():
        num_params += np.prod(param.size())
        print(name, param.size())
    print(f"Total Paramaters: {num_params.item()}")

    # TRAIN & EVALUATE
    num_steps = config["num_steps"]
    log_steps = config["log_steps"]
    save_steps = config["save_steps"]
    log_dir = config["log_dir"]
    writer = SummaryWriter(logdir=log_dir)
    with open(os.path.join(log_dir, "config.yaml"), "w") as fp:
        yaml.dump(config, fp, default_flow_style=False, sort_keys=False)
    for step in range(start_step, num_steps + 1):
        try:
            train_batch = next(train_gen)
        except StopIteration:
            train_gen = iter(train_loader)
            train_batch = next(train_gen)
        train_result = train_step(
            train_batch, model, optim, reg_scheduler, step, device, config
        )
        summary_items = copy.deepcopy(train_result)
        if step % config["lr_decay_steps"] == 0:
            lr_scheduler.step()
        if step % config.get("reg_anneal_every", 1) == 0:
            reg_scheduler.step()
        if step % save_steps == 0 or step == num_steps:
            model_path = os.path.join(config["ckpt_dir"], f"model_{step}.pt")
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optim.state_dict(),
                    "scheduler": lr_scheduler.state_dict(),
                    "config": config,
                },
                model_path,
            )

        if step % log_steps == 0 or step == num_steps:
            folder = os.path.join(log_dir, "plots", f"step{step}")
            os.makedirs(folder, exist_ok=True)
            (
                wt_mean,
                wt_conf_interval,
                future_w_mean,
                future_w_conf_interval,
            ) = evaluate_pymunk_dataset(
                val_loader,
                model,
                device=device,
                num_samples=config["num_forecast"],
                max_size=100,
            ).values()
            writer.add_scalar("future_w_mean", future_w_mean.item(), global_step=step)
            writer.add_scalar(
                "future_w_conf_interval",
                future_w_conf_interval.item(),
                global_step=step,
            )
            fig = show_wasserstein_distance(
                (15, 2),
                wt_mean,
                conf_intervals=wt_conf_interval,
                fig_title="Wasserstein Distance",
            )
            writer.add_figure("w_dist", fig, global_step=step)
            plot_count = 0
            for test_batch in val_loader:
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
                )
                reconstruction = predict_result["reconstruction"]
                forecast = predict_result["forecast"]
                full_prediction = torch.cat([reconstruction, forecast], dim=-2)
                full_target = torch.cat([past_target, future_target], dim=-2)
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
                    full_target_j[:T][mask[j] == 0.0] = 0.0
                    # Plot first five samples
                    show_pymunk_forecast(
                        torch2numpy(full_target_j),
                        torch2numpy(full_prediction_j[:5]),
                        os.path.join(folder, f"series_{plot_count}.png"),
                    )
                    plot_count += 1

                    if plot_count == config["num_plots"]:
                        break
                if plot_count == config["num_plots"]:
                    break
        for k, v in summary_items.items():
            writer.add_scalar(k, v, global_step=step)
        writer.flush()


if __name__ == "__main__":
    main()
