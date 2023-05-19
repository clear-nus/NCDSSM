import ot
import torch
import scipy
import numpy as np
import multiprocessing

from tqdm.auto import tqdm

from .torch_utils import torch2numpy


def student_t_conf_interval(samples, confidence_level=0.95, axis=0):
    deg_free = samples.shape[axis] - 1
    mean = np.mean(samples, axis=axis)
    str_err = scipy.stats.sem(samples, axis=axis)
    a, b = scipy.stats.t.interval(confidence_level, deg_free, mean, str_err)
    return mean, a, b


def compute_wasserstein_distance(img_gt, img_model, metric="euclidean"):
    assert img_gt.ndim == img_model.ndim == 2

    # get positions in x-y-plane for pixels that
    # take value "1" (interpreted as our samples).
    pos_gt = np.stack(np.where(img_gt == 1), axis=-1)
    pos_model = np.stack(np.where(img_model == 1), axis=-1)

    # assume that the binary distribution over
    # pixel value taking value 1 at x-y position
    # is the uniform empirical distribution.
    prob_gt = ot.unif(len(pos_gt))
    prob_model = ot.unif(len(pos_model))

    # euclidean distance times number of pixels
    #  --> *total* (not avg) number of pixel movements.
    M = ot.dist(pos_gt, pos_model, metric=metric)
    dist_avg = ot.emd2(prob_gt, prob_model, M)
    # dist_total = dist_avg * len(pos_gt)
    return dist_avg


def _wasserstein_worker_function(data):
    orig_data, pred_data = data
    assert orig_data.ndim == 2
    assert pred_data.ndim == 3
    assert orig_data.shape[0] == pred_data.shape[1]
    N, T, _ = pred_data.shape
    w_dist = np.zeros((N, T))
    for t in range(T):
        for n in range(N):
            gt_img = orig_data[t]
            pred_img = pred_data[n, t]
            wass_dist = compute_wasserstein_distance(
                gt_img.reshape(32, 32), pred_img.reshape(32, 32)
            )
            w_dist[n, t] = wass_dist
    return w_dist


def evaluate_pymunk_dataset(
    dataloader,
    model,
    device=torch.device("cpu"),
    num_samples: int = 80,
    max_size: int = np.inf,  # type: ignore
    no_state_sampling: bool = False,
    use_smooth: bool = False,
):
    def _binarize_image(img):
        img[img < 0.5] = 0.0
        img[img >= 0.5] = 1.0
        return img

    wass_dists = []
    size = 0
    for test_batch in tqdm(dataloader, desc="Evaluating"):
        past_target = test_batch["past_target"].to(device)
        mask = test_batch["past_mask"].to(device)
        future_target = test_batch["future_target"].to(device)
        past_times = test_batch["past_times"].to(device)
        future_times = test_batch["future_times"].to(device)
        predict_result = model.forecast(
            past_target,
            mask,
            past_times.view(-1),
            future_times.view(-1),
            num_samples=num_samples,
            no_state_sampling=no_state_sampling,
            use_smooth=use_smooth,
        )
        reconstruction = predict_result["reconstruction"]
        forecast = predict_result["forecast"]
        full_prediction = torch.cat([reconstruction, forecast], dim=-2)
        full_target = torch.cat([past_target, future_target], dim=1)

        full_prediction = torch2numpy(full_prediction)
        full_target = torch2numpy(full_target)

        full_prediction = np.swapaxes(full_prediction, 0, 1)  # Batch first

        batch_wass_dist = []
        mp_pool = multiprocessing.Pool(
            initializer=None, processes=multiprocessing.cpu_count()
        )
        batch_wass_dist = mp_pool.map(
            func=_wasserstein_worker_function,
            iterable=zip(full_target, full_prediction),
        )
        mp_pool.close()
        mp_pool.join()
        wass_dists.append(np.array(batch_wass_dist))
        size += past_target.shape[0]
        if size >= max_size:
            break
    # Concate on batch dim
    wass_dists: np.ndarray = np.concatenate(wass_dists, axis=0)  # shape = B, N, T
    # Average over the batch
    batch_wass_dists = wass_dists.mean(0)
    wt_mean, _, wt_conf_interval = student_t_conf_interval(
        batch_wass_dists, confidence_level=0.95, axis=0
    )  # Used for plotting
    future_w_mean, _, future_w_conf_interval = student_t_conf_interval(
        batch_wass_dists[:, past_target.size(1) :].mean(axis=-1),
        confidence_level=0.95,
        axis=0,
    )
    return dict(
        wt_mean=wt_mean,
        wt_conf_interval=wt_conf_interval - wt_mean,
        future_w_mean=future_w_mean,
        future_w_conf_interval=future_w_conf_interval - future_w_mean,
    )


def evaluate_simple_ts(
    dataloader,
    model,
    device=torch.device("cpu"),
    num_samples=50,
    no_state_sampling=False,
    use_smooth=False,
    return_states=False,
):
    imputation_sq_errs = []
    forecast_sq_errs = []
    mses_mean_forecast = []
    mses_mean_imputation = []
    mask_sum = 0.0
    states = []
    for test_batch in tqdm(dataloader, desc="Evaluating"):
        past_target = test_batch["past_target"].to(device)
        B, T, F = past_target.shape
        mask = test_batch["past_mask"].to(device)
        future_target = test_batch["future_target"].to(device)
        past_times = test_batch["past_times"].to(device)
        future_times = test_batch["future_times"].to(device)

        predict_result = model.forecast(
            past_target,
            mask,
            past_times.view(-1),
            future_times.view(-1),
            num_samples=num_samples,
            no_state_sampling=no_state_sampling,
            use_smooth=use_smooth,
        )
        reconstruction = predict_result["reconstruction"]
        forecast = predict_result["forecast"]
        if return_states:
            states.append(
                torch.cat(
                    [predict_result["z_reconstruction"], predict_result["z_forecast"]],
                    dim=-2,
                )
            )
        # Compute MSE using samples
        rec_sq_err = (past_target[None] - reconstruction) ** 2
        rec_sq_err = torch.einsum("sbtf, bt -> sbtf", rec_sq_err, 1 - mask)
        mask_sum += torch.sum(1 - mask, dim=(-1, -2))
        imputation_sq_errs.append(rec_sq_err)
        forecast_sq_err = (future_target[None] - forecast) ** 2
        forecast_sq_errs.append(forecast_sq_err)
        # Compute MSE using mean forecast
        mean_rec = reconstruction.mean(0)
        batch_mse_mean_rec = torch.einsum(
            "btf, bt -> b", (past_target - mean_rec) ** 2, 1 - mask
        ) / (torch.sum(1 - mask, dim=-1) * F)
        mses_mean_imputation.append(batch_mse_mean_rec)
        mean_forecast = forecast.mean(0)
        batch_mse_mean_forecast = torch.mean(
            (future_target - mean_forecast) ** 2, (1, 2)
        )
        mses_mean_forecast.append(batch_mse_mean_forecast)
    imputation_sq_errs = torch.cat(imputation_sq_errs, 1)
    imputation_msq_errs = (
        (torch.sum(imputation_sq_errs, (1, 2, 3)) / (mask_sum * F))
        .detach()
        .cpu()
        .numpy()
    )
    imputation_mean_mse, _, imputation_conf_interval = student_t_conf_interval(
        imputation_msq_errs, confidence_level=0.95, axis=0
    )

    forecast_sq_errs = torch.cat(forecast_sq_errs, 1)
    forecast_msq_errs = torch.mean(forecast_sq_errs, (1, 2, 3)).detach().cpu().numpy()
    forecast_mean_mse, _, forecast_conf_interval = student_t_conf_interval(
        forecast_msq_errs, confidence_level=0.95, axis=0
    )

    mse_mean_imputation = torch.cat(mses_mean_imputation, 0).mean(0)
    mse_mean_forecast = torch.cat(mses_mean_forecast, 0).mean(0)
    imputation_delta_ci = imputation_conf_interval - imputation_mean_mse
    forecast_delta_ci = forecast_conf_interval - forecast_mean_mse
    print(
        f"Imputation MSE: {imputation_mean_mse.item():.5f} +/- {imputation_delta_ci:.5f}"  # noqa
    )
    print(f"Imputation MSE (of mean rec): {mse_mean_imputation.item():.5f}")
    print(f"MSE: {forecast_mean_mse.item():.5f} +/- {forecast_delta_ci:.5f}")
    print(f"MSE (of mean forecast): {mse_mean_forecast.item():.5f}")

    extra_return = {}
    if return_states:
        states = torch.cat(states, dim=1)
        extra_return = {"states": states}
    return dict(
        imputation_mse=imputation_mean_mse.item(),
        mse_imputation_rec=mse_mean_imputation.item(),
        imputation_conf_interval=imputation_delta_ci.item(),
        forecast_mse=forecast_mean_mse.item(),
        mse_mean_forecast=mse_mean_forecast.item(),
        forecast_conf_interval=forecast_delta_ci.item(),
        **extra_return,
    )


def evaluate_sporadic(
    dataloader,
    model,
    device=torch.device("cpu"),
    num_samples=50,
    no_state_sampling=False,
    use_smooth=False,
):
    sq_err_sum = 0
    mask_sum = 0
    forecast_sq_errs = []
    mses_mean_forecast = []
    for test_batch in tqdm(dataloader, desc="Evaluating"):
        past_target = test_batch["past_target"].to(device)
        B, T, F = past_target.shape
        mask = test_batch["past_mask"].to(device)
        future_target = test_batch["future_target"].to(device)
        past_times = test_batch["past_times"].to(device)
        future_times = test_batch["future_times"].to(device)
        future_mask = test_batch["future_mask"].to(device)

        predict_result = model.forecast(
            past_target,
            mask,
            past_times.view(-1),
            future_times.view(-1),
            num_samples=num_samples,
            no_state_sampling=no_state_sampling,
            use_smooth=use_smooth,
        )
        forecast = predict_result["forecast"]
        # Compute MSE using samples

        forecast_sq_err = (future_target[None] - forecast) ** 2
        forecast_sq_err = torch.einsum(
            "sbtf, btf -> sb", forecast_sq_err, future_mask
        ) / (torch.sum(future_mask, dim=(-1, -2)))
        forecast_sq_errs.append(forecast_sq_err)
        # Compute MSE using mean forecast

        mean_forecast = forecast.mean(0)
        batch_mse_mean_forecast = torch.einsum(
            "btf, btf -> b", (future_target - mean_forecast) ** 2, future_mask
        ) / (torch.sum(future_mask, dim=(-1, -2)))
        mses_mean_forecast.append(batch_mse_mean_forecast)
        sq_err_sum += torch.einsum(
            "btf, btf -> ", (future_target - mean_forecast) ** 2, future_mask
        )
        mask_sum += torch.sum(future_mask)

    forecast_sq_errs = torch.cat(forecast_sq_errs, 1)
    forecast_msq_errs = torch.mean(forecast_sq_errs, 1).detach().cpu().numpy()
    forecast_mean_mse, _, forecast_conf_interval = student_t_conf_interval(
        forecast_msq_errs, confidence_level=0.95, axis=0
    )

    mse_mean_forecast = torch.cat(mses_mean_forecast, 0).mean(0)
    delta_ci = forecast_conf_interval - forecast_mean_mse
    mse_gru_ode_style = sq_err_sum / mask_sum
    print(f"MSE (as computed in GRUODE-B): {mse_gru_ode_style:5f}")

    return dict(
        forecast_mse=forecast_mean_mse.item(),
        mse_mean_forecast=mse_mean_forecast.item(),
        forecast_conf_interval=delta_ci.item(),
        mse_gru_ode_style=mse_gru_ode_style.item(),
    )
