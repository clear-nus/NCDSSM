import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torchvision.utils import save_image


sns.set(style="white")
color_names = [
    "purple",
    "orange",
    "windows blue",
    "red",
    "amber",
    "faded green",
    "dusty purple",
    "clay",
    "pink",
    "green",
    "greyish",
    "light cyan",
    "steel blue",
    "pastel purple",
    "mint",
    "salmon",
]
xkcd_colors = colors = sns.xkcd_palette(color_names)
# colors = sns.color_palette("muted")


def plot_on_axis(
    ax,
    past_times,
    future_times,
    inputs,
    masked_inputs,
    sorted_rec_and_forecast,
    prediction_intervals=[90.0],
    yticks_off=True,
    ylim=None,
    colors=colors,
    ylabel="",
    shade_context=False,
):
    all_times = np.hstack([past_times, future_times])
    num_samples, _, num_feats = sorted_rec_and_forecast.shape

    for c in prediction_intervals:
        assert 0.0 <= c <= 100.0

    ps = [50.0] + [
        50.0 + f * c / 2.0 for c in prediction_intervals for f in [-1.0, +1.0]
    ]
    percentiles_sorted = sorted(set(ps))

    def alpha_for_percentile(p):
        return (p / 100.0) ** 0.6

    def quantile(q):
        sample_idx = int(np.round((num_samples - 1) * q))
        return sorted_rec_and_forecast[sample_idx, :]

    ps_data = [quantile(p / 100.0) for p in percentiles_sorted]
    i_p50 = len(percentiles_sorted) // 2

    p50_data = ps_data[i_p50]
    if shade_context:
        ax.axvspan(0, future_times[0], facecolor=xkcd_colors[-6], alpha=0.2)
    for o in range(num_feats):
        ax.plot(
            all_times,
            inputs[:, o],
            ls="--",
            lw=1.4,
            color=colors[o],
            alpha=0.6,
        )
        # Median forecast
        ax.plot(all_times, p50_data[:, o], ls="-", lw=1.5, color=colors[o], alpha=1.0)
        ax.scatter(
            past_times,
            masked_inputs[: len(past_times), o],
            s=20,
            marker="o",
            color=colors[o],
            alpha=0.6,
        )
        ax.axvline(future_times[0], color=xkcd_colors[-6], ls=":")

        for i in range(len(percentiles_sorted) // 2):
            ptile = percentiles_sorted[i]
            alpha = alpha_for_percentile(ptile)

            ax.fill_between(
                all_times,
                ps_data[i][:, o],
                ps_data[-i - 1][:, o],
                facecolor=colors[o],
                # edgecolor=colors[o],
                alpha=alpha,
                interpolate=True,
                # label=f"{prediction_intervals[i]}% PI",
            )
            ax.set_ylabel(ylabel)
            if ylim:
                ax.set_ylim(ylim)
            if yticks_off:
                ax.set_yticks([])
    return ax


def show_time_series_forecast(
    fig_size,
    past_times,
    future_times,
    inputs,
    masked_inputs,
    reconstruction,
    forecast,
    prediction_intervals=[90.0],
    fig_title=None,
    file_path=None,
    max_feats=6,
    single_plot=False,
    yticks_off=True,
    ylim=None,
):
    obs_dim = inputs.shape[-1]
    max_feats = min(max_feats, obs_dim)

    rec_and_forecast = np.concatenate([reconstruction, forecast], axis=1)
    rec_and_forecast = rec_and_forecast[..., :, :max_feats]
    inputs = inputs[..., :, :max_feats]
    all_times = np.hstack([past_times, future_times])  # [:150]
    # num_samples = rec_and_forecast.shape[0]

    if single_plot:
        fig_size = (12, 1.0)
        fig, axn = plt.subplots(figsize=fig_size, nrows=1, sharex=True)
        axn = [axn] * max_feats
    else:
        fig_size = (12, max_feats * 1.0)
        fig, axn = plt.subplots(figsize=fig_size, nrows=max_feats, sharex=True)
        if max_feats == 1:
            axn = [axn]
    if fig_title:
        plt.title(fig_title)

    sorted_rec_and_forecast = np.sort(rec_and_forecast, axis=0)
    axn[0].scatter(
        [],
        [],
        s=20,
        marker="o",
        color="k",
        alpha=0.6,
        label="Observations",
    )
    axn[0].plot(
        [],
        [],
        ls="--",
        lw=1.4,
        color="k",
        alpha=0.6,
        label="Ground Truth",
    )
    axn[0].plot(
        [],
        [],
        ls="-",
        lw=1.5,
        color="k",
        label="Median Prediction",
    )
    for o in range(max_feats):
        plot_on_axis(
            axn[o],
            past_times,
            future_times,
            inputs[:, o : o + 1],
            masked_inputs[:, o : o + 1],
            sorted_rec_and_forecast[:, :, o : o + 1],
            prediction_intervals,
            yticks_off,
            ylim,
            colors=colors[o : o + 1],
            ylabel=f"$y_{o}$",
        )

    axn[-1].set_xlabel("Time")
    axn[0].legend(bbox_to_anchor=(0.5, 1.6), ncol=3, loc="upper center")
    axn[0].set_xlim((all_times[0], all_times[-1]))
    # plt.tight_layout()
    if file_path:
        plt.savefig(file_path, dpi=200, bbox_inches="tight")
    return fig


def show_pymunk_forecast(orig, pred, file_path):
    assert orig.shape == pred.shape[1:]
    N, T, C, H, W = pred.shape
    orig = torch.as_tensor(orig)
    pred = torch.as_tensor(pred)
    orig = orig.unsqueeze(0)
    img = torch.cat([orig, pred], 0)
    img = img.view((N + 1) * T, C, H, W)
    save_image(img, file_path, nrow=T, pad_value=0.5)
    return img


def show_wasserstein_distance(
    fig_size, w_dist, conf_intervals=None, fig_title=None, ylim=None, file_path=None
):
    fig = plt.figure(figsize=fig_size)
    if fig_title:
        plt.title(fig_title)
    ax = fig.gca()
    if conf_intervals is not None:
        ax.errorbar(
            np.arange(w_dist.shape[0]),
            w_dist,
            yerr=conf_intervals,
            capsize=5,
            color=colors[-1],
        )
    else:
        ax.plot(w_dist, color=colors[-1])
    ax.set_ylabel("W")
    ax.set_xlabel("T")
    if ylim is not None:
        ax.set_ylim(ylim)
    if file_path:
        plt.savefig(file_path, dpi=200, bbox_inches="tight")
    return fig


def show_latents(fig_size, time, latents, fig_title, file_path=None):
    fig, axn = plt.subplots(figsize=fig_size, nrows=len(latents), sharex=True)
    if len(latents) == 1:
        axn = [axn]
    if fig_title:
        plt.suptitle(fig_title)
    for i, key in enumerate(latents):
        ts = latents[key]
        dims = ts.shape[-1]
        for d in range(dims):
            axn[i].plot(time, ts[:, d], color=colors[d])
        axn[i].set_ylabel(key)
    axn[-1].set_xlabel("time")
    if file_path:
        plt.savefig(file_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        return fig
