import math
import torch

from torchdiffeq._impl.rk_common import rk4_alt_step_func as rk4_step_func

from .type import Tensor, Union
from .functions import cholesky, mbvp, bm_t, bmbvp, qr, symmetrize, sum_mat_sqrts
from .models.dynamics import (
    ContinuousLTI,
    ContinuousNL,
    ContinuousLL,
)


def analytic_linear_step(mu: Tensor, LSigma: Tensor, t0, t1, dynamics: ContinuousLTI):
    # Analytic step for homogenous linear SDE using matrix exponentials
    # Based on Sarkka and Solin, Section 6.2 & 6.3
    batch_size = mu.size(0)
    z_dim = dynamics.z_dim
    F = dynamics.F
    Q = dynamics.Q

    mexp_F_t1mt0 = torch.matrix_exp(F * (t1 - t0))
    mu_pred = mbvp(mexp_F_t1mt0, mu)
    C_t0 = LSigma @ bm_t(LSigma)
    D_t0 = torch.eye(z_dim, device=C_t0.device).repeat(batch_size, 1, 1)
    CD_t0 = torch.cat([C_t0, D_t0], dim=1)
    tmp = torch.zeros(2 * z_dim, 2 * z_dim, device=C_t0.device)
    tmp[:z_dim, :z_dim] = F
    tmp[:z_dim, z_dim:] = Q
    tmp[z_dim:, z_dim:] = -F.T
    mexp_tmp_t1mt0 = torch.matrix_exp(tmp * (t1 - t0))

    CD_t1 = mexp_tmp_t1mt0[None] @ CD_t0

    C_t1 = CD_t1[:, :z_dim]
    D_t1 = CD_t1[:, z_dim:]

    Sigma_pred = C_t1 @ torch.inverse(D_t1)
    LSigma_pred = cholesky(symmetrize(Sigma_pred))

    return mu_pred, LSigma_pred


def linear_step(
    mu: Tensor,
    Phi: Tensor,
    sqrt_Phi_sum: Tensor,
    tn,
    h,
    dynamics: ContinuousLTI,
    method: str = "rk4",
):
    assert method in {"euler", "rk4"}, f"Unknown solver: {method}!"
    F = dynamics.F
    LQ = cholesky(dynamics.Q[None])
    batched_F = F[None]

    def _mu_rate_func(t, z, **unused_kwargs):
        dz_by_dt = mbvp(F, z)
        return dz_by_dt

    def _Phi_rate_func(t, z, **unused_kwargs):
        return batched_F @ z

    if method == "euler":
        mu_next = mu + h * _mu_rate_func(tn, mu)
        Phi_next = Phi + h * _Phi_rate_func(tn, Phi)
    elif method == "rk4":
        mu_next = mu + rk4_step_func(_mu_rate_func, tn, h, tn + h, mu)
        Phi_next = Phi + rk4_step_func(_Phi_rate_func, tn, h, tn + h, Phi)

    Rtmp = sum_mat_sqrts(Phi @ LQ, Phi_next @ LQ)
    Rtmp = math.sqrt(h / 2) * Rtmp
    Rtmp = sum_mat_sqrts(Rtmp, sqrt_Phi_sum)
    sqrt_Phi_sum = Rtmp

    mu = mu_next
    Phi = Phi_next
    return mu, Phi, sqrt_Phi_sum


def cont_disc_linear_predict(
    mu: Tensor,
    LSigma: Tensor,
    dynamics: ContinuousLTI,
    t0: float,
    t1: float,
    step_size: float,
    method: str = "rk4",
    cache_params: bool = False,
    min_step_size: float = 1e-5,
):
    batch_size = mu.size(0)
    z_dim = dynamics.z_dim
    t = t0
    mu_pred = mu

    if method == "matrix_exp":
        return *analytic_linear_step(mu, LSigma, t0, t1, dynamics), ([], [], [])

    Phi = torch.eye(z_dim, device=mu.device).repeat(batch_size, 1, 1)
    sqrt_Phi_sum = torch.zeros(batch_size, z_dim, z_dim, device=mu.device)
    cached_mus = []
    cached_LSigmas = []
    cached_timestamps = []

    while t < t1:
        h = min(step_size, t1 - t)
        if h < min_step_size:
            break
        mu_pred, Phi, sqrt_Phi_sum = linear_step(
            mu_pred, Phi, sqrt_Phi_sum, t, h, dynamics, method
        )
        if cache_params:
            LSigma_pred = sum_mat_sqrts(Phi @ LSigma, sqrt_Phi_sum)
            cached_mus.append(mu_pred.detach().clone())
            cached_LSigmas.append(LSigma_pred.detach().clone())
        t += h
        if cache_params:
            cached_timestamps.append(t)
    if cache_params:
        # Remove the predicted distribution for t1
        # because this will be replaced by filter distribution
        cached_mus.pop()
        cached_LSigmas.pop()
        cached_timestamps.pop()
    cache = (cached_mus, cached_LSigmas, cached_timestamps)
    LSigma_pred = sum_mat_sqrts(Phi @ LSigma, sqrt_Phi_sum)
    return mu_pred, LSigma_pred, cache


def cont_disc_linear_update(y, mask, mu_pred, LSigma_pred, H, R, sporadic=False):
    batch_size = y.size(0)
    y_dim = y.size(-1)
    z_dim = mu_pred.size(-1)

    if R.size(0) != batch_size:
        R = R.repeat(batch_size, 1, 1)
    LR = cholesky(R)
    if H.size(0) != batch_size:
        H = H.repeat(batch_size, 1, 1)

    if sporadic:
        # mask.shape = B x D
        mask_mat = torch.diag_embed(mask)
        inv_mask_mat = torch.diag_embed(1 - mask)
        # mask_mat.shape = B x D x D

        # y.shape = B x D
        y = y * mask

        # Sqrt factor for:
        # mask_mat @ R @ mask_mat.T + inv_mask_mat @ inv_mask_mat.T
        LR = sum_mat_sqrts(mask_mat @ LR, inv_mask_mat)
        # H.shape = B x D x M
        H = mask_mat @ H

    tmp = torch.zeros(batch_size, y_dim + z_dim, y_dim + z_dim, device=mu_pred.device)
    tmp[:, :y_dim, :y_dim] = LR
    tmp[:, :y_dim, y_dim:] = H @ LSigma_pred
    tmp[:, y_dim:, y_dim:] = LSigma_pred

    _, U = qr(bm_t(tmp))
    U = bm_t(U)
    X = U[:, :y_dim, :y_dim]
    Y = U[:, y_dim:, :y_dim]
    Z = U[:, y_dim:, y_dim:]

    K = Y @ torch.inverse(X)  # Kalman Gain
    y_pred = bmbvp(H, mu_pred)
    r = y - y_pred  # Residual
    # Update mu
    mu = mu_pred + bmbvp(K, r)
    # Update LSigma
    LSigma = Z
    # Compute LS for loss
    LS = sum_mat_sqrts(H @ LSigma_pred, LR)
    return mu, LSigma, y_pred, LS


def locallylinear_step(
    mu: Tensor,
    Phi: Tensor,
    sqrt_Phi_sum: Tensor,
    tn: float,
    h: float,
    dynamics: ContinuousLL,
    method: str = "rk4",
):
    assert method in {"euler", "rk4"}, f"Unknown solver: {method}!"
    F = dynamics.F
    alpha0 = dynamics.alpha_net(mu)
    F0 = torch.einsum(
        "bk, knm -> bnm", alpha0, F
    )  # Assuming fixed F(t) in the interval for cov
    LQ = cholesky(dynamics.Q[None])

    def _mu_rate_func(t, z, **unused_kwargs):
        alpha = dynamics.alpha_net(z)
        Ft = torch.einsum("bk, knm -> bnm", alpha, F)
        dz_by_dt = bmbvp(Ft, z)
        return dz_by_dt

    def _Phi_rate_func(t, z, **unused_kwargs):
        # NOTE: Can matrix exp be used here?
        return F0 @ z

    if method == "euler":
        mu_next = mu + h * _mu_rate_func(tn, mu)
        Phi_next = Phi + h * _Phi_rate_func(tn, Phi)
    elif method == "rk4":
        mu_next = mu + rk4_step_func(_mu_rate_func, tn, h, tn + h, mu)
        Phi_next = Phi + rk4_step_func(_Phi_rate_func, tn, h, tn + h, Phi)

    Rtmp = sum_mat_sqrts(Phi @ LQ, Phi_next @ LQ)
    Rtmp = math.sqrt(h / 2) * Rtmp
    Rtmp = sum_mat_sqrts(Rtmp, sqrt_Phi_sum)
    sqrt_Phi_sum = Rtmp

    mu = mu_next
    Phi = Phi_next
    return mu, Phi, sqrt_Phi_sum


def cont_disc_locallylinear_predict(
    mu,
    LSigma,
    dynamics,
    t0,
    t1,
    step_size,
    method="rk4",
    cache_params: bool = False,
    min_step_size: float = 1e-5,
):
    batch_size = mu.size(0)
    z_dim = dynamics.z_dim
    t = t0
    mu_pred = mu
    Phi = torch.eye(z_dim, device=mu.device).repeat(batch_size, 1, 1)
    sqrt_Phi_sum = torch.zeros(batch_size, z_dim, z_dim, device=mu.device)
    cached_mus = []
    cached_LSigmas = []
    cached_timestamps = []

    while t < t1:
        h = min(step_size, t1 - t)
        if h < min_step_size:
            break
        mu_pred, Phi, sqrt_Phi_sum = locallylinear_step(
            mu_pred, Phi, sqrt_Phi_sum, t, h, dynamics, method
        )
        if cache_params:
            LSigma_pred = sum_mat_sqrts(Phi @ LSigma, sqrt_Phi_sum)
            cached_mus.append(mu_pred.detach().clone())
            cached_LSigmas.append(LSigma_pred.detach().clone())
        t += h
        if cache_params:
            cached_timestamps.append(t)
    if cache_params:
        # Remove the predicted distribution for t1
        # because this will be replaced by filter distribution
        cached_mus.pop()
        cached_LSigmas.pop()
        cached_timestamps.pop()
    cache = (cached_mus, cached_LSigmas, cached_timestamps)
    LSigma_pred = sum_mat_sqrts(Phi @ LSigma, sqrt_Phi_sum)
    return mu_pred, LSigma_pred, cache


def cont_disc_locallylinear_update(
    y: Tensor,
    mask: Tensor,
    mu_pred: Tensor,
    LSigma_pred: Tensor,
    H: Tensor,
    R: Tensor,
    sporadic: bool = False,
):
    return cont_disc_linear_update(
        y, mask, mu_pred, LSigma_pred, H, R, sporadic=sporadic
    )


def nonlinear_step(
    mu: Tensor,
    Phi: Tensor,
    sqrt_Phi_sum: Tensor,
    tn,
    h,
    dynamics: ContinuousNL,
    method: str = "rk4",
):
    assert method in {"euler", "rk4"}, f"Unknown solver: {method}!"
    f = dynamics.f
    J_f0 = dynamics.jac_f(mu)  # Assuming fixed Jac_f(t) in the interval for cov
    LGQGt = cholesky(dynamics.GQGt(mu))

    def _mu_rate_func(t, z, **unused_kwargs):
        dz_by_dt = f(z)
        return dz_by_dt

    def _Phi_rate_func(t, z, **unused_kwargs):
        # NOTE: Can matrix exp be used here?
        return J_f0 @ z

    if method == "euler":
        mu_next = mu + h * _mu_rate_func(tn, mu)
        Phi_next = Phi + h * _Phi_rate_func(tn, Phi)
    elif method == "rk4":
        mu_next = mu + rk4_step_func(_mu_rate_func, tn, h, tn + h, mu)
        Phi_next = Phi + rk4_step_func(_Phi_rate_func, tn, h, tn + h, Phi)

    Rtmp = sum_mat_sqrts(Phi @ LGQGt, Phi_next @ LGQGt)
    Rtmp = math.sqrt(h / 2) * Rtmp
    Rtmp = sum_mat_sqrts(Rtmp, sqrt_Phi_sum)
    sqrt_Phi_sum = Rtmp

    mu = mu_next
    Phi = Phi_next
    return mu, Phi, sqrt_Phi_sum


def cont_disc_nonlinear_predict(
    mu: Tensor,
    LSigma: Tensor,
    dynamics: ContinuousNL,
    t0: float,
    t1: float,
    step_size: float,
    method: str = "rk4",
    cache_params: bool = False,
    min_step_size: float = 1e-5,
):
    batch_size = mu.size(0)
    z_dim = dynamics.z_dim
    t = t0
    mu_pred = mu
    Phi = torch.eye(z_dim, device=mu.device).repeat(batch_size, 1, 1)
    sqrt_Phi_sum = torch.zeros(batch_size, z_dim, z_dim, device=mu.device)
    cached_mus = []
    cached_LSigmas = []
    cached_timestamps = []

    while t < t1:
        h = min(step_size, t1 - t)
        if h < min_step_size:
            break
        mu_pred, Phi, sqrt_Phi_sum = nonlinear_step(
            mu_pred, Phi, sqrt_Phi_sum, t, h, dynamics, method
        )
        if cache_params:
            LSigma_pred = sum_mat_sqrts(Phi @ LSigma, sqrt_Phi_sum)
            cached_mus.append(mu_pred.detach().clone())
            cached_LSigmas.append(LSigma_pred.detach().clone())
        t += h
        if cache_params:
            cached_timestamps.append(t)
    if cache_params:
        # Remove the predicted distribution for t1
        # because this will be replaced by filter distribution
        cached_mus.pop()
        cached_LSigmas.pop()
        cached_timestamps.pop()
    cache = (cached_mus, cached_LSigmas, cached_timestamps)
    LSigma_pred = sum_mat_sqrts(Phi @ LSigma, sqrt_Phi_sum)
    return mu_pred, LSigma_pred, cache


def cont_disc_nonlinear_update(y, mask, mu_pred, LSigma_pred, H, R, sporadic=False):
    return cont_disc_linear_update(
        y, mask, mu_pred, LSigma_pred, H, R, sporadic=sporadic
    )


def linear_smooth_step(
    mu_s: Tensor,
    Phi_s: Tensor,
    sqrt_Phi_sum: Tensor,
    mu_f: Tensor,
    LSigma_f: Tensor,
    tn,
    h,
    dynamics: ContinuousLTI,
    method: str = "rk4",
):
    assert h <= 0
    assert method in {"euler", "rk4"}, f"Unknown solver: {method}!"
    F = dynamics.F
    LQ = cholesky(dynamics.Q[None])
    Sigma_f_inv = torch.cholesky_inverse(LSigma_f)
    QSigma_f_inv = dynamics.Q[None] @ Sigma_f_inv
    A = F[None] + QSigma_f_inv

    def _mu_rate_func(t, z, **unused_kwargs):
        dz_by_dt = mbvp(F, z) + bmbvp(QSigma_f_inv, z - mu_f)
        return dz_by_dt

    def _Phi_rate_func(t, z, **unused_kwargs):
        # NOTE: Can matrix exp be used here?
        return A @ z

    if method == "euler":
        mu_s_next = mu_s + h * _mu_rate_func(tn, mu_s)
        Phi_s_next = Phi_s + h * _Phi_rate_func(tn, Phi_s)
    elif method == "rk4":
        mu_s_next = mu_s + rk4_step_func(_mu_rate_func, tn, h, tn + h, mu_s)
        Phi_s_next = Phi_s + rk4_step_func(_Phi_rate_func, tn, h, tn + h, Phi_s)

    Rtmp = sum_mat_sqrts(Phi_s @ LQ, Phi_s_next @ LQ)
    Rtmp = math.sqrt(-h / 2) * Rtmp
    Rtmp = sum_mat_sqrts(Rtmp, sqrt_Phi_sum)
    sqrt_Phi_sum = Rtmp

    mu_s = mu_s_next
    Phi_s = Phi_s_next
    return mu_s, Phi_s, sqrt_Phi_sum


def type2_locallylinear_smooth_step(
    mu_s: Tensor,
    Phi_s: Tensor,
    sqrt_Phi_sum: Tensor,
    mu_f: Tensor,
    LSigma_f: Tensor,
    tn,
    h,
    dynamics: ContinuousLL,
    method: str = "rk4",
):
    assert h <= 0
    assert method in {"euler", "rk4"}, f"Unknown solver: {method}!"
    F = dynamics.F
    alpha0 = dynamics.alpha_net(mu_f)
    F0 = torch.einsum("bk, knm -> bnm", alpha0, F)
    LQ = cholesky(dynamics.Q[None])
    Sigma_f_inv = torch.cholesky_inverse(LSigma_f)
    QSigma_f_inv = dynamics.Q[None] @ Sigma_f_inv
    A = F0 + QSigma_f_inv

    def _mu_rate_func(t, z, **unused_kwargs):
        dz_by_dt = bmbvp(F0, mu_f) + bmbvp(F0 + QSigma_f_inv, z - mu_f)
        return dz_by_dt

    def _Phi_rate_func(t, z, **unused_kwargs):
        # NOTE: Can matrix exp be used here?
        return A @ z

    if method == "euler":
        mu_s_next = mu_s + h * _mu_rate_func(tn, mu_s)
        Phi_s_next = Phi_s + h * _Phi_rate_func(tn, Phi_s)
    elif method == "rk4":
        mu_s_next = mu_s + rk4_step_func(_mu_rate_func, tn, h, tn + h, mu_s)
        Phi_s_next = Phi_s + rk4_step_func(_Phi_rate_func, tn, h, tn + h, Phi_s)

    Rtmp = sum_mat_sqrts(Phi_s @ LQ, Phi_s_next @ LQ)
    Rtmp = math.sqrt(-h / 2) * Rtmp
    Rtmp = sum_mat_sqrts(Rtmp, sqrt_Phi_sum)
    sqrt_Phi_sum = Rtmp

    mu_s = mu_s_next
    Phi_s = Phi_s_next
    return mu_s, Phi_s, sqrt_Phi_sum


def type2_nonlinear_smooth_step(
    mu_s: Tensor,
    Phi_s: Tensor,
    sqrt_Phi_sum: Tensor,
    mu_f: Tensor,
    LSigma_f: Tensor,
    tn,
    h,
    dynamics: ContinuousNL,
    method: str = "rk4",
):
    assert h <= 0
    assert method in {"euler", "rk4"}, f"Unknown solver: {method}!"
    f = dynamics.f
    GQGt = dynamics.GQGt(mu_f)
    LGQGt = cholesky(GQGt)
    Sigma_f_inv = torch.cholesky_inverse(LSigma_f)
    QSigma_f_inv = GQGt @ Sigma_f_inv
    J_f0 = dynamics.jac_f(mu_f)
    A = J_f0 + QSigma_f_inv

    def _mu_rate_func(t, z, **unused_kwargs):
        dz_by_dt = f(mu_f) + bmbvp(J_f0 + QSigma_f_inv, z - mu_f)
        return dz_by_dt

    def _Phi_rate_func(t, z, **unused_kwargs):
        # NOTE: Can matrix exp be used here?
        return A @ z

    if method == "euler":
        mu_s_next = mu_s + h * _mu_rate_func(tn, mu_s)
        Phi_s_next = Phi_s + h * _Phi_rate_func(tn, Phi_s)
    elif method == "rk4":
        mu_s_next = mu_s + rk4_step_func(_mu_rate_func, tn, h, tn + h, mu_s)
        Phi_s_next = Phi_s + rk4_step_func(_Phi_rate_func, tn, h, tn + h, Phi_s)

    Rtmp = sum_mat_sqrts(Phi_s @ LGQGt, Phi_s_next @ LGQGt)
    Rtmp = math.sqrt(-h / 2) * Rtmp
    Rtmp = sum_mat_sqrts(Rtmp, sqrt_Phi_sum)
    sqrt_Phi_sum = Rtmp

    mu_s = mu_s_next
    Phi_s = Phi_s_next
    return mu_s, Phi_s, sqrt_Phi_sum


@torch.no_grad()
def cont_disc_smooth(
    filter_mus: Tensor,
    filter_LSigmas: Tensor,
    filter_timestamps,
    dynamics: Union[
        ContinuousLTI,
        ContinuousLL,
        ContinuousNL,
    ],
    method: str = "rk4",
):
    batch_size = filter_mus.size(1)
    z_dim = dynamics.z_dim

    filter_mus = torch.flip(filter_mus, dims=(0,))
    filter_LSigmas = torch.flip(filter_LSigmas, dims=(0,))
    filter_timestamps = torch.flip(filter_timestamps, dims=(0,))

    mu_s = filter_mus[0]
    LSigma_s = filter_LSigmas[0]

    smoothed_mus = [mu_s]
    smoothed_LSigmas = [LSigma_s]

    if isinstance(dynamics, ContinuousLTI):
        _smooth_step = linear_smooth_step
    elif isinstance(dynamics, ContinuousLL):
        _smooth_step = type2_locallylinear_smooth_step
    elif isinstance(dynamics, ContinuousNL):
        _smooth_step = type2_nonlinear_smooth_step
    else:
        raise ValueError(f"Unknown dynamics type {type(dynamics)}!")

    for idx, t0 in enumerate(filter_timestamps[:-1]):
        t1 = filter_timestamps[idx + 1]
        h = t1.item() - t0.item()
        mu_f = filter_mus[idx]
        LSigma_f = filter_LSigmas[idx]
        Phi_s = torch.eye(z_dim, device=filter_mus.device).repeat(batch_size, 1, 1)
        sqrt_Phi_sum = torch.zeros(batch_size, z_dim, z_dim, device=Phi_s.device)
        mu_s, Phi_s, sqrt_Phi_sum = _smooth_step(
            mu_s, Phi_s, sqrt_Phi_sum, mu_f, LSigma_f, t0.item(), h, dynamics, method
        )
        LSigma_s = sum_mat_sqrts(Phi_s @ LSigma_s, sqrt_Phi_sum)
        smoothed_mus.append(mu_s.clone())
        smoothed_LSigmas.append(LSigma_s.clone())

    smoothed_mus: Tensor = torch.stack(smoothed_mus)
    smoothed_LSigmas: Tensor = torch.stack(smoothed_LSigmas)
    smoothed_mus, smoothed_LSigmas = torch.flip(smoothed_mus, dims=(0,)), torch.flip(
        smoothed_LSigmas, dims=(0,)
    )
    return smoothed_mus, smoothed_LSigmas
