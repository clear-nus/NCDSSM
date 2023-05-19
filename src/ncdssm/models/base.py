from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal


from .dynamics import (
    ContinuousLTI,
    ContinuousNL,
    ContinuousLL,
)
from ..inference import (
    cont_disc_linear_predict,
    cont_disc_linear_update,
    cont_disc_locallylinear_predict,
    cont_disc_locallylinear_update,
    cont_disc_nonlinear_predict,
    cont_disc_nonlinear_update,
    cont_disc_smooth,
)
from ..torch_utils import skew_symmetric_init_
from ..functions import cholesky, bmbvp
from ..type import Optional, Tensor, List, Dict, Union


class Base(ABC):
    @abstractmethod
    def predict_step(
        self,
        mu: Tensor,
        LSigma: Tensor,
        dynamics: Union[ContinuousLTI, ContinuousNL, ContinuousLL],
        t0: float,
        t1: float,
        step_size: float,
        method: str,
        cache_params: bool = False,
        min_step_size: float = 1e-5,
    ):
        pass

    @abstractmethod
    def update_step(
        self,
        y: Tensor,
        mask: Tensor,
        mu_pred: Tensor,
        LSigma_pred: Tensor,
        H: Tensor,
        R: Tensor,
        sporadic: bool = False,
    ):
        pass

    def filter(
        self,
        y: Tensor,
        mask: Tensor,
        times: Tensor,
        cache_params: bool = False,
    ) -> Dict[str, Tensor]:
        """The filter step of the continuous-discrete model.

        Parameters
        ----------
        y
            The tensor of observations, of shape (batch_size, num_timesteps, y_dim)
        mask
            The mask of missing values (1: observed, 0: missing),
            of shape (batch_size, num_timesteps, y_dim), if sporadic,
            else (batch_size, num_timesteps)
        times
            The tensor observations times, of shape (num_timesteps,)
        cache_params, optional
            Whether to cache the intermediate distributions computed during
            predict step, by default False and all distrbutions at observation ``times``
            are cached in any case,
            should be set to True if you're planning to use smoothing

        Returns
        -------
            The dictionary of filter outputs, including the log-likelihood
            and the parameters of filtered distributions
        """
        if self.sporadic:
            assert (
                y.size() == mask.size()
            ), f"Shapes of y ({y.size()}) and mask ({mask.size()}) should match!"
        else:
            assert y.size()[:-1] == mask.size(), (
                f"Shapes of y ({y.size()}) and mask ({mask.size()})"
                " should match except in last dim!"
            )
        batch_size = y.size(0)

        # Assumes that t = 0 is in times
        assert times[0] == 0.0, "First timestep should be 0!"
        mu_pred = self.mu0[None].repeat(batch_size, 1)
        LSigma_pred = cholesky(self.Sigma0[None]).repeat(batch_size, 1, 1)
        log_prob = []

        filtered_mus = []
        filtered_LSigmas = []
        cached_mus = []
        cached_LSigmas = []
        cached_timestamps = []

        for idx, t in enumerate(times):
            H = self.H[None]
            R = self.R[None]
            # UPDATE step
            y_i = y[:, idx]
            mask_i = mask[:, idx]
            mu, LSigma, y_pred, LS = self.update_step(
                y=y_i,
                mask=mask_i,
                mu_pred=mu_pred,
                LSigma_pred=LSigma_pred,
                H=H,
                R=R,
                sporadic=self.sporadic,
            )
            if not self.sporadic:
                # Mask out updates
                # For the sporadic case, masks are directly incorporated
                # during the update step
                # TODO: Modify update step such that masking is not required here
                mu = mask_i[:, None] * mu + (1 - mask_i[:, None]) * mu_pred
                LSigma = (
                    mask_i[:, None, None] * LSigma
                    + (1 - mask_i[:, None, None]) * LSigma_pred
                )

            # Calculate log_prob
            if not self.sporadic:
                dist = MultivariateNormal(loc=y_pred, scale_tril=LS)
                log_prob.append(dist.log_prob(y_i) * mask_i)
            else:
                dist = MultivariateNormal(loc=y_pred * mask_i, scale_tril=LS)
                log_prob.append(dist.log_prob(y_i * mask_i))

            # Cache filtered distribution
            filtered_mus.append(mu.clone())
            filtered_LSigmas.append(LSigma.clone())
            cached_mus.append(mu.detach().clone())
            cached_LSigmas.append(LSigma.detach().clone())
            cached_timestamps.append(t.item())
            if idx == len(times) - 1:
                break
            # PREDICT step
            (
                mu_pred,
                LSigma_pred,
                (cached_mus_t, cached_LSigmas_t, cached_timestamps_t),
            ) = self.predict_step(
                mu=mu,
                LSigma=LSigma,
                dynamics=self.dynamics,
                t0=t.item(),
                t1=times[idx + 1].item(),
                step_size=self.integration_step_size,
                method=self.integration_method,
                cache_params=cache_params,
            )
            cached_mus.extend(cached_mus_t)
            cached_LSigmas.extend(cached_LSigmas_t)
            cached_timestamps.extend(cached_timestamps_t)

        filtered_mus: Tensor = torch.stack(filtered_mus)
        filtered_LSigmas: Tensor = torch.stack(filtered_LSigmas)
        cached_mus: Tensor = torch.stack(cached_mus)
        cached_LSigmas: Tensor = torch.stack(cached_LSigmas)
        cached_timestamps: Tensor = torch.tensor(
            cached_timestamps, dtype=times.dtype, device=times.device
        )
        log_prob = torch.stack(log_prob).sum(0)
        return dict(
            filtered_mus=filtered_mus,
            filtered_LSigmas=filtered_LSigmas,
            last_filtered_dist=(mu, LSigma),
            log_prob=log_prob,
            cached_mus=cached_mus,
            cached_LSigmas=cached_LSigmas,
            cached_timestamps=cached_timestamps,
        )

    def emit(self, z_t: Tensor) -> Tensor:
        """Emit an observation from the latent state.

        Parameters
        ----------
        z_t
            The latent state at a specific time, of shape (batch_size, z_dim)

        Returns
        -------
            The observation at a specific time, of shape (batch_size, y_dim)
        """
        R = self.R
        H = self.H[None]
        p_v = MultivariateNormal(
            torch.zeros(1, self.y_dim, device=z_t.device), covariance_matrix=R
        )
        v = p_v.sample((z_t.shape[0],)).view(z_t.shape[0], self.y_dim)
        y = bmbvp(H, z_t) + v
        return y

    @torch.no_grad()
    def forecast(
        self,
        y: Tensor,
        mask: Tensor,
        past_times: Tensor,
        future_times: Tensor,
        num_samples: int = 80,
        no_state_sampling: bool = False,
        use_smooth: bool = False,
        **unused_kwargs,
    ) -> Dict[str, Tensor]:
        """Make predictions (imputation and forecast) using the observed data.

        Parameters
        ----------
        y
            The tensor of observations, of shape (batch_size, num_timesteps, y_dim)
        mask
            The mask of missing values (1: observed, 0: missing),
            of shape (batch_size, num_timesteps, y_dim), if sporadic,
            else (batch_size, num_timesteps)
        past_times
            The times of the past observations, of shape (num_past_steps,)
        future_times
            The times of the forecast, of shape (num_forecast_steps,)
        num_samples, optional
            The number of sample paths to draw, by default 80
        no_state_sampling, optional
            Whether to sample from the predicted state distributions,
            by default False and only uses the means of the distributions
        use_smooth, optional
            Whether to perform smoothing after filtering (useful for imputation),
            by default False

        Returns
        -------
            The reconstructed context (imputing values, if required) and the forecast
        """
        B, _, _ = y.shape
        filter_result = self.filter(y, mask, past_times, cache_params=use_smooth)
        filtered_mus = filter_result["filtered_mus"]
        filtered_LSigmas = filter_result["filtered_LSigmas"]

        if use_smooth:
            cached_mus = filter_result["cached_mus"]
            cached_LSigmas = filter_result["cached_LSigmas"]
            cached_timestamps = filter_result["cached_timestamps"]
            smoothed_mus, smoothed_LSigmas = cont_disc_smooth(
                filter_mus=cached_mus,
                filter_LSigmas=cached_LSigmas,
                filter_timestamps=cached_timestamps,
                dynamics=self.dynamics,
                method=self.integration_method,
            )
            relevant_indices = torch.isin(cached_timestamps, past_times)
            assert torch.allclose(cached_timestamps[relevant_indices], past_times)
            filtered_mus, filtered_LSigmas = (
                smoothed_mus[relevant_indices],
                smoothed_LSigmas[relevant_indices],
            )

        filter_dists = MultivariateNormal(filtered_mus, scale_tril=filtered_LSigmas)
        z_filtered = filter_dists.sample([num_samples])
        # z_filtered.shape = num_samples x T x B x z_dim

        z_reconstruction = []
        y_reconstruction = []
        for i, t in enumerate(past_times):
            z_t = z_filtered[:, i].reshape(-1, self.z_dim)
            y_t = self.emit(z_t)
            z_reconstruction.append(z_t)
            y_reconstruction.append(y_t)

        z_reconstruction: Tensor = torch.stack(z_reconstruction)
        y_reconstruction: Tensor = torch.stack(y_reconstruction)

        z_reconstruction = z_reconstruction.view(
            past_times.shape[0], num_samples, B, self.z_dim
        )
        y_reconstruction = y_reconstruction.view(
            past_times.shape[0], num_samples, B, self.y_dim
        )
        z_reconstruction = z_reconstruction.permute(1, 2, 0, 3)
        y_reconstruction = y_reconstruction.permute(1, 2, 0, 3)

        (mu, LSigma) = filter_result["last_filtered_dist"]
        future_times = torch.cat([past_times[-1:], future_times], 0)
        mu_t = mu.repeat(num_samples, 1)
        LSigma_t = LSigma.repeat(num_samples, 1, 1)
        y_forecast = []
        z_forecast = []
        for t1, t2 in zip(future_times[:-1], future_times[1:]):
            mu_t2, LSigma_t2, _ = self.predict_step(
                mu=mu_t,
                LSigma=LSigma_t,
                dynamics=self.dynamics,
                t0=t1.item(),
                t1=t2.item(),
                step_size=self.integration_step_size,
                method=self.integration_method,
            )
            pred_dist = MultivariateNormal(mu_t2, scale_tril=LSigma_t2)
            z_t2 = pred_dist.mean if no_state_sampling else pred_dist.sample()
            y_t2 = self.emit(z_t2)
            z_forecast.append(z_t2)
            y_forecast.append(y_t2)
            mu_t = mu_t2
            LSigma_t = LSigma_t2

        z_forecast = torch.stack(z_forecast)
        y_forecast = torch.stack(y_forecast)
        z_forecast = z_forecast.view(
            future_times.shape[0] - 1, num_samples, B, self.z_dim
        )
        y_forecast = y_forecast.view(
            future_times.shape[0] - 1, num_samples, B, self.y_dim
        )
        z_forecast = z_forecast.permute(1, 2, 0, 3)
        y_forecast = y_forecast.permute(1, 2, 0, 3)
        return dict(
            reconstruction=y_reconstruction,
            forecast=y_forecast,
            z_reconstruction=z_reconstruction,
            z_forecast=z_forecast,
        )


class BaseLTI(nn.Module, Base):
    """Base continuous-discrete linear time-invariant state space model.

    Parameters
    ----------
    z_dim
        The dimension of latent state z
    y_dim
        The dimension of observation y
    u_dim
        The dimension of control input u
    F, optional
        The linear dynamics matrix F, by default None
    B, optional
        The linear control matrix B, by default None
    Q, optional
        The state covariance matrix Q, by default None
    H, optional
        The observation matrix H, by default None
    R, optional
        The observation covariance matrix R, by default None
    mu0, optional
        The mean of the initial state distribution, by default None
    Sigma0, optional
        The covariance of the initial state distribution, by default None
    integration_method, optional
        The ODE integration method, should be one of "euler" or "rk4", by default "rk4"
    integration_step_size, optional
        The ODE integration step size, by default 0.1
    sporadic, optional
        A flag to indicate whether the dataset is sporadic,
        i.e., with values missing in both time and feature dimensions, by default False
    """

    def __init__(
        self,
        z_dim: int,
        y_dim: int,
        u_dim: int,
        F: Optional[Tensor] = None,
        B: Optional[Tensor] = None,
        Q: Optional[Tensor] = None,
        H: Optional[Tensor] = None,
        R: Optional[Tensor] = None,
        mu0: Optional[Tensor] = None,
        Sigma0: Optional[Tensor] = None,
        integration_method: str = "rk4",
        integration_step_size: float = 0.1,
        sporadic: bool = False,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.u_dim = u_dim
        self.integration_method = integration_method
        self.integration_step_size = integration_step_size
        self.sporadic = sporadic

        self.mu0 = nn.Parameter(
            mu0
            if mu0 is not None
            else torch.zeros(
                z_dim,
            ),
        )
        self.uSigma0 = nn.Parameter(
            torch.log(Sigma0)
            if Sigma0 is not None
            else torch.zeros(
                z_dim,
            )
        )

        F = (
            F
            if F is not None
            else (
                skew_symmetric_init_(torch.empty(z_dim, z_dim))
                if sporadic
                else nn.init.xavier_uniform_(torch.empty(z_dim, z_dim))
            )
        )

        uQ = (
            torch.log(Q)
            if Q is not None
            else torch.zeros(
                z_dim,
            )
        )

        self.dynamics = ContinuousLTI(z_dim=z_dim, u_dim=u_dim, F=F, B=B, uQ=uQ)

        if H is not None:
            self.register_buffer("H", H)
        else:
            self.H = nn.Parameter(nn.init.xavier_uniform_(torch.empty(y_dim, z_dim)))
        self.uR = nn.Parameter(
            torch.log(R)
            if R is not None
            else torch.zeros(
                y_dim,
            )
        )
        self.register_buffer("I", torch.eye(z_dim))

    @property
    def Sigma0(self):
        return torch.diag(torch.clamp(torch.exp(self.uSigma0), min=1e-4))

    @property
    def R(self):
        return torch.diag(torch.clamp(torch.exp(self.uR), min=1e-4))

    def forward(
        self, y: Tensor, mask: Tensor, times: Tensor, **unused_kwargs
    ) -> Dict[str, Tensor]:
        likelihood = self.filter(y, mask, times)["log_prob"]
        regularizer = torch.tensor(0.0)
        return dict(likelihood=likelihood, regularizer=regularizer)

    def predict_step(
        self,
        mu: Tensor,
        LSigma: Tensor,
        dynamics: Union[ContinuousLTI, ContinuousNL, ContinuousLL],
        t0: float,
        t1: float,
        step_size: float,
        method: str,
        cache_params: bool = False,
        min_step_size: float = 1e-5,
    ):
        return cont_disc_linear_predict(
            mu,
            LSigma,
            dynamics,
            t0,
            t1,
            step_size,
            method,
            cache_params=cache_params,
            min_step_size=min_step_size,
        )

    def update_step(
        self,
        y: Tensor,
        mask: Tensor,
        mu_pred: Tensor,
        LSigma_pred: Tensor,
        H: Tensor,
        R: Tensor,
        sporadic: bool = False,
    ):
        return cont_disc_linear_update(y, mask, mu_pred, LSigma_pred, H, R, sporadic)


class BaseLL(nn.Module, Base):
    """Base continuous-discrete locally-linear state space model.

    Parameters
    ----------
    K
        The number of base matrices (i.e., dynamics)
    z_dim
        The dimension of latent state z
    y_dim
        The dimension of observation y
    u_dim
        The dimension of control input u
    alpha_net
        A mixing network that takes the state `z` as input
        and outputs the mixing coefficients for the base dynamics
    F, optional
        The linear dynamics matrix F, by default None
    B, optional
        The linear control matrix B, by default None
    Q, optional
        The state covariance matrix Q, by default None
    H, optional
        The observation matrix H, by default None
    R, optional
        The observation covariance matrix R, by default None
    mu0, optional
        The mean of the initial state distribution, by default None
    Sigma0, optional
        The covariance of the initial state distribution, by default None
    integration_method, optional
        The ODE integration method, should be one of "euler" or "rk4", by default "rk4"
    integration_step_size, optional
        The ODE integration step size, by default 0.1
    sporadic, optional
        A flag to indicate whether the dataset is sporadic,
        i.e., with values missing in both time and feature dimensions, by default False
    """

    def __init__(
        self,
        K: int,
        z_dim: int,
        y_dim: int,
        u_dim: int,
        alpha_net: nn.Module,
        F: Optional[Tensor] = None,
        B: Optional[Tensor] = None,
        Q: Optional[Tensor] = None,
        H: Optional[Tensor] = None,
        R: Optional[Tensor] = None,
        mu0: Optional[Tensor] = None,
        Sigma0: Optional[Tensor] = None,
        integration_method: str = "rk4",
        integration_step_size: float = 0.1,
        sporadic: bool = False,
    ):
        super().__init__()

        self.K = K
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.u_dim = u_dim
        self.integration_method = integration_method
        self.integration_step_size = integration_step_size
        self.sporadic = sporadic

        self.mu0 = nn.Parameter(
            mu0
            if mu0 is not None
            else torch.zeros(
                z_dim,
            ),
        )  # shared across the K base dynamics
        self.uSigma0 = nn.Parameter(
            torch.log(Sigma0)
            if Sigma0 is not None
            else torch.zeros(
                z_dim,
            )
        )  # shared across the K base dynamics

        F = F if F is not None else nn.init.orthogonal_(torch.empty(K, z_dim, z_dim))

        uQ = (
            torch.log(Q)
            if Q is not None
            else torch.zeros(
                z_dim,
            )
        )  # shared across the K base dynamics

        self.dynamics = ContinuousLL(
            z_dim=z_dim,
            u_dim=u_dim,
            K=K,
            F=F,
            B=B,
            uQ=uQ,
            alpha_net=alpha_net,
        )
        # H is shared across the K base dynamics
        if H is not None:
            self.register_buffer("H", H)
        else:
            self.H = nn.Parameter(nn.init.xavier_uniform_(torch.empty(y_dim, z_dim)))
        self.uR = nn.Parameter(
            torch.log(R)
            if R is not None
            else torch.zeros(
                y_dim,
            )
        )  # shared across the K base dynamics
        self.register_buffer("I", torch.eye(z_dim))

    @property
    def Sigma0(self):
        return torch.diag(torch.clamp(torch.exp(self.uSigma0), min=1e-4))

    @property
    def R(self):
        return torch.diag(torch.clamp(torch.exp(self.uR), min=1e-4))

    def forward(self, y: Tensor, mask: Tensor, times: Tensor, **unused_kwargs):
        likelihood = self.filter(y, mask, times)["log_prob"]
        regularizer = torch.tensor(0.0)
        return dict(likelihood=likelihood, regularizer=regularizer)

    def predict_step(
        self,
        mu: Tensor,
        LSigma: Tensor,
        dynamics: Union[ContinuousLTI, ContinuousNL, ContinuousLL],
        t0: float,
        t1: float,
        step_size: float,
        method: str,
        cache_params: bool = False,
        min_step_size: float = 1e-5,
    ):
        return cont_disc_locallylinear_predict(
            mu,
            LSigma,
            dynamics,
            t0,
            t1,
            step_size,
            method,
            cache_params=cache_params,
            min_step_size=min_step_size,
        )

    def update_step(
        self,
        y: Tensor,
        mask: Tensor,
        mu_pred: Tensor,
        LSigma_pred: Tensor,
        H: Tensor,
        R: Tensor,
        sporadic: bool = False,
    ):
        return cont_disc_locallylinear_update(
            y, mask, mu_pred, LSigma_pred, H, R, sporadic
        )

    @torch.no_grad()
    def forecast(
        self,
        y: Tensor,
        mask: Tensor,
        past_times: Tensor,
        future_times: Tensor,
        num_samples: int = 80,
        no_state_sampling: bool = False,
        use_smooth: bool = False,
        **unused_kwargs,
    ):
        """Same as :func:`Base.forecast` but additionally returns
        the mixing coefficients (alpha).
        """
        B, T, _ = y.shape
        filter_result = self.filter(y, mask, past_times, cache_params=use_smooth)
        filtered_mus = filter_result["filtered_mus"]
        filtered_LSigmas = filter_result["filtered_LSigmas"]

        if use_smooth:
            cached_mus = filter_result["cached_mus"]
            cached_LSigmas = filter_result["cached_LSigmas"]
            cached_timestamps = filter_result["cached_timestamps"]
            smoothed_mus, smoothed_LSigmas = cont_disc_smooth(
                filter_mus=cached_mus,
                filter_LSigmas=cached_LSigmas,
                filter_timestamps=cached_timestamps,
                dynamics=self.dynamics,
                method=self.integration_method,
            )
            relevant_indices = torch.isin(cached_timestamps, past_times)
            assert torch.allclose(cached_timestamps[relevant_indices], past_times)
            filtered_mus, filtered_LSigmas = (
                smoothed_mus[relevant_indices],
                smoothed_LSigmas[relevant_indices],
            )

        filter_dists = MultivariateNormal(filtered_mus, scale_tril=filtered_LSigmas)
        z_filtered = filter_dists.sample([num_samples])
        # z_filtered.shape = num_samples x T x B x z_dim
        z_reconstruction = []
        y_reconstruction = []
        alpha_reconstruction = []
        for i, t in enumerate(past_times):
            z_t = z_filtered[:, i].reshape(-1, self.z_dim)
            alpha_t = self.dynamics.alpha_net(z_t)
            y_t = self.emit(z_t)
            z_reconstruction.append(z_t)
            alpha_reconstruction.append(alpha_t)
            y_reconstruction.append(y_t)
        z_reconstruction = torch.stack(z_reconstruction)
        alpha_reconstruction = torch.stack(alpha_reconstruction)
        y_reconstruction = torch.stack(y_reconstruction)

        z_reconstruction = z_reconstruction.view(
            past_times.shape[0], num_samples, B, self.z_dim
        )
        alpha_reconstruction = alpha_reconstruction.view(
            past_times.shape[0], num_samples, B, self.K
        )
        y_reconstruction = y_reconstruction.view(
            past_times.shape[0], num_samples, B, self.y_dim
        )
        z_reconstruction = z_reconstruction.permute(1, 2, 0, 3)
        alpha_reconstruction = alpha_reconstruction.permute(1, 2, 0, 3)
        y_reconstruction = y_reconstruction.permute(1, 2, 0, 3)

        (mu, LSigma) = filter_result["last_filtered_dist"]
        future_times = torch.cat([past_times[-1:], future_times], 0)
        mu_t = mu.repeat(num_samples, 1)
        LSigma_t = LSigma.repeat(num_samples, 1, 1)

        y_forecast = []
        alpha_forecast = []
        z_forecast = []
        for t1, t2 in zip(future_times[:-1], future_times[1:]):
            mu_t2, LSigma_t2, _ = self.predict_step(
                mu=mu_t,
                LSigma=LSigma_t,
                dynamics=self.dynamics,
                t0=t1.item(),
                t1=t2.item(),
                step_size=self.integration_step_size,
                method=self.integration_method,
            )

            pred_dist = MultivariateNormal(mu_t2, scale_tril=LSigma_t2)
            z_t2 = pred_dist.mean if no_state_sampling else pred_dist.sample()
            alpha_t2 = self.dynamics.alpha_net(z_t2)
            y_t2 = self.emit(z_t2)
            z_forecast.append(z_t2)
            alpha_forecast.append(alpha_t2)
            y_forecast.append(y_t2)
            mu_t = mu_t2
            LSigma_t = LSigma_t2

        z_forecast = torch.stack(z_forecast)
        alpha_forecast = torch.stack(alpha_forecast)
        y_forecast = torch.stack(y_forecast)
        z_forecast = z_forecast.view(
            future_times.shape[0] - 1, num_samples, B, self.z_dim
        )
        alpha_forecast = alpha_forecast.view(
            future_times.shape[0] - 1, num_samples, B, self.K
        )
        y_forecast = y_forecast.view(
            future_times.shape[0] - 1, num_samples, B, self.y_dim
        )
        z_forecast = z_forecast.permute(1, 2, 0, 3)
        alpha_forecast = alpha_forecast.permute(1, 2, 0, 3)
        y_forecast = y_forecast.permute(1, 2, 0, 3)

        return dict(
            reconstruction=y_reconstruction,
            forecast=y_forecast,
            z_reconstruction=z_reconstruction,
            z_forecast=z_forecast,
            alpha_reconstruction=alpha_reconstruction,
            alpha_forecast=alpha_forecast,
        )


class BaseNL(nn.Module, Base):
    """Base continuous-discrete non-linear state space model.

    Parameters
    ----------
    z_dim
        The dimension of latent state z
    y_dim
        The dimension of observation y
    u_dim
        The dimension of control input u
    f, optional
        The dynamics/drift function f(z), by default None
    gs, optional
        The list of diffusion functions g(z), one for each z_dim, by default None
    B, optional
        The linear control matrix B, by default None
    H, optional
        The observation matrix H, by default None
    R, optional
        The observation covariance matrix R, by default None
    mu0, optional
        The mean of the initial state distribution, by default None
    Sigma0, optional
        The covariance of the initial state distribution, by default None
    integration_method, optional
        The ODE integration method, should be one of "euler" or "rk4", by default "rk4"
    integration_step_size, optional
        The ODE integration step size, by default 0.1
    sporadic, optional
        A flag to indicate whether the dataset is sporadic,
        i.e., with values missing in both time and feature dimensions, by default False
    """

    def __init__(
        self,
        z_dim: int,
        y_dim: int,
        u_dim: int,
        f: nn.Module,
        gs: Optional[List[nn.Module]] = None,
        B: Optional[Tensor] = None,
        H: Optional[Tensor] = None,
        R: Optional[Tensor] = None,
        mu0: Optional[Tensor] = None,
        Sigma0: Optional[Tensor] = None,
        integration_method: str = "rk4",
        integration_step_size: float = 0.1,
        sporadic: bool = False,
    ):
        super().__init__()

        self.z_dim = z_dim
        self.y_dim = y_dim
        self.u_dim = u_dim
        self.integration_method = integration_method
        self.integration_step_size = integration_step_size
        self.sporadic = sporadic

        self.mu0 = nn.Parameter(
            mu0
            if mu0 is not None
            else nn.init.normal_(
                torch.empty(
                    z_dim,
                ),
                std=0.01,
            )
        )  # shared
        self.uSigma0 = nn.Parameter(
            torch.log(Sigma0)
            if Sigma0 is not None
            else torch.zeros(
                z_dim,
            )
        )  # shared

        self.dynamics = ContinuousNL(z_dim=z_dim, u_dim=u_dim, f=f, gs=gs, B=B)

        # TODO: Fix initialization of params
        # If passed as args, they should be fixed
        if H is not None:
            self.register_buffer("H", H)
        else:
            self.H = nn.Parameter(nn.init.xavier_uniform_(torch.empty(y_dim, z_dim)))
        self.uR = nn.Parameter(
            torch.log(R)
            if R is not None
            else torch.zeros(
                y_dim,
            )
        )  # shared
        self.register_buffer("I", torch.eye(z_dim))

    @property
    def Sigma0(self):
        return torch.diag(torch.clamp(torch.exp(self.uSigma0), min=1e-4))

    @property
    def R(self):
        return torch.diag(torch.clamp(torch.exp(self.uR), min=1e-4))

    def forward(self, y: Tensor, mask: Tensor, times: Tensor, **unused_kwargs):
        likelihood = self.filter(y, mask, times)["log_prob"]
        regularizer = torch.tensor(0.0)
        return dict(likelihood=likelihood, regularizer=regularizer)

    def predict_step(
        self,
        mu: Tensor,
        LSigma: Tensor,
        dynamics: Union[ContinuousLTI, ContinuousNL, ContinuousLL],
        t0: float,
        t1: float,
        step_size: float,
        method: str,
        cache_params: bool = False,
        min_step_size: float = 1e-5,
    ):
        return cont_disc_nonlinear_predict(
            mu,
            LSigma,
            dynamics,
            t0,
            t1,
            step_size,
            method,
            cache_params=cache_params,
            min_step_size=min_step_size,
        )

    def update_step(
        self,
        y: Tensor,
        mask: Tensor,
        mu_pred: Tensor,
        LSigma_pred: Tensor,
        H: Tensor,
        R: Tensor,
        sporadic: bool = False,
    ):
        return cont_disc_nonlinear_update(y, mask, mu_pred, LSigma_pred, H, R, sporadic)
