import torch
import torch.nn as nn
import numpy as np

from .components import AuxInferenceModel
from .base import BaseNL, BaseLTI, BaseLL
from ..torch_utils import merge_leading_dims
from ..type import Tensor, Optional, List


class NCDSSM(nn.Module):
    """Base class for NCDSSM models.

    Parameters
    ----------
    aux_inference_net
        The auxiliary inference model parameterized by a neural network
    y_emission_net
        The emission model parameterized by a neural network
    aux_dim
        The dimension of the auxiliary variables
    z_dim
        The dimension of the latent states
    y_dim
        The dimension of the observations
    u_dim
        The dimension of the control inputs
    integration_method, optional
        The integration method, can be one of "euler" or "rk4",
        by default "rk4"
    integration_step_size, optional
        The integration step size, by default 0.1
    sporadic, optional
        A flag to indicate whether the dataset is sporadic,
        i.e., with values missing in both time and feature dimensions,
        by default False
    """

    def __init__(
        self,
        aux_inference_net: nn.Module,
        y_emission_net: nn.Module,
        aux_dim: int,
        z_dim: int,
        y_dim: int,
        u_dim: int,
        integration_method: str = "rk4",
        integration_step_size: float = 0.1,
        sporadic: bool = False,
        **kwargs,
    ):
        super().__init__()
        assert u_dim == 0, "Support for control inputs is not implemented yet"
        self.aux_inference_net = aux_inference_net
        self.y_emission_net = y_emission_net
        self.aux_dim = aux_dim
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.u_dim = u_dim
        self.integration_method = integration_method
        self.integration_step_size = integration_step_size
        self.sporadic = sporadic

    def forward(
        self,
        y: Tensor,
        mask: Tensor,
        times: Tensor,
        num_samples: int = 1,
        deterministic: bool = False,
    ):
        assert (
            y.size() == mask.size() or y.size()[:-1] == mask.size()
        ), "Shapes of y and mask should match!"
        # Currently assumes that t = 0 is in times
        assert times[0] == 0.0, "First timestep should be 0!"
        batch_size = y.size(0)

        aux_samples, aux_entropy, aux_post_log_prob = self.aux_inference_net(
            y,
            mask,
            num_samples=num_samples,
            deterministic=deterministic,
        )
        aux_samples = merge_leading_dims(aux_samples, ndims=2)
        aux_entropy = merge_leading_dims(aux_entropy, ndims=2)
        aux_post_log_prob = merge_leading_dims(aux_post_log_prob, ndims=2)
        # Compute likelihoods
        if mask.ndim < y.ndim:
            mask = mask.unsqueeze(-1)
        repeated_mask = mask.repeat(num_samples, 1, 1)
        emission_dist = self.y_emission_net(aux_samples)
        y_log_likelihood = emission_dist.log_prob(y.repeat(num_samples, 1, 1))
        y_log_likelihood = y_log_likelihood * repeated_mask
        y_log_likelihood = y_log_likelihood.sum(-1).sum(-1)
        filter_result = self.base_ssm.filter(
            aux_samples,
            repeated_mask if self.sporadic else (repeated_mask.sum(-1) > 0).float(),
            times,
        )
        aux_log_likelihood = filter_result["log_prob"]
        aux_entropy = aux_entropy.sum(dim=-1)
        # Compute ELBO
        regularizer = aux_log_likelihood + aux_entropy
        elbo = y_log_likelihood + regularizer
        # Compute IWELBO
        iwelbo = torch.logsumexp(
            y_log_likelihood.view(num_samples, batch_size)
            + aux_log_likelihood.view(num_samples, batch_size)
            - aux_post_log_prob.sum(dim=-1).view(num_samples, batch_size),
            dim=0,
        ) - np.log(num_samples)
        return dict(
            elbo=elbo,
            iwelbo=iwelbo,
            likelihood=y_log_likelihood,
            regularizer=regularizer,
        )

    @torch.no_grad()
    def forecast(
        self,
        y: Tensor,
        mask: Tensor,
        past_times: Tensor,
        future_times: Tensor,
        num_samples: int = 80,
        deterministic: bool = False,
        no_state_sampling: bool = False,
        use_smooth: bool = False,
    ):
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
        deterministic, optional
            Whether to peform deterministic sampling from auxiliary model,
            by default False (not really used)
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
        B, T, _ = y.shape

        aux_samples, _, _ = self.aux_inference_net(
            y,
            mask,
            num_samples=1,
            deterministic=deterministic,
        )
        # aux_samples.shape = num_samples x B x time x aux_dim
        aux_samples = merge_leading_dims(aux_samples, ndims=2)
        if mask.ndim < y.ndim:
            mask = mask.unsqueeze(-1)
        # Generate predictions from the base CDKF
        base_predictions = self.base_ssm.forecast(
            aux_samples,
            mask if self.sporadic else (mask.sum(-1) > 0).float(),
            past_times,
            future_times,
            num_samples=num_samples,
            no_state_sampling=no_state_sampling,
            use_smooth=use_smooth,
        )
        aux_reconstruction = base_predictions["reconstruction"]
        aux_forecast = base_predictions["forecast"]
        z_reconstruction = base_predictions["z_reconstruction"]
        z_forecast = base_predictions["z_forecast"]

        # Decode aux --> y
        reconstruction_emit_dist = self.y_emission_net(
            merge_leading_dims(aux_reconstruction, ndims=2)
        )
        y_reconstruction = reconstruction_emit_dist.sample()
        y_reconstruction = y_reconstruction.view(
            num_samples, B, aux_reconstruction.shape[-2], self.y_dim
        )
        forecast_emit_dist = self.y_emission_net(
            merge_leading_dims(aux_forecast, ndims=2)
        )
        y_forecast = forecast_emit_dist.sample()
        y_forecast = y_forecast.view(num_samples, B, aux_forecast.shape[-2], self.y_dim)

        return dict(
            reconstruction=y_reconstruction,
            forecast=y_forecast,
            z_reconstruction=z_reconstruction,
            z_forecast=z_forecast,
            aux_reconstruction=aux_reconstruction,
            aux_forecast=aux_forecast,
        )


class NCDSSMLTI(NCDSSM):
    """The NCDSSM model with linear time-invariant dynamics.

    Parameters
    ----------
    aux_inference_net
        The auxiliary inference model parameterized by a neural network
    y_emission_net
        The emission model parameterized by a neural network
    aux_dim
        The dimension of the auxiliary variables
    z_dim
        The dimension of the latent states
    y_dim
        The dimension of the observations
    u_dim
        The dimension of the control inputs
    integration_method, optional
        The integration method, can be one of "euler" or "rk4",
        by default "rk4"
    integration_step_size, optional
        The integration step size, by default 0.1
    sporadic, optional
        A flag to indicate whether the dataset is sporadic,
        i.e., with values missing in both time and feature dimensions,
        by default False
    """

    def __init__(
        self,
        aux_inference_net: nn.Module,
        y_emission_net: nn.Module,
        aux_dim: int,
        z_dim: int,
        y_dim: int,
        u_dim: int,
        integration_method: str = "rk4",
        integration_step_size: float = 0.1,
        sporadic: bool = False,
        **kwargs,
    ):
        super().__init__(
            aux_inference_net,
            y_emission_net,
            aux_dim,
            z_dim,
            y_dim,
            u_dim,
            integration_method,
            integration_step_size,
            sporadic,
            **kwargs,
        )
        self.base_ssm = BaseLTI(
            z_dim=z_dim,
            y_dim=aux_dim,
            u_dim=u_dim,
            integration_method=integration_method,
            integration_step_size=integration_step_size,
            sporadic=sporadic,
            **kwargs,
        )


class NCDSSMLL(NCDSSM):
    """The NCDSSM model with locally-linear dynamics.

    Parameters
    ----------
    aux_inference_net
        The auxiliary inference model parameterized by a neural network
    y_emission_net
        The emission model parameterized by a neural network
    aux_dim
        The dimension of the auxiliary variables
    K
        The number of base matrices (i.e., dynamics)
    z_dim
        The dimension of the latent states
    y_dim
        The dimension of the observations
    u_dim
        The dimension of the control inputs
    alpha_net
        A mixing network that takes the state `z` as input
        and outputs the mixing coefficients for the base dynamics
    integration_method, optional
        The integration method, can be one of "euler" or "rk4",
        by default "rk4"
    integration_step_size, optional
        The integration step size, by default 0.1
    sporadic, optional
        A flag to indicate whether the dataset is sporadic,
        i.e., with values missing in both time and feature dimensions,
        by default False
    """

    def __init__(
        self,
        aux_inference_net: AuxInferenceModel,
        y_emission_net: nn.Module,
        aux_dim: int,
        K: int,
        z_dim: int,
        y_dim: int,
        u_dim: int,
        alpha_net: nn.Module,
        integration_method: str = "rk4",
        integration_step_size: float = 0.1,
        sporadic: bool = False,
        **kwargs,
    ):
        super().__init__(
            aux_inference_net,
            y_emission_net,
            aux_dim,
            z_dim,
            y_dim,
            u_dim,
            integration_method,
            integration_step_size,
            sporadic,
            **kwargs,
        )
        self.base_ssm = BaseLL(
            K=K,
            z_dim=z_dim,
            y_dim=aux_dim,
            u_dim=u_dim,
            alpha_net=alpha_net,
            integration_method=integration_method,
            integration_step_size=integration_step_size,
            sporadic=sporadic,
            **kwargs,
        )

    @torch.no_grad()
    def forecast(
        self,
        y: Tensor,
        mask: Tensor,
        past_times: Tensor,
        future_times: Tensor,
        num_samples: int = 80,
        deterministic: bool = False,
        no_state_sampling: bool = False,
        use_smooth: bool = False,
    ):
        B, T, _ = y.shape

        aux_samples, _, _ = self.aux_inference_net(
            y,
            mask,
            num_samples=1,
            deterministic=deterministic,
        )
        # aux_samples.shape = num_samples x B x time x aux_dim
        aux_samples = merge_leading_dims(aux_samples, ndims=2)
        if mask.ndim < y.ndim:
            mask = mask.unsqueeze(-1)
        # Generate predictions from the base CDKF
        base_predictions = self.base_ssm.forecast(
            aux_samples,
            mask if self.sporadic else (mask.sum(-1) > 0).float(),
            past_times,
            future_times,
            num_samples=num_samples,
            no_state_sampling=no_state_sampling,
            use_smooth=use_smooth,
        )
        aux_reconstruction = base_predictions["reconstruction"]
        aux_forecast = base_predictions["forecast"]
        z_reconstruction = base_predictions["z_reconstruction"]
        z_forecast = base_predictions["z_forecast"]
        alpha_reconstruction = base_predictions["alpha_reconstruction"]
        alpha_forecast = base_predictions["alpha_forecast"]

        # Decode aux --> y
        reconstruction_emit_dist = self.y_emission_net(
            merge_leading_dims(aux_reconstruction, ndims=2)
        )
        y_reconstruction = reconstruction_emit_dist.sample()
        y_reconstruction = y_reconstruction.view(
            num_samples, B, aux_reconstruction.shape[-2], self.y_dim
        )
        forecast_emit_dist = self.y_emission_net(
            merge_leading_dims(aux_forecast, ndims=2)
        )
        y_forecast = forecast_emit_dist.sample()
        y_forecast = y_forecast.view(num_samples, B, aux_forecast.shape[-2], self.y_dim)

        return dict(
            reconstruction=y_reconstruction,
            forecast=y_forecast,
            z_reconstruction=z_reconstruction,
            z_forecast=z_forecast,
            alpha_reconstruction=alpha_reconstruction,
            alpha_forecast=alpha_forecast,
            aux_reconstruction=aux_reconstruction,
            aux_forecast=aux_forecast,
        )


class NCDSSMNL(NCDSSM):
    """The NCDSSM model with non-linear dynamics.

    Parameters
    ----------
    aux_inference_net
        The auxiliary inference model parameterized by a neural network
    y_emission_net
        The emission model parameterized by a neural network
    aux_dim
        The dimension of the auxiliary variables
    z_dim
        The dimension of the latent states
    y_dim
        The dimension of the observations
    u_dim
        The dimension of the control inputs
    f, optional
        The dynamics/drift function f(z), by default None
    gs, optional
        The list of diffusion functions g(z), one for each z_dim, by default None
    integration_method, optional
        The integration method, can be one of "euler" or "rk4",
        by default "rk4"
    integration_step_size, optional
        The integration step size, by default 0.1
    sporadic, optional
        A flag to indicate whether the dataset is sporadic,
        i.e., with values missing in both time and feature dimensions,
        by default False
    """

    def __init__(
        self,
        aux_inference_net: AuxInferenceModel,
        y_emission_net: nn.Module,
        aux_dim: int,
        z_dim: int,
        y_dim: int,
        u_dim: int,
        f: nn.Module,
        gs: Optional[List[nn.Module]] = None,
        integration_method: str = "rk4",
        integration_step_size: float = 0.1,
        sporadic: bool = False,
        **kwargs,
    ):
        super().__init__(
            aux_inference_net,
            y_emission_net,
            aux_dim,
            z_dim,
            y_dim,
            u_dim,
            integration_method,
            integration_step_size,
            sporadic,
            **kwargs,
        )
        self.base_ssm = BaseNL(
            z_dim=z_dim,
            y_dim=aux_dim,
            u_dim=u_dim,
            f=f,
            gs=gs,
            integration_method=integration_method,
            integration_step_size=integration_step_size,
            sporadic=sporadic,
            **kwargs,
        )
