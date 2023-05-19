import torch
import torch.nn as nn
import torch.distributions as td

from torch.nn.functional import softplus

from ..functions import inverse_softplus


class GaussianOutput(nn.Module):
    def __init__(
        self,
        network,
        dist_dim,
        use_tied_cov=False,
        use_trainable_cov=True,
        sigma=0.1,
        use_independent=True,
        scale_function="exp",
        min_scale=0.01,
    ):
        super().__init__()
        assert scale_function in {"exp", "softplus"}
        self.scale_function = scale_function
        self.dist_dim = dist_dim
        self.network = network
        self.use_tied_cov = use_tied_cov
        self.use_trainable_cov = use_trainable_cov
        self.use_independent = use_independent
        self.min_scale = min_scale
        if not self.use_trainable_cov:
            self.sigma = sigma
        if self.use_trainable_cov and self.use_tied_cov:
            if scale_function == "softplus":
                init_sigma = inverse_softplus(torch.full([1, dist_dim], sigma))
            else:
                init_sigma = torch.zeros([1, dist_dim])
            self.usigma = nn.Parameter(init_sigma)

    def forward(self, tensor):
        args_tensor = self.network(tensor)
        mean_tensor = args_tensor[..., : self.dist_dim]
        if self.use_trainable_cov:
            if self.use_tied_cov:
                if self.scale_function == "softplus":
                    scale_tensor = softplus(self.usigma)
                else:
                    scale_tensor = torch.exp(self.usigma)
                scale_tensor = torch.clamp(scale_tensor, min=self.min_scale)
                out_dist = td.normal.Normal(mean_tensor, scale_tensor)
                if self.use_independent:
                    out_dist = td.independent.Independent(out_dist, 1)
            else:
                if self.scale_function == "softplus":
                    scale_tensor = softplus(args_tensor[..., self.dist_dim :])
                else:
                    scale_tensor = torch.exp(args_tensor[..., self.dist_dim :])
                scale_tensor = torch.clamp(scale_tensor, min=self.min_scale)
                out_dist = td.normal.Normal(mean_tensor, scale_tensor)
                if self.use_independent:
                    out_dist = td.independent.Independent(out_dist, 1)
        else:
            out_dist = td.normal.Normal(mean_tensor, self.sigma)
            if self.use_independent:
                out_dist = td.independent.Independent(out_dist, 1)
        return out_dist


class BernoulliOutput(nn.Module):
    def __init__(self, network, dist_dim, use_indepedent=True):
        super().__init__()
        self.network = network
        self.dist_dim = dist_dim
        self.use_indepedent = use_indepedent

    def forward(self, x):
        h = self.network(x)
        assert h.shape[-1] == self.dist_dim
        dist = td.Bernoulli(logits=h)
        if self.use_indepedent:
            dist = td.Independent(dist, 1)
        return dist


class AuxInferenceModel(nn.Module):
    def __init__(
        self,
        base_network: nn.Module,
        dist_network: nn.Module,
        aux_dim: int,
        concat_mask: bool = False,
    ):
        super().__init__()
        self.aux_dim = aux_dim
        self.base_network = base_network
        self.dist_network = dist_network
        self.concat_mask = concat_mask

    def forward(
        self,
        y: torch.Tensor,
        mask: torch.Tensor,
        num_samples: int = 1,
        deterministic: bool = False,
    ):
        assert y.ndim >= 3
        if self.concat_mask:
            assert y.size() == mask.size()
            # Concatenate mask to feature dim
            y = torch.cat([y, mask], dim=-1)
            # Convert feature mask to at least one feature presence mask
            # i.e. mask = 1 if at least one feature is present at that time
            # step.
            mask = (mask.sum(dim=-1) > 0).float()

        sporadic = False
        if y.ndim > mask.ndim:
            mask = mask.unsqueeze(-1)
        else:
            sporadic = True

        h = self.base_network(y)
        h = h * mask
        dist = self.dist_network(h)
        if deterministic:
            aux_samples = dist.mean.unsqueeze(0)
            one_args = (1,) * dist.mean.ndim
            aux_samples = aux_samples.repeat(num_samples, *one_args)
        else:
            aux_samples = dist.rsample((num_samples,))

        aux_samples = aux_samples * mask
        if sporadic:
            aux_entropy = (dist.base_dist.entropy() * mask).sum(dim=-1)
            aux_log_prob = (dist.base_dist.log_prob(aux_samples) * mask).sum(dim=-1)
        else:
            aux_entropy = dist.entropy()
            aux_log_prob = dist.log_prob(aux_samples)
            aux_entropy = aux_entropy * mask.squeeze(-1)
            aux_log_prob = aux_log_prob * mask.squeeze(-1)

        aux_entropy = aux_entropy[None].repeat(num_samples, 1, 1)
        return (
            aux_samples,  # .shape = N, B, T, aux_dim
            aux_entropy,  # .shape = N, B, T
            aux_log_prob,  # .shape = N, B, T
        )
