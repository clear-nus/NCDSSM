import torch
import torch.nn as nn

from ..modules import Lambda

from ..functions import batch_jacobian, bm_t, mbvp, bmbvp
from ..type import Tuple, Tensor, List


class ContinuousLTI(nn.Module):
    def __init__(
        self,
        z_dim: int,
        u_dim: int,
        F: Tensor,
        B: Tensor,
        uQ: Tensor,
    ):
        super().__init__()
        self.z_dim = z_dim
        if u_dim > 0:
            if B is not None:
                self.register_buffer("B", B)
            else:
                self.B = nn.Parameter(
                    nn.init.xavier_uniform_(torch.empty(z_dim, u_dim))
                )

        self.F = nn.Parameter(F)
        self.uQ = nn.Parameter(uQ)

    @property
    def Q(self):
        return torch.diag(torch.clamp(torch.exp(self.uQ), min=1e-4))

    def forward(
        self,
        t: Tensor,
        m_and_P: Tuple[Tensor, Tensor],
    ):
        """Dynamics for mean and covariance.

        Args:
            t (Tensor): A 1-D tensor of times.
            m_and_P (Tuple[Tensor, Tensor]): A tuple of
                mean and covariance.

        Returns:
            Tuple: Tuple of "velocity" of mean and covariance.
        """

        m, P = m_and_P
        # Dynamics for mean
        velocity_m = mbvp(self.F, m)

        # Dynamics for Covariance
        F = self.F[None]
        Q = self.Q[None]
        velocity_P = F @ P + P @ bm_t(F) + Q
        return velocity_m, velocity_P


class ContinuousLL(nn.Module):
    def __init__(
        self,
        z_dim: int,
        u_dim: int,
        K: int,
        F: Tensor,
        B: Tensor,
        uQ: Tensor,
        alpha_net: nn.Module,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.u_dim = u_dim
        self.K = K
        if u_dim > 0:
            if B is not None:
                self.register_buffer("B", B)
            else:
                self.B = nn.Parameter(
                    nn.init.xavier_uniform_(torch.empty(K, z_dim, u_dim))
                )
        self.F = nn.Parameter(F)
        self.uQ = nn.Parameter(uQ)
        self.alpha_net = alpha_net

    @property
    def Q(self):
        return torch.diag(torch.clamp(torch.exp(self.uQ), min=1e-4))

    def forward(self, t: Tensor, m_and_P: Tuple[Tensor, Tensor]):
        """Dynamics for mean and covariance.

        Args:
            t (Tensor): A 1-D tensor of times.
            m_and_P (Tuple[Tensor, Tensor]):
                A tuple of mean and covariance.

        Returns:
            Tuple: Tuple of "velocity" of mean and covariance.
        """

        m, P = m_and_P
        # Linear combination of base matrices
        alpha = self.alpha_net(m)
        F = torch.einsum("bk, knm -> bnm", alpha, self.F)
        # F.shape = B x z_dim x z_dim
        # m.shape = B x z_dim
        # Dynamics for mean
        velocity_m = bmbvp(F, m)

        # Dynamics for Covariance
        Q = self.Q[None]
        velocity_P = F @ P + P @ bm_t(F) + Q
        return velocity_m, velocity_P


class ContinuousNL(nn.Module):
    def __init__(
        self,
        z_dim: int,
        u_dim: int,
        f: nn.Module,
        gs: List[nn.Module],
        B: Tensor,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.f = f

        if u_dim > 0:
            if B is not None:
                self.register_buffer("B", B)
            else:
                self.B = nn.Parameter(
                    nn.init.xavier_uniform_(torch.empty(z_dim, u_dim))
                )

        self.fixed_diffusion = gs is None
        if gs is None:
            gs = [
                nn.Sequential(
                    nn.Linear(1, 1, bias=False), Lambda(lambda x: torch.exp(x))
                )
                for _ in range(z_dim)
            ]
            with torch.no_grad():
                for g in gs:
                    g[0].weight.data.zero_()
        self.gs = nn.ModuleList(gs)

    def jac_f(self, z: Tensor) -> Tensor:
        J = batch_jacobian(self.f, z, create_graph=True, vectorize=True)
        return J

    def GQGt(self, z):
        # Assuming Q = I, so GQGt = GGt
        # For fixed diffusion the normalizer is Exp(), see __init__().
        # For non linear diffusion, it is assumed that
        # the diffusion networks normalize the scale to R+.

        if self.fixed_diffusion:
            z = torch.ones_like(z)
        zs = torch.split(z, split_size_or_sections=1, dim=-1)
        diag_G = torch.cat([g_i(z_i) for (g_i, z_i) in zip(self.gs, zs)], dim=-1)
        diag_Gsq = diag_G * diag_G
        return torch.diag_embed(torch.clamp(diag_Gsq, min=1e-4))

    def forward(self, t: Tensor, m_P: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        """Dynamics for mean and covariance.

        Args:
            t (Tensor): A 1-D tensor of times.
            m_P (Tuple[Tensor, Tensor, Tensor]):
                A tuple of mean, and covariance.

        Returns:
            Tuple: Tuple of "velocity" of mean and covariance.
        """

        m, P = m_P
        # Dynamics for mean
        velocity_m = self.f(m)

        # Dynamics for Covariance
        GQGt = self.GQGt(m)
        J_f = self.jac_f(m)
        velocity_P = J_f @ P + P @ bm_t(J_f) + GQGt
        return velocity_m, velocity_P
