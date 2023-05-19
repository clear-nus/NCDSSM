import torch
import warnings
from .type import Tensor, Tuple


def inverse_softplus(tensor: Tensor):
    return tensor + torch.log(1 - torch.exp(-tensor))


def bm_t(tensor: Tensor):
    """Batched matrix transpose. Swaps the last two axes.

    Args:
        tensor (torch.Tensor): N-D tensor.

    Returns:
        torch.Tensor: tensor with last two dims swapped.
    """
    return torch.transpose(tensor, -2, -1)


def mbvp(matrix: Tensor, batched_vectors: Tensor):
    """Compute matrix-batched vector product.

    Args:
        matrix (Tensor): 2D tensor.
        batched_vectors (Tensor): 2D tensor.

    Returns:
        Tensor: The matrix-batched vector product.
    """
    assert (
        matrix.ndim == 2 and batched_vectors.ndim == 2
    ), "matrix and batched_vectors should be 2D tensors!"
    m, n = matrix.size()
    b, d = batched_vectors.size()
    assert n == d, f"dim 2 of matrix ({n}) should match dim 2 of batched_vectors ({d})!"
    batched_matrix = matrix.unsqueeze(0)
    return bmbvp(batched_matrix, batched_vectors)


def bmbvp(batched_matrix: Tensor, batched_vectors: Tensor):
    """Compute batched matrix-batched vector product.

    Args:
        batched_matrix (Tensor): 3D tensor.
        batched_vectors (Tensor): 2D tensor.

    Returns:
        Tensor: The batched matrix-batched vector product.
    """
    assert batched_matrix.ndim == 3 and batched_vectors.ndim == 2, (
        "batched_matrix and batched_vectors should "
        "be 3D and 2D tensors, respectively!"
    )
    bm, m, n = batched_matrix.size()
    bv, d = batched_vectors.size()
    assert bm == bv or bm == 1 or bv == 1, (
        f"Matrix ({bm}) and vector ({bv}) batch dim should match " "or be singletons!"
    )
    assert n == d, (
        f"dim 3 of matrix ({n}) should match dim 2 " f"of batched_vectors ({d})!"
    )
    batched_vectors = batched_vectors.unsqueeze(-1)
    output = torch.matmul(batched_matrix, batched_vectors)
    output = output.squeeze(-1)
    return output


def linear_interpolation(t0, x0, t1, x1, t):
    assert t0 <= t <= t1, f"Incorrect time order: t0={t0}, t={t}, t1={t1}!"
    x = (t1 - t) / (t1 - t0) * x0 + (t - t0) / (t1 - t0) * x1
    return x


def symmetrize(matrix: Tensor) -> Tensor:
    """Symmetrize a matrix A by (A + A^T) / 2.

    Args:
        matrix (Tensor): An N-D tensor with N > 2.

    Returns:
        Tensor: The symmetrized matrices.
    """
    return 0.5 * (matrix + matrix.transpose(-1, -2))


def regularize_matrix(
    batched_matrix, regularization=1e-6, relative_regularization=True
):
    batch_size = batched_matrix.size(0)
    batched_identity = (
        torch.eye(
            batched_matrix.size(-1),
            dtype=batched_matrix.dtype,
            device=batched_matrix.device,
        )
        .unsqueeze(0)
        .repeat(batch_size, 1, 1)
    )

    if relative_regularization:
        diag_mean = (
            torch.einsum("jii->ji", batched_matrix)
            .mean(dim=-1)
            .view(batch_size, 1, 1)
            .detach()
        )
        regularized_matrix = (
            batched_matrix + regularization * diag_mean * batched_identity
        )
    else:
        regularized_matrix = batched_matrix + regularization * batched_identity
    return regularized_matrix


def make_positive_definite(
    batched_matrix,
    regularization=1e-4,
    max_regularization=1,
    step_factor=10,
    warn=True,
    relative_regularization=True,
):
    assert batched_matrix.ndim == 3
    if regularization > max_regularization:
        raise ValueError(
            "Attempted to regularize beyond max_regularization:" f"{max_regularization}"
        )
    if warn:
        warnings.warn(f"Regularizing matrix with factor: {regularization}!")
    regularized_matrix = regularize_matrix(
        batched_matrix,
        regularization=regularization,
        relative_regularization=relative_regularization,
    )
    is_pd = torch.all(torch.linalg.cholesky_ex(regularized_matrix).info.eq(0))
    if is_pd:
        return regularized_matrix
    else:
        return make_positive_definite(
            batched_matrix,
            regularization=regularization * step_factor,
            max_regularization=max_regularization,
            step_factor=step_factor,
            warn=warn,
            relative_regularization=relative_regularization,
        )


def cholesky(batched_matrix: Tensor):
    """Cholesky decomposition which attempts to regularize the
    matrix diagonal with some salt if the decomposition fails.

    Args:
        batched_matrix (Tensor): A 3-D tensor.

    Returns:
        Tensor: The cholesky factor.
    """
    try:
        cholesky_factor = torch.linalg.cholesky(batched_matrix)
    except RuntimeError:
        regularized_matrix = make_positive_definite(batched_matrix)
        cholesky_factor = torch.linalg.cholesky(regularized_matrix)
    return cholesky_factor


def qr(batched_matrix, positive_diag=True) -> Tuple[Tensor, Tensor]:
    """QR decomposition with the option of making the diagonal elements
    of R positive.

    Args:
        batched_matrix (Tensor): An N-D tensor with N > 2.
        positive_diag (bool, optional): Whether to make the diagonal elements
            of R positive. Defaults to True.

    Returns:
        Tuple[Tensor, Tensor]: The Q and R.
    """
    Q, R = torch.linalg.qr(batched_matrix)
    if positive_diag:
        # The QR decomposition returned by pytorch does not guarantee
        # that the matrix R has positive diagonal elements.
        # To ensure that, we first construct a signature matrix S
        # which is a diagonal matrix with diagonal elements +1 or -1
        # depending on the sign of the corresponding diagonal element in R.
        # If we premultiply R with S, we ensure that the diagonal is positive;
        # however, the corresponding Q needs to be changed now to ensure that
        # that we get back the original matrix when we multiply the new Q and
        # the new R. This can be done by postmultiplying Q with S because
        # then we get (Q @ S) @ (S @ R) = Q @ R as S @ S = I, where (Q @ S)
        # and (S @ R) are the new Q and new R respectively.
        R_diag_sign = torch.sgn(torch.diagonal(R, dim1=-2, dim2=-1))
        S = torch.diag_embed(R_diag_sign, dim1=-2, dim2=-1)
        Q = Q @ S
        R = S @ R
    return Q, R


def bm_diag_positive(A):
    dim = A.size(-1)
    diag = torch.abs(torch.diagonal(A, dim1=-2, dim2=-1))
    mask = torch.eye(dim, device=A.device)[None]
    A = mask * torch.diag_embed(diag) + (1 - mask) * A
    return A


def sum_mat_sqrts(sqrt_A, sqrt_B):
    tmp = bm_t(torch.cat([sqrt_A, sqrt_B], dim=-1))
    _, Rtmp = qr(tmp)
    return bm_t(Rtmp)


def batch_jacobian(func, x, create_graph=False, vectorize=False):
    def _func_sum(x):
        return func(x).sum(dim=0)

    return torch.autograd.functional.jacobian(
        _func_sum, x, create_graph=create_graph, vectorize=vectorize
    ).permute(1, 0, 2)
