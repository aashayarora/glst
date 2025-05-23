from __future__ import annotations

import torch
from torch import nn
from torch import Tensor as T

"""
Taken from https://github.com/object-condensation/object_condensation/blob/main/src/object_condensation/pytorch/losses.py
"""

class CondensationLoss(nn.Module):
    def __init__(
        self,
        q_min: float = 0.1,
        noise_threshold: int = 0,
        max_n_rep: int = 0 ):
        super().__init__()
        self.q_min = q_min
        self.noise_threshold = noise_threshold
        self.max_n_rep = max_n_rep

    @staticmethod
    @torch.compile
    def condensation_loss_tiger(
        *,
        beta: T,
        x: T,
        object_id: T,
        weights: T | None = None,
        q_min: float,
        noise_threshold: int = 0,
        max_n_rep: int = 0 ) -> dict[str, T]:
        """Condensation losses

        Args:
            beta: Condensation likelihoods
            x: Clustering coordinates
            object_id: Labels for objects.
            weights: Weights per hit, multiplied to attractive/repulsive potentials
            q_min: Minimal charge
            noise_threshold: Threshold for noise hits. Hits with ``object_id <= noise_thld``
                are considered to be noise
            max_n_rep: Maximum number of elements to consider for repulsive loss.
                Set to 0 to disable.
            torch_compile: Torch compile loss function. This is recommended, but might not
                work in older pytorch version or in cutting edge python.

        Returns:
            Dictionary of scalar tensors; see readme.
            `n_rep`: Number of repulsive elements (before sampling).
        """
        # To protect against nan in divisions
        eps = 1e-9

        # x: n_nodes x n_outdim
        not_noise = object_id > noise_threshold
        unique_oids = torch.unique(object_id[not_noise])
        assert len(unique_oids) > 0, "No particles found, cannot evaluate loss"
        # n_nodes x n_pids
        # The nodes in every column correspond to the hits of a single particle and
        # should attract each other
        attractive_mask = object_id.view(-1, 1) == unique_oids.view(1, -1)

        q = torch.arctanh(beta) ** 2 + q_min
        assert not torch.isnan(q).any(), "q contains NaNs"
        # n_objs
        alphas = torch.argmax(q.view(-1, 1) * attractive_mask, dim=0)

        # _j means indexed by hits
        # _k means indexed by objects

        # n_objs x n_outdim
        x_k = x[alphas]
        # 1 x n_objs
        q_k = q[alphas].view(1, -1)

        dist_j_k = torch.cdist(x, x_k)

        qw_j_k = weights.view(-1, 1) * q.view(-1, 1) * q_k

        att_norm_k = (attractive_mask.sum(dim=0) + eps) * len(unique_oids)
        qw_att = (qw_j_k / att_norm_k)[attractive_mask]

        # Attractive potential/loss
        v_att = (qw_att * torch.square(dist_j_k[attractive_mask])).sum()

        repulsive_mask = (~attractive_mask) & (dist_j_k < 1)
        n_rep_k = (~attractive_mask).sum(dim=0)
        n_rep = repulsive_mask.sum()
        # Don't normalize to repulsive_mask, it includes the dist < 1 count,
        # (less points within the radius 1 ball should translate to lower loss)
        rep_norm = (n_rep_k + eps) * len(unique_oids)
        if n_rep > max_n_rep > 0:
            sampling_freq = max_n_rep / n_rep
            sampling_mask = (
                torch.rand_like(repulsive_mask, dtype=torch.float16) < sampling_freq
            )
            repulsive_mask &= sampling_mask
            rep_norm *= sampling_freq
        qw_rep = (qw_j_k / rep_norm)[repulsive_mask]
        v_rep = (qw_rep * (1 - dist_j_k[repulsive_mask])).sum()

        l_coward = torch.mean(1 - beta[alphas])
        not_noise = not_noise.view(-1)
        l_noise = torch.mean(beta[~not_noise])

        return {
            "attractive": v_att,
            "repulsive": v_rep,
            "coward": l_coward,
            "noise": l_noise
        }

    def forward(
        self,
        beta: T,
        x: T,
        object_id: T,
        weights: T | None = None) -> dict[str, T]:
        
        if weights is None:
            weights = torch.ones_like(beta)
        
        return self.condensation_loss_tiger(
            beta=beta,
            x=x,
            object_id=object_id,
            weights=weights,
            q_min=self.q_min,
            noise_threshold=self.noise_threshold,
            max_n_rep=self.max_n_rep,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} (q_min={self.q_min}, noise_threshold={self.noise_threshold}, max_n_rep={self.max_n_rep})"