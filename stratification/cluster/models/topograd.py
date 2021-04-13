from __future__ import annotations
import logging
import math
from typing import Any, Literal
import warnings

import matplotlib.pyplot as plt
from numba import jit
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim
from tqdm import tqdm

from faiss import IndexFlatL2

from .topograd_orig import topoclustergrad


__all__ = [
    "rbf",
    "compute_rips",
    "compute_density_map",
    "TopoGradFn",
    "TopoGradLoss",
    "TopoGradCluster",
]


def rbf(x: np.ndarray, y: np.ndarray, scale: float, axis: int = -1) -> np.ndarray:
    "Compute the distance between two vectors using an RBF kernel."
    return np.exp(-np.linalg.norm(x - y, axis=axis) ** 2 / scale)


def compute_rips(pc: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """"Compute the delta-Rips Graph."""
    pc = pc.astype(np.float32)
    cpuindex = IndexFlatL2(pc.shape[1])
    cpuindex.add(pc)

    return cpuindex.search(pc, k)


def compute_density_map(x: np.ndarray, k: int, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute the k-nearest neighbours kernel density estimate."""
    x = x.astype(np.float32)
    index = IndexFlatL2(x.shape[1])
    index.add(x)
    values, indexes = index.search(x, k)
    result = np.sum(np.exp(-values / scale), axis=1) / (k * scale)
    return result / max(result), indexes


@jit(nopython=True)
def find_entry_idx_by_point(entries: dict[int, list[int]], point_idx: int) -> np.int64:
    for index, entry in entries.items():
        for i in entry:
            if i == point_idx:
                return np.int64(index)
    return np.int64(point_idx)


@jit(nopython=True)
def merge(
    density_map: np.ndarray,
    entries: dict[int, list[int]],
    ref_idx: int,
    e_up: int,
    us_idxs: list[int],
    threshold: float,
) -> tuple[dict[int, list[int]], np.ndarray]:
    pers_pairs = np.array([[-1, -1]])

    for idx in us_idxs:
        entry_idx = find_entry_idx_by_point(entries, idx)

        if e_up != entry_idx:
            persistence = density_map[entry_idx] - density_map[ref_idx]

            if persistence < threshold:
                entries[e_up] = np.append(entries[e_up], entries[entry_idx])
                entries.pop(entry_idx)

            pers_pairs = np.append(pers_pairs, np.array([[entry_idx, ref_idx]]), axis=0)

    return entries, pers_pairs


@jit(nopython=True)
def cluster(
    density_map: np.ndarray, rips_idxs: np.ndarray, threshold: float
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    N = density_map.shape[0]
    pers_pairs = np.array([[-1, -1]])
    #  initialize the union-find data-structure with the final index pointing only to itself

    entries = {N - 1: np.array([N - 1])}

    for i in np.arange(N - 2, -1, -1):
        nbr_idxs = rips_idxs[i]
        # compute the upper star Si = {(i, j1), · · · , (i, jk)} of vertex i in R_δ(L);
        us_idxs = nbr_idxs[nbr_idxs > i]
        # check whether vertex i is a local maximum of f within R_δ
        if us_idxs.size == 0:
            entries[i] = np.array([i])  #  create an entry for the local maximum
        else:
            # approximate the gradient of the underlying probability density function by connecting
            # i to its neighbour in the graph with the highest function value
            g_i = np.max(us_idxs)  #  find the maximum index
            # Attach vertex i to the tree t containing g(i)
            e_up = find_entry_idx_by_point(entries, g_i)
            entries[e_up] = np.append(entries[e_up], i)
            entries, pers_pairs_i = merge(
                density_map=density_map,
                entries=entries,
                ref_idx=i,
                e_up=e_up,
                us_idxs=us_idxs,
                threshold=threshold,
            )
            if len(pers_pairs_i) > 1:
                pers_pairs = np.append(pers_pairs, pers_pairs_i, axis=0)

    return entries, pers_pairs


def tomato(
    pc: np.ndarray, k_kde: int, k_rips: int, scale: float, threshold: float
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Topological mode analysis tool (Chazal et al., 2013).

    Args:
        pc (np.ndarray): The point-cloud to be clustered
        k_kde (int): Number of neighbors to use when computing the density map
        k_rips (int): Number of neighbors to use when computing the Rips graph.
        scale (float): Bandwidth of the kernel used when computing the density map.
        threshold (float): Thresholding parameter (tau in the paper)

    Returns:
        Tuple[Dict[int, np.ndarray[np.int]], np.ndarray[np.float32]]: Clusters and their pers_pairs
    """
    pc = pc.astype(float)
    #  Compute the k-NN KDE
    density_map, _ = compute_density_map(pc, k_kde, scale)
    density_map = density_map.astype(np.float32)
    sorted_idxs = np.argsort(density_map)
    density_map_sorted = density_map[sorted_idxs]
    pc = pc[sorted_idxs]
    _, rips_idxs = compute_rips(pc, k=k_rips)
    entries, pers_pairs = cluster(density_map_sorted, rips_idxs, threshold=threshold)
    if threshold == 1:
        see = np.array([elem for elem in pers_pairs if (elem != np.array([-1, -1])).any()])
        result = []
        if see.size > 0:
            for i in np.unique(see[:, 0]):
                result.append(
                    [
                        see[np.where(see[:, 0] == i)[0]][0, 0],
                        max(see[np.where(see[:, 0] == i)[0]][:, 1]),
                    ]
                )
            result = np.array(result)
            for key, value in entries.items():
                entries[key] = sorted_idxs[value]
        else:
            warnings.warn("Clustering unsuccessful; consider expanding the VRC neighbourhood.")
        return entries, density_map_sorted[result]
    else:
        for key, value in entries.items():
            entries[key] = sorted_idxs[value]
        return entries, np.array([[0, 0]])


def newI1(I1, sortrule):
    newI1 = I1[sortrule]
    for i in range(newI1.shape[0]):
        for j in range(newI1.shape[1]):
            newI1[i, j] = np.where(sortrule == newI1[i, j])[0][0]
    return newI1


def topograd_loss(pc: Tensor, k_kde: int, k_rips: int, scale: float, destnum: int) -> Tensor:
    kde_dists, _ = compute_density_map(pc, k_kde, scale)

    sorted_idxs = torch.argsort(kde_dists, descending=False)
    kde_dists_sorted = kde_dists[sorted_idxs]
    pc_sorted = pc[sorted_idxs]

    rips_idxs = compute_rips(pc_sorted, k_rips)
    _, pers_pairs = cluster(
        density_map=kde_dists_sorted.detach().cpu().numpy(),
        rips_idxs=rips_idxs.cpu().numpy(),
        threshold=1.0,
    )

    pers_pairs = torch.as_tensor(pers_pairs, device=pc.device)
    seen = pers_pairs[~torch.all(pers_pairs == -1, dim=1)]

    pd_pairs = []
    for i in torch.unique(seen[:, 0]):
        pd_pairs.append(
            [
                seen[torch.where(seen[:, 0] == i)[0]][0, 0],
                max(seen[torch.where(seen[:, 0] == i)[0]][:, 1]),
            ]
        )
    if not pd_pairs:
        print(
            "FIltering failed to yield any persistence pairs required for computation of "
            "the topological loss. Returning 0 instead."
        )
        return pc.new_zeros(())
    pd_pairs = torch.as_tensor(pd_pairs, device=pc.device)
    oripd = kde_dists_sorted[pd_pairs]
    pers_idxs_sorted = torch.argsort(oripd[:, 0] - oripd[:, 1])

    changing = pers_idxs_sorted[:-destnum]
    nochanging = pers_idxs_sorted[-destnum:-1]

    biggest = oripd[pers_idxs_sorted[-1]]
    dest = torch.as_tensor([biggest[0], biggest[1]], device=pc.device)
    changepairs = pd_pairs[changing]
    nochangepairs = pd_pairs[nochanging]
    pd11 = kde_dists_sorted[changepairs]

    weakdist = torch.sum(pd11[:, 0] - pd11[:, 1]) / math.sqrt(2)
    strongdist = torch.sum(torch.norm(kde_dists_sorted[nochangepairs] - dest, dim=1))
    return weakdist + strongdist


class TopoGradLoss(nn.Module):
    def __init__(self, k_kde: int, k_rips: int, scale: float, destnum: int) -> None:
        super().__init__()
        self.k_kde = k_kde
        self.k_rips = k_rips
        self.scale = scale
        self.destnum = destnum

    def forward(self, x: Tensor) -> Tensor:
        return topograd_loss(
            pc=x, k_kde=self.k_kde, k_rips=self.k_rips, scale=self.scale, destnum=self.destnum
        )
        # return topoclustergrad.apply(x, self.k_kde, self.k_rips, self.scale, self.destnum)


class TopoGradCluster:
    labels: np.ndarray
    pers_pairs: np.ndarray
    split_indces: np.ndarray

    def __init__(
        self,
        destnum: int,
        k_kde: int = 10,
        k_rips: int = 10,
        scale: float = 0.5,
        merge_threshold: float = 1,
        iters: int = 100,
        lr: float = 1.0e-3,
        optimizer_cls=torch.optim.AdamW,
        **optimizer_kwargs: dict[str, Any],
    ) -> None:
        super().__init__()
        self.k_kde = k_kde
        self.k_rips = k_rips
        self.scale = scale
        self.destnum = destnum
        self.merge_threshold = merge_threshold
        self.lr = lr
        self.iters = iters
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs
        self._loss_fn = TopoGradLoss(k_kde=k_kde, k_rips=k_rips, scale=scale, destnum=destnum)
        self.logger = logging.getLogger(__name__)

    def plot(self) -> plt.Figure:
        fig, ax = plt.subplots(dpi=100)
        ax.scatter(self.pers_pairs[:, 0], self.pers_pairs[:, 1], s=15, c="orange")  # type: ignore[arg-type]
        ax.plot(np.array([0, 1]), np.array([0, 1]), c="black", alpha=0.6)  # type: ignore[call-arg]
        ax.set_xlabel("Death")
        ax.set_ylabel("Birth")
        ax.set_title("Persistence Diagram")

        return fig

    def fit(self, x: Tensor | np.ndarray, split_sizes: tuple[int, int, int]) -> TopoGradCluster:
        if isinstance(x, np.ndarray):
            x = torch.as_tensor(x).requires_grad_(True)
        else:
            x = x.cpu().detach().clone().requires_grad_(True)
        optimizer = self.optimizer_cls((x,), lr=self.lr)
        with tqdm(desc="topograd", total=self.iters) as pbar:
            for _ in range(self.iters):
                loss = self._loss_fn(x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.set_postfix(loss=loss.item())
                pbar.update()

        clusters, pers_pairs = tomato(
            x.detach().numpy(),
            k_kde=self.k_kde,
            k_rips=self.k_rips,
            scale=self.scale,
            threshold=self.merge_threshold,
        )
        cluster_labels = np.empty(x.shape[0])
        for k, v in enumerate(clusters.values()):
            cluster_labels[v] = k
        self.labels = cluster_labels
        self.split_indices = np.cumsum(split_sizes)
        self.pers_pairs = pers_pairs

        return self

    def fit_predict(self, x: Tensor | np.ndarray, split_indices: tuple[int, int]) -> np.ndarray:
        return self.fit(x, split_sizes=split_indices).labels

    def predict(self, x: Tensor | np.ndarray, split: Literal["train", "val", "test"]) -> np.ndarray:
        labels = self.labels
        if split == "train":
            return labels[: self.split_indices[0]]
        elif split == "val":
            return labels[self.split_indices[0] : self.split_indices[1]]
        else:  # split == test
            return labels[self.split_indices[1] :]
