from __future__ import annotations
import logging
from typing import Any, Literal, Optional, Tuple, cast
import warnings

from faiss import IndexFlatL2
import matplotlib.pyplot as plt
from numba import jit
import numpy as np
import torch
from torch import Tensor
from torch.autograd import Function
import torch.nn as nn
import torch.optim
from tqdm import tqdm

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


class TopoGradFn(Function):
    @staticmethod
    def forward(
        ctx: Any,
        pc: Tensor,
        k_kde: int,
        k_rips: int,
        scale: float,
        destnum: int,
        **kwargs,
    ) -> Tensor:
        pc_np = pc.detach().cpu().numpy()
        dists_kde, idxs_kde = compute_density_map(pc_np, k_kde, scale)
        dists_kde = dists_kde.astype(float)
        sorted_idxs = np.argsort(dists_kde)
        idxs_kde_sorted = newI1(idxs_kde, sorted_idxs)
        dists_kde_sorted = dists_kde[sorted_idxs]
        pc_np = pc_np[sorted_idxs]
        _, rips_idxs = compute_rips(pc_np, k_rips)
        _, pers_pairs = cluster(dists_kde_sorted, rips_idxs, 1)

        see = np.array([elem for elem in pers_pairs if (elem != np.array([-1, -1])).any()])
        result = []

        for i in np.unique(see[:, 0]):
            result.append([see[see[:, 0] == i][0, 0], max(see[see[:, 0] == i][:, 1])])
        result = np.array(result)
        pdpairs = result

        oripd = dists_kde_sorted[result]
        sorted_idxs = np.argsort(oripd[:, 0] - oripd[:, 1])

        changing = sorted_idxs[:-destnum]
        nochanging = sorted_idxs[-destnum:-1]
        biggest = oripd[sorted_idxs[-1]]
        dest = np.array([biggest[0], biggest[1]])
        changepairs = pdpairs[changing]
        nochangepairs = pdpairs[nochanging]
        pd11 = dists_kde_sorted[changepairs]
        weakdist = np.sum(pd11[:, 0] - pd11[:, 1]) / np.sqrt(2)
        strongdist = np.sum(np.linalg.norm(dists_kde_sorted[nochangepairs] - dest, axis=1))

        ctx.pc = pc_np
        ctx.idxs_kde = idxs_kde_sorted
        ctx.dists_kde = dists_kde_sorted
        ctx.scale = scale
        ctx.changepairs = changepairs
        ctx.nochangepairs = nochangepairs
        ctx.dest = dest

        return torch.as_tensor(weakdist + strongdist)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        pc = ctx.pc
        idxs_kde = ctx.idxs_kde
        dists_kde = ctx.dists_kde
        scale = ctx.scale
        changepairs = ctx.changepairs
        nochangepairs = ctx.nochangepairs
        dest = ctx.dest
        grad_input = cast(np.ndarray, np.zeros_like(pc))

        #  Compute the gradient for changing pairs
        pc_cp_tiled = pc[changepairs][:, :, None]
        coeff_cp_pre = np.sqrt(2) / len(changepairs)  # type: ignore
        coeff_cp = coeff_cp_pre * rbf(
            x=pc_cp_tiled, y=pc[idxs_kde[changepairs]], scale=scale, axis=-1
        )
        direction_cp = pc_cp_tiled - pc[idxs_kde[changepairs]]
        grad_cp = direction_cp * coeff_cp[..., None]
        grad_cp[:, 1] *= -1
        grad_input[idxs_kde[changepairs]] = grad_cp

        #  Compute the gradient for non-changing pairs
        dists = dists_kde[nochangepairs] - dest
        coeff_ncp_pre = (1 / np.linalg.norm(dists) * dists / scale / len(nochangepairs))[..., None]
        pc_ncp_tiled = pc[nochangepairs][:, :, None]
        coeff_ncp = coeff_ncp_pre * rbf(
            x=pc_ncp_tiled, y=pc[idxs_kde[nochangepairs]], scale=scale, axis=-1
        )
        direction_ncp = pc_ncp_tiled - pc[idxs_kde[nochangepairs]]
        grad_ncp = direction_ncp * coeff_ncp[..., None]
        grad_input[idxs_kde[nochangepairs]] = grad_ncp

        grad_input = torch.as_tensor(grad_input)

        return grad_input, None, None, None, None


def recover(xxx):
    ffff = []
    for i in range(len(xxx)):
        ffff.append(np.where(xxx == i)[0][0])

    ffff = np.array(ffff)
    return ffff


class TopoGradLoss(nn.Module):
    def __init__(self, k_kde: int, k_rips: int, scale: float, destnum: int) -> None:
        super().__init__()
        self.k_kde = k_kde
        self.k_rips = k_rips
        self.scale = scale
        self.destnum = destnum

    def forward(self, x: Tensor) -> Tensor:
        return topoclustergrad.apply(x, self.k_kde, self.k_rips, self.scale, self.destnum)


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
