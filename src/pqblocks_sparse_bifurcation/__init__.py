"""Sparse Structure-Aware Discrete Simulated Bifurcation ."""

from .solver import (
    SparseDSBConfig,
    DSBResult,
    RunReceipt,
    SparseStructureAwareDSB,
)
from .layout import (
    CommunityLayout,
    detect_community_layout,
)
from .reference import (
    ReferenceDSB,
)
from .kernel import (
    CompiledKernelDSB,
    dsb_dense_kernel,
    dsb_csr_kernel,
    _sign,
)

__all__ = [
    "SparseDSBConfig",
    "DSBResult",
    "RunReceipt",
    "SparseStructureAwareDSB",
    "CommunityLayout",
    "detect_community_layout",
    "ReferenceDSB",
    "CompiledKernelDSB",
    "dsb_dense_kernel",
    "dsb_csr_kernel",
    "_sign",
]
