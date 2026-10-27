"""TABM Core — Token-Aware Memory scheduling layer for RK3588."""

from RK3588.edgeflow_rk3588.runtime.tabm.adapter_worker import (
    MNNAdapterWorkerBackend,
)
from RK3588.edgeflow_rk3588.runtime.tabm.admission import (
    AdmissionState,
    Decision,
    RequestCost,
    TokenFlowPolicy,
)

# NOT exported on purpose: mnn_worker.MNNDemoWorkerBackend is the legacy
# llm_demo-parsing backend, queued for demotion once e2e_smoke migrates to
# the adapter backend — import it from its module directly if needed.
from RK3588.edgeflow_rk3588.runtime.tabm.dispatch import LaneDispatcher, ReapRecord
from RK3588.edgeflow_rk3588.runtime.tabm.lane import (
    FakeWorkerBackend,
    Lane,
    ProcessWorkerBackend,
    WorkItem,
    WorkResult,
    WorkerBackend,
)
from RK3588.edgeflow_rk3588.runtime.tabm.metrics import (
    MetricsCollector,
    RequestMetrics,
)
from RK3588.edgeflow_rk3588.runtime.tabm.prefix_registry import (
    CacheProvider,
    MNNCacheFileProvider,
    NullCacheProvider,
    PrefixCompat,
    PrefixEntry,
    PrefixKind,
    PrefixRegistry,
    PrefixValidity,
    PrefixValidityError,
    fnv1a64_token_ids,
)
from RK3588.edgeflow_rk3588.runtime.tabm.shm_slot import (
    ShmSlotPool,
    SlotBufferDescriptor,
    attach_embedding,
)
from RK3588.edgeflow_rk3588.runtime.tabm.slot import (
    LaneOwner,
    Slot,
    SlotPool,
    SlotState,
    SlotTransitionError,
    TABM_SLOT_ABI_VERSION,
)

__all__ = [
    "AdmissionState",
    "CacheProvider",
    "Decision",
    "FakeWorkerBackend",
    "Lane",
    "LaneDispatcher",
    "LaneOwner",
    "MNNAdapterWorkerBackend",
    "MNNCacheFileProvider",
    "MetricsCollector",
    "NullCacheProvider",
    "PrefixCompat",
    "PrefixEntry",
    "PrefixKind",
    "PrefixRegistry",
    "PrefixValidity",
    "PrefixValidityError",
    "ProcessWorkerBackend",
    "ReapRecord",
    "RequestCost",
    "RequestMetrics",
    "ShmSlotPool",
    "Slot",
    "SlotBufferDescriptor",
    "SlotPool",
    "SlotState",
    "SlotTransitionError",
    "TABM_SLOT_ABI_VERSION",
    "TokenFlowPolicy",
    "WorkItem",
    "WorkResult",
    "WorkerBackend",
    "attach_embedding",
    "fnv1a64_token_ids",
]
