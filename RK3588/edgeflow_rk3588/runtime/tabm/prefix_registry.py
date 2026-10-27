"""Prefix identity, registry and lane-local cache-provider interfaces.

Prefixes are keyed by token identity plus model, tokenizer, template,
quantization and backend compatibility. Registration can start with a
provisional text identity; set_token_identity() installs the final key and
merges duplicate registrations.

LRU and TTL eviction remove registry entries. The explicit EXPIRED state
instead keeps an entry inspectable while marking it unusable. Physical
materialization and cache validity are supplied by the caller's backend and
callbacks; an unwired provider cannot execute requests.

This module re-exports the identity types from prefix_identity, persistence
and registry operations from prefix_store, and providers from cache_providers.
It contains no device calibration or experiment records.
"""
from RK3588.edgeflow_rk3588.runtime.tabm.prefix_identity import (  # noqa: F401
    _FNV64_MASK,
    _FNV64_OFFSET_BASIS,
    _FNV64_PRIME,
    _TOKEN_IDS_HASH_RE,
    PrefixCompat,
    PrefixKind,
    _compat_key_string,
    _final_prefix_id,
    _provisional_prefix_id,
    _validate_token_identity,
    fnv1a64_token_ids,
)
from RK3588.edgeflow_rk3588.runtime.tabm.prefix_store import (  # noqa: F401
    PrefixEntry,
    PrefixRegistry,
    PrefixValidity,
    PrefixValidityError,
    _LEGAL_VALIDITY_MOVES,
    _VALIDITY_RANK,
    _compat_from_dict,
    _compat_to_dict,
    _default_revalidate,
    _entry_from_dict,
    _entry_to_dict,
    _merge_entries,
)
from RK3588.edgeflow_rk3588.runtime.tabm.cache_providers import (  # noqa: F401
    CacheProvider,
    MNNCacheFileProvider,
    NullCacheProvider,
    _default_handle_exists,
)

__all__ = [
    "CacheProvider",
    "MNNCacheFileProvider",
    "NullCacheProvider",
    "PrefixCompat",
    "PrefixEntry",
    "PrefixKind",
    "PrefixRegistry",
    "PrefixValidity",
    "PrefixValidityError",
    "fnv1a64_token_ids",
]
