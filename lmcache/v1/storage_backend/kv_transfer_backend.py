# SPDX-License-Identifier: Apache-2.0
"""Deprecated re-export. Use lmcache_kv_transfer.backend instead."""

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

try:
    from lmcache_kv_transfer.backend import (  # noqa: F401
        KV_TRANSFER_MEM_INDEX_UNAVAILABLE,
        KvTransferBackend,
        KvTransferPeerResult,
    )
except ImportError as exc:
    raise ImportError(
        "KvTransferBackend moved to the lmcache-kv-transfer plugin. "
        "Install it with: pip install -e plugins/kv_transfer"
    ) from exc

logger.warning(
    "lmcache.v1.storage_backend.kv_transfer_backend is deprecated; "
    "import from lmcache_kv_transfer.backend instead"
)
