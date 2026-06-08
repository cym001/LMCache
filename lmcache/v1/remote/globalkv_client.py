# SPDX-License-Identifier: Apache-2.0
"""Deprecated re-export. Use lmcache_kv_transfer.metadata_client instead."""

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

try:
    from lmcache_kv_transfer.metadata_client import KvCacheClient  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "KvCacheClient moved to the lmcache-kv-transfer plugin. "
        "Install it with: pip install -e plugins/kv_transfer"
    ) from exc

logger.warning(
    "lmcache.v1.remote.globalkv_client is deprecated; "
    "import from lmcache_kv_transfer.metadata_client instead"
)
