# SPDX-License-Identifier: Apache-2.0
"""Deprecated re-export. Use lmcache_kv_transfer.globalkv_server instead."""

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

try:
    from lmcache_kv_transfer.globalkv_server import (  # noqa: F401
        GlobalKvServer,
        LmcacheServerServicer,
    )
except ImportError as exc:
    raise ImportError(
        "GlobalKvServer moved to the lmcache-kv-transfer plugin. "
        "Install it with: pip install -e plugins/kv_transfer"
    ) from exc

logger.warning(
    "lmcache.v1.remote.server.globalkv_server is deprecated; "
    "import from lmcache_kv_transfer.globalkv_server instead"
)
