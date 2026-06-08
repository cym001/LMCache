# SPDX-License-Identifier: Apache-2.0
# Standard
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.kv_transfer_status import KV_TRANSFER_FAILED
from lmcache.v1.plugin.kv_migration import KvMigrationPluginInterface


class _StubMigrationPlugin(KvMigrationPluginInterface):
    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self.last_transfer: dict | None = None

    def start(self, engine: LMCacheEngine) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def transfer(
        self,
        hashes: list[int],
        offsets: list[int],
        old_position: str,
        peer_ip: str,
        peer_init_port: int,
        event_id: str,
        do_copy: bool = True,
        token_ids: list[int] | None = None,
    ) -> int:
        self.last_transfer = {
            "hashes": hashes,
            "offsets": offsets,
            "old_position": old_position,
            "peer_ip": peer_ip,
            "peer_init_port": peer_init_port,
            "event_id": event_id,
            "do_copy": do_copy,
            "token_ids": token_ids,
        }
        return 16

    def get_metadata_reporter(self):
        return None


def test_kv_transfer_without_plugin_returns_failed() -> None:
    engine = LMCacheEngine.__new__(LMCacheEngine)
    engine._kv_migration_plugin = None

    result = LMCacheEngine.kv_transfer(
        engine,
        hashes=[101],
        offsets=[16],
        old_position="LocalCPUBackend",
        peer_ip="127.0.0.1",
        peer_init_port=5555,
        event_id="evt",
    )

    assert result == KV_TRANSFER_FAILED


def test_kv_transfer_delegates_to_plugin() -> None:
    engine = LMCacheEngine.__new__(LMCacheEngine)
    plugin = _StubMigrationPlugin()
    engine._kv_migration_plugin = plugin

    result = LMCacheEngine.kv_transfer(
        engine,
        hashes=[101],
        offsets=[16],
        old_position="LocalCPUBackend",
        peer_ip="127.0.0.1",
        peer_init_port=5555,
        event_id="evt",
        token_ids=[1, 2, 3],
    )

    assert result == 16
    assert plugin.last_transfer is not None
    assert plugin.last_transfer["token_ids"] == [1, 2, 3]


def test_load_kv_migration_plugin_reads_extra_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lmcache.v1.config import LMCacheEngineConfig
    from lmcache.v1.plugin import kv_migration as kv_migration_module

    config = LMCacheEngineConfig.from_legacy(
        backend="cpu",
        enable_kv_transfer=True,
        kv_transfer_host="127.0.0.1",
        kv_transfer_init_port=5555,
        extra_config={
            "kv_migration_plugin.globalkv.module_path": (
                "lmcache_kv_transfer.migration"
            ),
            "kv_migration_plugin.globalkv.class_name": "GlobalKvMigrationPlugin",
        },
    )

    plugin = kv_migration_module.load_kv_migration_plugin(config)
    assert plugin is not None
    assert isinstance(plugin, kv_migration_module.KvMigrationPluginInterface)
