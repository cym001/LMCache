# SPDX-License-Identifier: Apache-2.0
"""KV migration plugin interfaces and dynamic loading."""

# Standard
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional
import importlib

# First Party
from lmcache.logging import init_logger
from lmcache.v1.config import LMCacheEngineConfig

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.cache_engine import LMCacheEngine
    from lmcache.v1.metadata import LMCacheMetadata

logger = init_logger(__name__)

DEFAULT_KV_TRANSFER_STORAGE_PLUGIN = "kv_transfer"
DEFAULT_KV_MIGRATION_PLUGIN = "globalkv"
DEFAULT_KV_TRANSFER_BACKEND_MODULE = "lmcache_kv_transfer.backend"
DEFAULT_KV_TRANSFER_BACKEND_CLASS = "KvTransferBackend"
DEFAULT_KV_MIGRATION_MODULE = "lmcache_kv_transfer.migration"
DEFAULT_KV_MIGRATION_CLASS = "GlobalKvMigrationPlugin"


class KvMetadataReporterInterface(ABC):
    """Reports KV cache metadata lifecycle events to an external service."""

    @abstractmethod
    def on_kv_stored(self, tokens: list[int]) -> None:
        """Report tokens stored into the local cache."""

    @abstractmethod
    def on_kv_retrieved(self, hit_tokens: list[int]) -> None:
        """Report tokens retrieved from the local cache."""

    @abstractmethod
    def on_kv_removed(self, chunk_hashes: list[bytes]) -> None:
        """Report removed KV chunk hashes."""

    @abstractmethod
    def on_request_start(self, request_id: str, tokens: list[int]) -> None:
        """Report the start of an inference request."""

    @abstractmethod
    def on_request_end(self, request_id: str, tokens: list[int]) -> None:
        """Report the end of an inference request."""

    @abstractmethod
    def close(self) -> None:
        """Release reporter resources."""


class KvMigrationPluginInterface(ABC):
    """Engine extension for KV cache migration and optional RPC serving."""

    @abstractmethod
    def start(self, engine: "LMCacheEngine") -> None:
        """Start migration workers and optional RPC services."""

    @abstractmethod
    def stop(self) -> None:
        """Stop migration workers and optional RPC services."""

    @abstractmethod
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
        """Execute a KV cache transfer request."""

    @abstractmethod
    def get_metadata_reporter(self) -> KvMetadataReporterInterface | None:
        """Return the metadata reporter owned by this plugin, if any."""

    def configure_kv_events_sink(self, engine: "LMCacheEngine") -> None:
        """Optionally wire KV event sinks after storage backends are ready."""

    def get_transfer_backend_name(self) -> str:
        """Return the storage backend name used for peer transfer."""
        return "KvTransferBackend"


def _ensure_extra_config(config: LMCacheEngineConfig) -> dict[str, Any]:
    if config.extra_config is None:
        config.extra_config = {}
    return config.extra_config


def apply_kv_transfer_plugin_compat(config: LMCacheEngineConfig) -> None:
    """Map legacy KV transfer flags to plugin configuration."""
    extra_config = _ensure_extra_config(config)

    enable_kv_transfer = bool(getattr(config, "enable_kv_transfer", False))
    has_transfer_endpoints = (
        getattr(config, "kv_transfer_host", None) is not None
        and getattr(config, "kv_transfer_init_port", None) is not None
    )

    if enable_kv_transfer and has_transfer_endpoints:
        storage_plugins = list(config.storage_plugins or [])
        if DEFAULT_KV_TRANSFER_STORAGE_PLUGIN not in storage_plugins:
            storage_plugins.append(DEFAULT_KV_TRANSFER_STORAGE_PLUGIN)
            config.storage_plugins = storage_plugins
            logger.warning(
                "enable_kv_transfer is deprecated; add storage_plugins: "
                "['%s'] with storage_plugin.%s.* in extra_config",
                DEFAULT_KV_TRANSFER_STORAGE_PLUGIN,
                DEFAULT_KV_TRANSFER_STORAGE_PLUGIN,
            )

        module_path_key = (
            f"storage_plugin.{DEFAULT_KV_TRANSFER_STORAGE_PLUGIN}.module_path"
        )
        class_name_key = (
            f"storage_plugin.{DEFAULT_KV_TRANSFER_STORAGE_PLUGIN}.class_name"
        )
        extra_config.setdefault(module_path_key, DEFAULT_KV_TRANSFER_BACKEND_MODULE)
        extra_config.setdefault(class_name_key, DEFAULT_KV_TRANSFER_BACKEND_CLASS)

    enable_globalkv = bool(getattr(config, "enable_globalkv_server", False))
    has_meta_client = (
        enable_kv_transfer
        or extra_config.get("globalkv_meta_host") is not None
        or extra_config.get("globalkv_meta_port") is not None
    )
    if enable_globalkv or has_meta_client:
        migration_plugins = list(getattr(config, "kv_migration_plugins", None) or [])
        if DEFAULT_KV_MIGRATION_PLUGIN not in migration_plugins:
            migration_plugins.append(DEFAULT_KV_MIGRATION_PLUGIN)
            config.kv_migration_plugins = migration_plugins
            if enable_globalkv:
                logger.warning(
                    "enable_globalkv_server is deprecated; add kv_migration_plugins: "
                    "['%s'] with kv_migration_plugin.%s.* in extra_config",
                    DEFAULT_KV_MIGRATION_PLUGIN,
                    DEFAULT_KV_MIGRATION_PLUGIN,
                )

        plugin_prefix = f"kv_migration_plugin.{DEFAULT_KV_MIGRATION_PLUGIN}"
        extra_config.setdefault(f"{plugin_prefix}.module_path", DEFAULT_KV_MIGRATION_MODULE)
        extra_config.setdefault(f"{plugin_prefix}.class_name", DEFAULT_KV_MIGRATION_CLASS)


def _load_plugin_class(module_path: str, class_name: str) -> type[Any]:
    module = importlib.import_module(module_path)
    plugin_class = getattr(module, class_name)
    return plugin_class


def _build_plugin_params(config: LMCacheEngineConfig, plugin_name: str) -> dict[str, Any]:
    extra_config = config.extra_config or {}
    prefix = f"kv_migration_plugin.{plugin_name}."
    params: dict[str, Any] = {}
    for key, value in extra_config.items():
        if key.startswith(prefix):
            params[key[len(prefix) :]] = value
    return params


def load_kv_migration_plugin(
    config: LMCacheEngineConfig,
    metadata: Optional["LMCacheMetadata"] = None,
) -> KvMigrationPluginInterface | None:
    """Load the first configured KV migration plugin, if any."""
    apply_kv_transfer_plugin_compat(config)
    plugin_names = getattr(config, "kv_migration_plugins", None) or []
    if not plugin_names:
        return None

    plugin_name = plugin_names[0]
    extra_config = config.extra_config or {}
    module_path = extra_config.get(f"kv_migration_plugin.{plugin_name}.module_path")
    class_name = extra_config.get(f"kv_migration_plugin.{plugin_name}.class_name")
    if not module_path or not class_name:
        logger.warning(
            "KV migration plugin '%s' is missing module_path or class_name",
            plugin_name,
        )
        return None

    try:
        plugin_class = _load_plugin_class(module_path, class_name)
        plugin_params = _build_plugin_params(config, plugin_name)
        plugin = plugin_class(
            config=config,
            metadata=metadata,
            plugin_params=plugin_params,
        )
        if not isinstance(plugin, KvMigrationPluginInterface):
            raise TypeError(
                f"{class_name} must implement KvMigrationPluginInterface"
            )
        return plugin
    except Exception as exc:
        logger.error(
            "Failed to load KV migration plugin '%s': %s",
            plugin_name,
            exc,
            exc_info=True,
        )
        return None


def load_metadata_reporter(
    config: LMCacheEngineConfig,
    metadata: Optional["LMCacheMetadata"] = None,
    *,
    meta_host: str | None = None,
    meta_port: int | None = None,
) -> KvMetadataReporterInterface | None:
    """Load a metadata reporter from the configured migration plugin."""
    plugin = load_kv_migration_plugin(config, metadata)
    if plugin is None:
        return None

    create_reporter = getattr(plugin, "create_metadata_reporter", None)
    if callable(create_reporter):
        return create_reporter(meta_host=meta_host, meta_port=meta_port)

    return plugin.get_metadata_reporter()


def find_kv_transfer_backend(storage_manager: Any) -> Any | None:
    """Locate the configured KV transfer storage backend."""
    for backend in storage_manager.storage_backends.values():
        if str(backend) == "KvTransferBackend":
            return backend
        transfer_to_peer = getattr(backend, "transfer_to_peer", None)
        if callable(transfer_to_peer):
            return backend
    return None
