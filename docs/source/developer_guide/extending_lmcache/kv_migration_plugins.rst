KV Migration Plugins
====================

LMCache supports optional KV cache migration and GlobalKV metadata reporting
through the ``lmcache-kv-transfer`` plugin package.

Install the plugin:

.. code-block:: bash

   pip install -e plugins/kv_transfer

Configure LMCache with storage and migration plugins:

.. code-block:: yaml

   enable_kv_transfer: true
   enable_globalkv_server: true
   kv_transfer_host: 127.0.0.1
   kv_transfer_init_port: 5555
   kv_transfer_http_port: 8010
   kv_transfer_rpc_port: 17071

   storage_plugins: ["kv_transfer"]
   kv_migration_plugins: ["globalkv"]

   extra_config:
     storage_plugin.kv_transfer.module_path: lmcache_kv_transfer.backend
     storage_plugin.kv_transfer.class_name: KvTransferBackend
     kv_migration_plugin.globalkv.module_path: lmcache_kv_transfer.migration
     kv_migration_plugin.globalkv.class_name: GlobalKvMigrationPlugin
     kv_migration_plugin.globalkv.meta_host: 127.0.0.1
     kv_migration_plugin.globalkv.meta_port: 17070
     kv_migration_plugin.globalkv.enable_rpc_server: true

Legacy flags such as ``enable_kv_transfer`` and ``enable_globalkv_server`` are
still accepted and automatically mapped to the plugin configuration, but they
emit deprecation warnings.

Core interfaces
---------------

- ``KvMigrationPluginInterface`` in ``lmcache.v1.plugin.kv_migration``
- ``KvMetadataReporterInterface`` in ``lmcache.v1.plugin.kv_migration``

``LMCacheEngine.kv_transfer()`` remains the public API and delegates to the
loaded migration plugin.
