# lmcache-kv-transfer

GlobalKV metadata reporting and KV cache migration plugin for LMCache.

Install in editable mode from the LMCache repository:

```bash
pip install -e plugins/kv_transfer
```

Configure LMCache with:

```yaml
enable_kv_transfer: true
enable_globalkv_server: true
kv_transfer_host: 127.0.0.1
kv_transfer_init_port: 5555
kv_transfer_http_port: 8010
kv_transfer_rpc_port: 17071
globalkv_protocol: v1
```

`globalkv_protocol` accepts `v1`, `dual`, or `v2` and defaults to `v1`.
`dual` keeps all V1 reporting and migration behavior while probing the V2
capability endpoint. `dual` and `v2` require a stable `lmcache_instance_id`.
The V2 `InstanceKey` is `(lmcache_instance_id, worker_id)`, where `worker_id`
is the global distributed rank embedded in `CacheEngineKey`; `local_worker_id`
is not part of the identity because it can repeat on different hosts.

Legacy flags are mapped automatically to:

```yaml
storage_plugins: ["kv_transfer"]
kv_migration_plugins: ["globalkv"]
extra_config:
  storage_plugin.kv_transfer.module_path: lmcache_kv_transfer.backend
  storage_plugin.kv_transfer.class_name: KvTransferBackend
  kv_migration_plugin.globalkv.module_path: lmcache_kv_transfer.migration
  kv_migration_plugin.globalkv.class_name: GlobalKvMigrationPlugin
```
