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
globalkv_protocol: v2
lmcache_instance_id: stable-instance-name
pre_caching_hash_algorithm: sha256_cbor
extra_config:
  globalkv_advertised_host: 10.0.0.12
  globalkv_api_scheme: http
  globalkv_api_path: /v1
```

`globalkv_protocol` accepts `v1`, `dual`, or `v2` and defaults to `v1`.
`dual` keeps all V1 reporting and migration behavior while probing the V2
capability endpoint. `dual` and `v2` require a stable `lmcache_instance_id`.
The V2 `InstanceKey` is `(lmcache_instance_id, worker_id)`, where `worker_id`
is the global distributed rank embedded in `CacheEngineKey`; `local_worker_id`
is not part of the identity because it can repeat on different hosts.
When `builtin` hashing is selected for dual/V2, every process must set the same
`PYTHONHASHSEED`; `sha256_cbor` avoids that process-local hash dependency.

`globalkv_advertised_host` is the address returned to other cluster nodes and
must not be `0.0.0.0` or `::`. The API URL is consistently assembled as
`{scheme}://{advertised_host}:{http_port}{api_path}` for registration and
search results. V1 remains available for one compatibility release but is
deprecated; migrate from `v1` to `dual`, verify parity, and then switch to
`v2`. The legacy `enable_kv_transfer` and `enable_globalkv_server` flags remain
mapped to plugins during the same deprecation window.

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
