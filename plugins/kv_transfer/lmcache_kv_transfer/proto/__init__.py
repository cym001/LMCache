# SPDX-License-Identifier: Apache-2.0
"""Generated GlobalKV protobuf bindings and descriptor verification helpers."""

from hashlib import sha256

from google.protobuf import descriptor_pb2


def v2_descriptor_sha256() -> bytes:
    """Return the canonical two-file V2 descriptor-set digest."""
    from lmcache_kv_transfer.proto import kvcache_v2_pb2, kvserver_v2_pb2

    descriptor_set = descriptor_pb2.FileDescriptorSet()
    kvcache_v2_pb2.DESCRIPTOR.CopyToProto(descriptor_set.file.add())
    kvserver_v2_pb2.DESCRIPTOR.CopyToProto(descriptor_set.file.add())
    return sha256(descriptor_set.SerializeToString()).digest()
