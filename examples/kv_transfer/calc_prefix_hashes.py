# SPDX-License-Identifier: Apache-2.0

"""Calculate ChunkedTokenDatabase-style prefix hashes for token IDs.

This script reproduces the prefix-hash chaining logic in
`lmcache/v1/token_database.py`, but currently only targets `sha256_cbor`.
"""

# Standard
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

# Third Party
import cbor2  # pyright: ignore[reportMissingImports]

DEFAULT_TOKEN_ID_FILE = Path(__file__).with_name("token_id")


def _parse_token_ids(token_ids_arg: str) -> list[int]:
    """Parse token IDs from a JSON array string."""
    try:
        parsed: Any = json.loads(token_ids_arg)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON for --token-ids: {exc}") from exc

    if not isinstance(parsed, list):
        raise ValueError("--token-ids must be a JSON array, e.g. '[1, 2, 3]'.")

    token_ids: list[int] = []
    for i, value in enumerate(parsed):
        if not isinstance(value, int):
            raise ValueError(
                f"token_ids[{i}] is not an integer: {value!r} (type={type(value)})"
            )
        token_ids.append(value)

    return token_ids


def _ensure_pythonhashseed() -> str:
    """Ensure PYTHONHASHSEED is set for reproducible NONE_HASH."""
    hash_seed = os.getenv("PYTHONHASHSEED")
    if hash_seed is None:
        os.environ["PYTHONHASHSEED"] = "0"
        hash_seed = "0"
    return hash_seed


def _sha256_cbor(input_obj: Any) -> bytes:
    """Hash objects serialized with canonical CBOR using SHA-256."""
    input_bytes = cbor2.dumps(input_obj, canonical=True)
    return hashlib.sha256(input_bytes).digest()


def _init_none_hash() -> bytes:
    """Initialize NONE_HASH with PYTHONHASHSEED-compatible behavior."""
    hash_seed = _ensure_pythonhashseed()
    return _sha256_cbor(hash_seed)


def _canonicalize_hash_inputs(
    prefix_hash: bytes | None,
    tokens_tuple: tuple[int, ...],
    extra_keys: list[Any] | None,
    none_hash: bytes,
) -> tuple[bytes, tuple[int, ...], tuple[Any, ...]]:
    """Canonicalize hash inputs to match TokenDatabase semantics."""
    return (
        prefix_hash if prefix_hash is not None else none_hash,
        tokens_tuple,
        tuple(extra_keys) if extra_keys is not None else (),
    )


def _hash_tokens(
    tokens: list[int],
    none_hash: bytes,
    prefix_hash: bytes | None = None,
    extra_keys: list[Any] | None = None,
) -> bytes:
    """Hash token chunk with chained prefix hash."""
    tokens_tuple = tuple(tokens)
    canon_prefix, canon_tokens, canon_extra = _canonicalize_hash_inputs(
        prefix_hash, tokens_tuple, extra_keys, none_hash
    )
    return _sha256_cbor((canon_prefix, canon_tokens, canon_extra))


def _chunk_tokens(tokens: list[int], block_size: int) -> list[list[int]]:
    """Chunk token IDs by block size (keeps the last unfilled chunk)."""
    return [tokens[i : i + block_size] for i in range(0, len(tokens), block_size)]


def _load_token_ids(args: argparse.Namespace) -> list[int]:
    """Load token IDs from CLI JSON string or from a file."""
    if args.token_ids is not None:
        return _parse_token_ids(args.token_ids)

    if args.token_ids_file is not None:
        content = Path(args.token_ids_file).read_text(encoding="utf-8").strip()
        return _parse_token_ids(content)

    content = DEFAULT_TOKEN_ID_FILE.read_text(encoding="utf-8").strip()
    return _parse_token_ids(content)


def _calculate_prefix_hashes(
    token_ids: list[int], block_size: int = 256
) -> tuple[list[bytes], bytes]:
    """Calculate chained prefix hashes with block_size."""
    none_hash = _init_none_hash()
    prefix_hashes: list[bytes] = []
    prefix_hash = none_hash
    for token_chunk in _chunk_tokens(token_ids, block_size):
        prefix_hash = _hash_tokens(token_chunk, none_hash, prefix_hash=prefix_hash)
        prefix_hashes.append(prefix_hash)
    return prefix_hashes, none_hash


def _hash_bytes_to_u8_32_array(hash_bytes: bytes) -> list[int]:
    """Convert hash bytes to u8[32], with pad/trim safety."""
    if len(hash_bytes) < 32:
        hash_bytes = hash_bytes.rjust(32, b"\x00")
    elif len(hash_bytes) > 32:
        hash_bytes = hash_bytes[-32:]
    return list(hash_bytes)


def _format_block_hashes_u8(prefix_hashes: list[bytes]) -> list[list[int]]:
    """Convert all prefix hashes into u8[32] arrays."""
    return [_hash_bytes_to_u8_32_array(h) for h in prefix_hashes]


def _format_hex_list(prefix_hashes: list[bytes]) -> list[str]:
    """Convert all prefix hashes to hex strings."""
    return [h.hex() for h in prefix_hashes]


def _load_expected_hashes(args: argparse.Namespace) -> list[list[int]] | None:
    """Load expected u8[32] hashes from CLI JSON if provided."""
    if args.expected_u8_file is None:
        return None
    content = Path(args.expected_u8_file).read_text(encoding="utf-8").strip()
    parsed: Any = json.loads(content)
    if not isinstance(parsed, list):
        raise ValueError("--expected-u8-file content must be a JSON array.")
    return parsed


def _validate_expected(
    actual: list[list[int]], expected: list[list[int]] | None
) -> bool:
    """Validate actual hashes against expected ones when provided."""
    if expected is None:
        return True
    return actual == expected


def _print_test_data_notice(args: argparse.Namespace, token_ids: list[int]) -> None:
    """Print short notice for token_ids data source."""
    if args.token_ids is not None:
        print(f"Using --token-ids input (len={len(token_ids)})")
        return

    if args.token_ids_file is not None:
        print(f"Using --token-ids-file={args.token_ids_file} (len={len(token_ids)})")
        return

    print(f"Using default token file: {DEFAULT_TOKEN_ID_FILE} (len={len(token_ids)})")


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Calculate ChunkedTokenDatabase prefix_hashes for token_ids "
            "with block_size=256."
        )
    )
    parser.add_argument(
        "--token-ids",
        type=str,
        default=None,
        help="Token IDs as JSON array string, e.g. '[1,2,3]'.",
    )
    parser.add_argument(
        "--token-ids-file",
        type=str,
        default=None,
        help="Path to a file containing a JSON array of token IDs.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=256,
        help="Block size (chunk_size). Default: 256.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Also print JSON output for machine parsing.",
    )
    parser.add_argument(
        "--expected-u8-file",
        type=str,
        default=None,
        help="Optional JSON file of expected u8[32] arrays for assertion.",
    )
    parser.add_argument(
        "--print-hex",
        action="store_true",
        help="Also print block hashes in hex form.",
    )
    return parser


def main() -> None:
    """Entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    if args.block_size <= 0:
        parser.error("--block-size must be a positive integer")

    token_ids = _load_token_ids(args)
    _print_test_data_notice(args, token_ids)

    prefix_hashes, none_hash = _calculate_prefix_hashes(
        token_ids, block_size=args.block_size
    )
    block_hashes_u8_arrays = _format_block_hashes_u8(prefix_hashes)
    expected = _load_expected_hashes(args)
    is_match = _validate_expected(block_hashes_u8_arrays, expected)

    print(f"PYTHONHASHSEED={os.environ['PYTHONHASHSEED']}")
    print(f"Initialized NONE_HASH={none_hash!r} (sha256_cbor)")
    print(f"NONE_HASH(sha256_cbor(seed))={none_hash.hex()}")

    print(f"Block hashes (u8[32]): {block_hashes_u8_arrays}")
    if args.print_hex:
        print(f"Block hashes (hex): {_format_hex_list(prefix_hashes)}")
    if expected is not None:
        print(f"Expected match: {is_match}")

    if args.json:
        output = {
            "block_size": args.block_size,
            "num_tokens": len(token_ids),
            "num_chunks": len(prefix_hashes),
            "hash_algorithm": "sha256_cbor",
            "pythonhashseed": os.environ["PYTHONHASHSEED"],
            "none_hash_hex": none_hash.hex(),
            "prefix_hashes_hex": _format_hex_list(prefix_hashes),
            "block_hashes_u8_arrays": block_hashes_u8_arrays,
            "expected_match": is_match if expected is not None else None,
        }
        print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
