#!/usr/bin/env python3
"""
List and (optionally) download objects from the STL Forge S3-compatible bucket.

Useful when:
- The /runsync POST hit Cloudflare's edge timeout but the worker actually
  finished — the STL is in the bucket, just not in the client response
- You want to recover old STLs from past runs
- Auditing what's in the bucket

Reads BUCKET_* env vars from the environment (or .env if you set
DOTENV_PATH). Compatible with any S3 endpoint: Cloudflare R2, Backblaze B2,
Filebase, MinIO, AWS S3, etc.

Usage:
    # List everything in the bucket
    python download_from_storage.py

    # Download a specific key to a local path
    python download_from_storage.py --key 'stl-forge/sync-abc-123.stl' --out ./my.stl

    # Download every object to a local directory
    python download_from_storage.py --download-all --out-dir /tmp/stls
"""
import argparse
import os
import sys
from pathlib import Path

# Make boto3 optional — show a helpful error if not installed
try:
    import boto3
    from botocore.config import Config
except ImportError as e:
    print(f"boto3 is required: pip install boto3  (error: {e})", file=sys.stderr)
    sys.exit(1)


# --- Config from env (BUCKET_* must be set, same vars the handler reads) ---
def _load_env_from_dotenv():
    env_path = os.environ.get("DOTENV_PATH", "<workspace-path>/.env")
    p = Path(env_path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())


def _require(name: str) -> str:
    v = os.environ.get(name, "").strip()
    if not v:
        print(f"missing required env var: {name}", file=sys.stderr)
        sys.exit(2)
    return v


def _client():
    _load_env_from_dotenv()
    return boto3.client(
        "s3",
        endpoint_url=_require("BUCKET_ENDPOINT_URL"),
        aws_access_key_id=_require("BUCKET_ACCESS_KEY_ID"),
        aws_secret_access_key=_require("BUCKET_SECRET_ACCESS_KEY"),
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Download STLs from the STL Forge S3 bucket")
    ap.add_argument("--bucket", help="override BUCKET_NAME env var")
    ap.add_argument("--key", help="specific object key to download")
    ap.add_argument("--out", help="local path to write the downloaded object (default: ./<filename>)")
    ap.add_argument("--download-all", action="store_true", help="download every object in the bucket")
    ap.add_argument("--out-dir", default=".", help="directory for --download-all (default: cwd)")
    args = ap.parse_args()

    _load_env_from_dotenv()
    s3 = _client()
    bucket = args.bucket or _require("BUCKET_NAME")

    if args.key:
        # Single object
        out_path = args.out or os.path.join(".", os.path.basename(args.key))
        print(f"downloading s3://{bucket}/{args.key} -> {out_path}")
        s3.download_file(bucket, args.key, out_path)
        size = os.path.getsize(out_path)
        print(f"  saved {size/1024/1024:.2f} MB")
        return 0

    # List
    print(f"listing s3://{bucket}/")
    resp = s3.list_objects_v2(Bucket=bucket)
    contents = resp.get("Contents", [])
    if not contents:
        print("  (empty)")
        return 0
    print(f"  {len(contents)} object(s):")
    for o in contents:
        size_mb = o["Size"] / 1024 / 1024
        print(f"    {o['Key']}  ({size_mb:.2f} MB, uploaded {o['LastModified']})")

    if args.download_all:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for o in contents:
            key = o["Key"]
            out_path = out_dir / os.path.basename(key)
            print(f"downloading s3://{bucket}/{key} -> {out_path}")
            s3.download_file(bucket, key, str(out_path))
            print(f"  saved {o['Size']/1024/1024:.2f} MB")
    else:
        # Hint: how to actually pull one
        print()
        print("To download a specific object:")
        print(f"  python {sys.argv[0]} --key '<filename>' --out /path/to/save.stl")
        print("To download everything:")
        print(f"  python {sys.argv[0]} --download-all --out-dir /tmp/stls")
    return 0


if __name__ == "__main__":
    sys.exit(main())
