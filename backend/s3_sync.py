import os
from pathlib import Path
import boto3
from botocore.exceptions import ClientError

S3_BUCKET = os.getenv("S3_BUCKET", "").strip()
AWS_REGION = os.getenv("AWS_REGION", "us-west-2").strip()


class S3StorageManager:
    def __init__(self):
        self.bucket = S3_BUCKET
        self.enabled = bool(self.bucket)
        self.client = boto3.client("s3", region_name=AWS_REGION) if self.enabled else None

    def upload_file(self, local_path: Path, s3_key: str) -> bool:
        if not self.enabled or not local_path.exists():
            return False
        try:
            self.client.upload_file(str(local_path), self.bucket, s3_key)
            return True
        except ClientError as e:
            print(f"[S3 Warning] Failed to upload {local_path} -> s3://{self.bucket}/{s3_key}: {e}")
            return False

    def upload_directory(self, local_dir: Path, s3_prefix: str):
        if not self.enabled or not local_dir.exists():
            return
        for item in local_dir.iterdir():
            if item.is_file() and not item.name.startswith("."):
                self.upload_file(item, f"{s3_prefix.rstrip('/')}/{item.name}")

    def download_prefix(self, s3_prefix: str, local_dir: Path) -> int:
        if not self.enabled:
            return 0
        local_dir.mkdir(parents=True, exist_ok=True)
        downloaded = 0
        try:
            paginator = self.client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.bucket, Prefix=s3_prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    filename = Path(key).name
                    if not filename or filename.startswith("."):
                        continue
                    dest = local_dir / filename
                    self.client.download_file(self.bucket, key, str(dest))
                    downloaded += 1
        except ClientError as e:
            print(f"[S3 Warning] Failed to download prefix {s3_prefix}: {e}")
        return downloaded
