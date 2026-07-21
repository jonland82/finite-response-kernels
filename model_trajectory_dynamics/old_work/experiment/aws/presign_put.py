#!/usr/bin/env python3
"""Create a short-lived S3 PutObject URL without exposing AWS credentials."""

from __future__ import annotations

import argparse

import boto3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--expires", type=int, default=7200)
    arguments = parser.parse_args()
    client = boto3.client("s3", region_name=arguments.region)
    print(
        client.generate_presigned_url(
            "put_object",
            Params={
                "Bucket": arguments.bucket,
                "Key": arguments.key,
                "ContentType": "application/gzip",
            },
            ExpiresIn=arguments.expires,
            HttpMethod="PUT",
        )
    )


if __name__ == "__main__":
    main()
