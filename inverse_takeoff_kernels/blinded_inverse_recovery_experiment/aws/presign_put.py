import argparse

import boto3


parser = argparse.ArgumentParser()
parser.add_argument("--bucket", required=True)
parser.add_argument("--key", required=True)
parser.add_argument("--region", required=True)
parser.add_argument("--expires", type=int, default=7200)
args = parser.parse_args()

client = boto3.client("s3", region_name=args.region)
print(
    client.generate_presigned_url(
        "put_object",
        Params={"Bucket": args.bucket, "Key": args.key, "ContentType": "application/gzip"},
        ExpiresIn=args.expires,
    )
)
