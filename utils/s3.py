import boto3


class S3Resource:
    def __init__(self, config):
        self.resource = boto3.resource('s3', endpoint_url=config.endpoint_url,
                                       aws_access_key_id=config.aws_access_key_id,
                                       aws_secret_access_key=config.aws_secret_access_key)
        self.bucket_name = config.bucket
        self.bucket = self.resource.Bucket(name=self.bucket_name)


def initialize_s3_resource(config):
    global s3_resource
    s3_resource = S3Resource(config)
