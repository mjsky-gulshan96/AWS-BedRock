import boto3
import json
import logging
from botocore.exceptions import ClientError
import os

# Retrieve the list of existing buckets
s3_client = boto3.client('s3')
response = s3_client.list_buckets()


# Output the bucket names
# for bucket in response['Buckets']:
#     print(f'  {bucket["Name"]}')


def upload_file():

    curr_path = os.getcwd()
    file = 'testfile.jsonl'
    file_name = os.path.join(curr_path, file)
    bucket = 'testdocreader'

    # Upload the file
    try:
        response = s3_client.upload_file(file_name, bucket, file)
        print(response)
    except ClientError as e:
        logging.error(e)
        print('false')

upload_file()