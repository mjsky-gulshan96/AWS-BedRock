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

BUCKET_NAME = 'testdocreader'

# upload file to s3 bucket
def upload_file():

    curr_path = os.getcwd()
    file = 'testfile.jsonl'
    file_name = os.path.join(curr_path, file)

    # Upload the file
    try:
        response = s3_client.upload_file(file_name, BUCKET_NAME, file)
        print(response)
    except ClientError as e:
        logging.error(e)
        print('false')

# upload_file()

# retrieve file from s3 bucket

def download_file():

    file_path = './data/testfile.jsonl'
    object_name = 'testfile.jsonl'
    s3_client.download_file(BUCKET_NAME, object_name, file_path)

# download_file()

def read_file_from_s3(bucket_name, file_name):
    resp = s3_client.get_object(Bucket=bucket_name, Key=file_name)
    data = resp['Body'].read()
    print(data)

read_file_from_s3(BUCKET_NAME, 'testfile.jsonl')