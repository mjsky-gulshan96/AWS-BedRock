import boto3

bedrock = boto3.client(
    service_name="bedrock",
    region_name='us-east-1'
)

job_name = 'test_job'
base_model_id = 'cohere.command-text-v14'
model_name = 'mycustom_model'
rolearn = 'arn:aws:iam::459425521263:role/AI-User-Role'

bedrock.create_model_customization_job(
    customizationType="FINE_TUNING",
    jobName=job_name,
    customModelName=model_name,
    roleArn=rolearn,
    baseModelIdentifier=base_model_id,
    hyperParameters = {
        "epochCount": "10",
        "batchSize": "8",
        "learningRate": "0.00001",
    },
    trainingDataConfig={"s3Uri": "s3://testdocreader/testfile.jsonl"},
    outputDataConfig={"s3Uri": "s3://testdocreader/output/"},
)

# check for job status

fine_tune_job = bedrock.get_model_customization_job(jobIdentifier = job_name)
print(fine_tune_job['status'])