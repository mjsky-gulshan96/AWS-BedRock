import boto3
import json

bedrock = boto3.client(
    service_name="bedrock",
    region_name='us-east-1'
)

foundation_model =  bedrock.get_foundation_model(
    modelIdentifier='cohere.command-light-text-v14'
)

# SageMaker = boto3.client('sagemaker')
# sagemaker_models = SageMaker.Client.list_models()

# print(sagemaker_models)

# models = bedrock.list_foundation_models().get('modelSummaries')

# for model in models:
#     print(model)


# list of models support fine tunning
for model in bedrock.list_foundation_models(
    byCustomizationType="FINE_TUNING")["modelSummaries"]:
    for key, value in model.items():
        print(key, ":", value)
    print("-----\n")
