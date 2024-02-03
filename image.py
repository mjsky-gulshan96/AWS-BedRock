import boto3
import json
import base64
import streamlit as st

bedrock = boto3.client(
    service_name="bedrock",
    region_name='us-east-1'
)

bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name='us-east-1'
)

#  amazon titan image gen

def image_response(prompt):

    stability_payload = {
    "modelId": "stability.stable-diffusion-xl-v0",
    "contentType": "application/json",
    "accept": "application/json",
    "body": "{\"text_prompts\":[{\"text\":\"" + prompt + "\"}],\"cfg_scale\":10,\"seed\":0,\"steps\":50}"
    }

    amazon_payload = {
    "modelId": "amazon.titan-image-generator-v1",
    "contentType": "application/json",
    "accept": "application/json",
    "body": "{\"taskType\": \"TEXT_IMAGE\", \"textToImageParams\": {\"text\": \""+ prompt +"\"},\"imageGenerationConfig\": {\"numberOfImages\": 3,\"quality\": \"standard\",\"height\": 1024,\"width\": 1024,\"cfgScale\": 8.0,\"seed\": 0 } }"
    }

    response = bedrock_runtime.invoke_model(
    body=amazon_payload["body"],
    contentType="application/json",
    accept="*/*",
    modelId=amazon_payload["modelId"]
    )

    return response



count = 0

st.title("Create Your Own Image")

prompt = st.text_input('', value="", max_chars=500, key=None, type="default", placeholder='enter your prompt')

# if using amazon model 
if prompt:
    image_title = prompt[0: 4]
    response = image_response(prompt)
    data = json.loads(response['body'].read())
    # amazon titan model
    for image in enumerate(data["images"]):
        with open(f"./images/{image_title}{count}.png", "wb") as f:
            f.write(base64.b64decode(image[1]))
            st.image(f"./images/{image_title}{count}.png")
            count +=1


# stablity model

# if prompt:
#     image_title = prompt[0: 4]
#     response = image_response(prompt)
#     data = json.loads(response['body'].read())
#     for i, image in enumerate(data["artifacts"]):
#         with open(f"./images/{image_title}{i}.png", "wb") as f:
#             f.write(base64.b64decode(image["base64"]))
#             st.image(f"./images/{image_title}{i}.png")

