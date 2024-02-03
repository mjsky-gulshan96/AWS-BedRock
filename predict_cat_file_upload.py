import boto3
import json
import pandas as pd
import streamlit as st
import os
import re

region = os.environ.get("AWS_DEFAULT_REGION", 'us-west-2')
bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name=region)
model = "anthropic.claude-v2:1"
all_categories_file = './categoryData/all_categories.txt'
prompt_file = './categoryData/prompt.txt'

st.title("Category prediction")

def model_response(prompt, to_replace):
    for key in to_replace:
        prompt = prompt.replace(key, to_replace[key])
    
    body = json.dumps({
        "prompt": "\n\nHuman: " + prompt + "\n\nAssistant:",
        "max_tokens_to_sample": 1000,
        "temperature": 0.5,
        "top_k": 250,
        "top_p": 1,
        "anthropic_version": "bedrock-2023-05-31"
    })

    response = bedrock_runtime.invoke_model(
        body=body,
        contentType="application/json",
        accept="*/*",
        modelId=model
    )
    return json.loads(response['body'].read()).get("completion")


def text_response(description):
    all_categories = ''
    prompt = ''

    with open(all_categories_file) as f:
        all_categories = f.read()

    with open(prompt_file) as f:
        prompt = f.read()

    to_replace = {
        "<<description>>": description,
        "<<all_categories>>": all_categories
    }
    return model_response(prompt, to_replace)


def filter_res(suggested_category):
    # check if response contains the list
    matches = re.search(r'\[([^\]]+)\]', suggested_category)
    if matches:
        return matches.group(1)
    
    # split if response has no list
    filter_categories = suggested_category.split(':')

    if len(filter_categories) == 1:
        return filter_categories[0]
    
    return filter_categories[1]

def generateXml(xml_data):
    xml_prompt_file = './categoryData/xml_prompt.txt'
    xml_out_format = './output/xml_output_format.txt'
    xml_out_file = './output/xml_output.txt'
    xml_prompt = ''
    with open(xml_prompt_file) as f:
        xml_prompt = f.read()
    data = f"""
    {xml_data}
    """
    to_replace = {
        "<<xml_data>>": data
    }
    xml_resp =  model_response(xml_prompt, to_replace)
    # st.write(xml_resp)
    start_ind = xml_resp.rindex("xml")
    end_ind = xml_resp.rindex("```")
    xml_resp = xml_resp[start_ind + 3:end_ind]
    xml_output = ''
    with open(xml_out_format) as f:
        xml_output = f.read()

    xml_output = xml_output.replace("<<to_replace_xml>>", xml_resp)
    f = open(xml_out_file, "w")
    f.write(xml_output)
    f.close()
    return xml_output

# upload the csv file
uploaded_csv_file = st.file_uploader("Choose your file (csv format supported)")

if uploaded_csv_file is not None:
    # convert to dataframe
    csv_dframe = pd.read_csv(uploaded_csv_file)
    st.write(csv_dframe)

    # read short description and pids from input file
    all_short_descriptions = csv_dframe.loc[:,'shortDescription__default']
    all_pids = csv_dframe.loc[:,'ID']
    all_suggested_categories = []
    xml_data = []
    for index, shortDesc in enumerate(all_short_descriptions):
        suggested_category = text_response(shortDesc)
        # st.write(suggested_category)
        suggested_category = filter_res(suggested_category)

        # if more than one category present , assign the first category
        if ',' in suggested_category:
            suggested_category = suggested_category.split(',')[0]
        ct_data = {
            "category-id": suggested_category,
            "product-id" : all_pids[index]
        }
        xml_data.append(ct_data)
        all_suggested_categories.append(suggested_category)

    csv_dframe['category'] = all_suggested_categories
    st.write(csv_dframe)
    output_file = "./output/output_res.csv"
    csv_dframe.to_csv(output_file)
    # st.write(xml_data)
    final_xml = generateXml(xml_data)
    # download the file
    if final_xml:
        with open('./output/xml_output.txt', 'rb') as f:
            st.download_button('Download xml file', f, file_name='xml_output.txt')
