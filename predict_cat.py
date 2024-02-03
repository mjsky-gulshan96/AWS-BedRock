import boto3
import json
import streamlit as st
import os

region_name = os.environ.get("AWS_DEFAULT_REGION", 'us-west-2')


bedrock = boto3.client(
    service_name="bedrock",
    region_name=region_name
)

bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name=region_name
)

all_categories = """newarrivals, newarrivals-womens, newarrivals-mens, newarrivals-electronics, 
womens, womens-clothing, womens-outfits, womens-clothing-tops, womens-clothing-dresses, 
womens-clothing-bottoms, womens-clothing-jackets, womens-clothing-feeling-red, womens-jewelry, 
womens-jewelry-earrings, womens-jewelry-bracelets, womens-jewelry-necklaces, womens-accessories, 
womens-accessories-scarves, womens-accessories-shoes, mens, mens-clothing, mens-clothing-suits, 
mens-clothing-jackets, mens-clothing-dress-shirts, mens-clothing-shorts, mens-clothing-pants, 
mens-accessories, mens-accessories-ties, mens-accessories-gloves, mens-accessories-luggage, 
electronics, electronics-televisions, electronics-televisions-flat-screen, 
electronics-televisions-projection, electronics-televisions-tv_dvd-combo, electronics-digital-cameras,
electronics-camcorders, electronics-digital-media-players, electronics-mobile-phones, electronics-gps-units, 
electronics-gaming, electronics-game-consoles, electronic-games, electronics-accessories, gift-certificates,
top-seller, hidden, sale, sale-mens, sale-mens-footwear, sale-mens-clothing, sale-mens-accessories, 
sale-womens, sale-womens-dresses, sale-womens-clothing, sale-womens-accessories, sale-electronics, 
sale-electronics-tv, sale-electronics-camcorders, sale-electronics-digital, sale-electronics-gps, 
sale-electronics-gaming, sale-electronics-mobile, sale-electronics-portable-audio, sale-electronics-accessories
"""

st.title("Category prediction")

model = "anthropic.claude-v2:1"
prompt = st.text_area('enter your prompt', value="",)
# context: here is the list of all the product categories provided between ##.
#     #{all_categories}#
def text_response(description):

    prompt = f"""

    here is product short description : {description}

    Question: based on the product short description, 
    give me the list of all the possible categories that matches only from the given {all_categories}.
    Please provide categories for both men and women. If there is no exact match
    in the given list give no result. Give result in only one word else give as "NA".
    """

    body = json.dumps({
            "prompt": "\n\nHuman: "+ prompt +"\n\nAssistant:",
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
    return response

def output_response(response):
    # print(data.keys())
    data = json.loads(response['body'].read())
    final_resp = data.get("completion")
    st.write(final_resp)
    st.write('--------fnal response-------------')
    start_ind = final_resp.find('"')
    end_ind = final_resp.find('"', start_ind + 1)
    substr = final_resp[start_ind + 1:end_ind]
    st.write(substr)


if prompt:
    response = text_response(prompt)
    output_response(response)
    