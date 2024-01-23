import boto3
import json
import streamlit as st
import speech_recognition as sr


bedrock = boto3.client(
    service_name="bedrock",
    region_name='us-east-1'
)

# to invoke any model use bedrock-runtime
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name='us-east-1'
)

# foundation_model =  bedrock.get_foundation_model(
#     modelIdentifier='cohere.command-light-text-v14'
# )

# print(foundation_model)
# models = bedrock.list_foundation_models().get('modelSummaries')
# for model in models:
#     print(model)

st.title("Bedrock text Generation")

models = {
    "AI21 Labs": "ai21.j2-mid-v1",
    "Amazon": "amazon.titan-text-lite-v1",
    'Anthropic': "anthropic.claude-v2:1",
    'Cohere': "cohere.command-text-v14",
    'Meta': "meta.llama2-13b-chat-v1"
}

model = st.sidebar.selectbox("select model", ['Amazon', 'Anthropic', 'Cohere', 'Meta'])

prompt = st.text_input('', value="", max_chars=500, key=None, type="default", placeholder='enter your prompt')

def text_response(prompt, model):

    # default body of cohere
    body = json.dumps({
        "prompt": prompt,
        "max_tokens": 1000,
        "temperature": 0.8
    })

    if model == 'Anthropic':
        body = json.dumps({
            "prompt": "\n\nHuman: "+ prompt +"\n\nAssistant:",
            "max_tokens_to_sample": 1000,
            "temperature": 0.5,
            "top_k": 250,
            "top_p": 1,
            "anthropic_version": "bedrock-2023-05-31"
        })
    elif model == 'Meta':
        body = json.dumps({
            "prompt": prompt,
            "max_gen_len": 512,
            "temperature": 0.2,
            "top_p": 0.9
        })
    elif model == "AI21 Labs":
        body = json.dumps({
            "prompt": prompt, 
            "maxTokens": 200, 
            "temperature": 0, 
            "topP": 250,
            "countPenalty":{"scale":0},
            "presencePenalty":{"scale":0},
            "frequencyPenalty":{"scale":0}
        })
    elif model == "Amazon":
        body = json.dumps({
            "inputText": prompt
        })

    response = bedrock_runtime.invoke_model(
        body=body,
        contentType="application/json",
        accept="*/*",
        modelId=models[model]
    )
    return response

def output_response(response, model_name):
    # print(data.keys())
    data = json.loads(response['body'].read())

    if model_name == "Anthropic":
        st.write(data.get("completion"))
    elif model_name == "Cohere":
        st.write(data.get("generations")[0]['text'])
    elif model_name == "Amazon":
        st.write(data.get('results')[0]['outputText'])
    elif model_name == "Meta":
        st.write(data.get('generation'))

def speech_recog():
    sr_instance = sr.Recognizer()

    with sr.Microphone() as mic:
        st.write("Please speak something...")
        sr_instance.adjust_for_ambient_noise(mic)
        audio = sr_instance.record(mic, duration=4)
        st.write("Audio captured. Recognizing...")

        try:
            text = sr_instance.recognize_google(audio, language='en-in')
            st.write("Your Text:", text)
            res = text_response(text, model)
            output_response(res, model)
        except sr.UnknownValueError:
            st.write("Sorry, could not understand audio.")
        except sr.RequestError as e:
            st.write(f"Error connecting to Google API: {e}")

input = st.button('say something', on_click=speech_recog)

if prompt:
    response = text_response(prompt, model)
    output_response(response, model)
    