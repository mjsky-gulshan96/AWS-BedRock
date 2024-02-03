import boto3
import os
import json
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from reportlab.pdfgen import canvas

# document upload
import numpy as np
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import JSONLoader, DirectoryLoader, TextLoader, PyPDFDirectoryLoader

# vector store through langchin
from langchain.vectorstores import FAISS
from langchain.indexes.vectorstore import VectorStoreIndexWrapper


region = os.environ.get("AWS_DEFAULT_REGION", 'us-east-1')
model_id = "amazon.titan-embed-text-v1"

# create bedrock client
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name=region
)

def get_resp(prompt):
    body = json.dumps({
            "prompt": "\n\nHuman: "+ prompt +"\n\nAssistant:",
            "max_tokens_to_sample": 1000,
            "temperature": 0.5,
            "top_k": 250,
            "top_p": 1,
            "anthropic_version": "bedrock-2023-05-31"
        })
    response = bedrock_client.invoke_model(
            body=body,
            contentType="application/json",
            accept="*/*",
            modelId='anthropic.claude-v2:1'
        )
    data = json.loads(response['body'].read())
    f = open("./category/response.txt", "w")
    f.write(data.get("completion"))
    f.close()
    return data.get("completion")

question = "suggest me some category id based on it and keep the response limited to category id only"

context = """Carrot jeans for kids have a distinct tapering from the hips or waist down to the ankles. This creates a narrow, carrot-like shape. (Varied Fabrics and Washes): Like other types of jeans, This Ben Martin carrot jeans come in a denim fabrics and washes dark, light to suit different tastes and occasions."""

prompt_data = f"""here is a short description of a product provided between ##
{context}

Question: {question}
"""
suggested_category = get_resp(prompt_data)
print(suggested_category)

llm = Bedrock(model_id=model_id, client=bedrock_client, model_kwargs={'max_tokens_to_sample': 200})
bedrock_embeddings = BedrockEmbeddings(model_id=model_id, client=bedrock_client)

def create_pdf_from_text(text):
    output_file="./category/output.pdf"
    # Create a PDF document
    pdf_canvas = canvas.Canvas(output_file)

    # Set font and size
    pdf_canvas.setFont("Helvetica", 12)

    # Split the text into lines and add them to the PDF
    lines = text.split('\n')
    for line in lines:
        pdf_canvas.drawString(100, 750, line)
        pdf_canvas.translate(0, -12)  # Move down by the line height

    # Save the PDF document
    pdf_canvas.save()

text_input = """
Based on the product description mentioning a practical travel 
case that can be pulled or carried, I would categorize this product under 
"mens-accessories-luggage". The description indicates it is a piece of luggage 
that is convenient for travel, which fits closest with the "mens-accessories-luggage" 
category in the provided list. None of the other categories seem to specifically match 
luggage or travel cases based on their names"""

# create_pdf_from_text(text_input)
fDD = open("./category/all_categories.txt", "r")
# loader = TextLoader("./category/all_categories.txt", encoding = 'UTF-8')
# loader = PyPDFDirectoryLoader("./category/")

file_content = fDD.read()
print(file_content)

fDD.close()
# text_loader_kwargs={'autodetect_encoding': True}
# loader = DirectoryLoader("./category/", glob="./*.txt", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
# loader = PyPDFDirectoryLoader("./catalog_data/")
# documents = loader.load()
# print(documents.page_content)
text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap  = 0)
# docs = text_splitter.split_documents(documents)

# print(docs)
vectorstore_faiss = FAISS.from_documents(file_content, bedrock_embeddings)

# wrapper_store_faiss = VectorStoreIndexWrapper(vectorstore=vectorstore_faiss)

# query_embedding = vectorstore_faiss.embedding_function.embed_query("what is the product category ?")
# np.array(query_embedding)

# relevant_documents = vectorstore_faiss.similarity_search_by_vector(query_embedding)

# print(f'{len(relevant_documents)} documents are fetched which are relevant to the query.')
# print('----')
# for i, rel_doc in enumerate(relevant_documents):
#     print(f'## Document {i+1}: {rel_doc.page_content}.......')
#     print('---')

