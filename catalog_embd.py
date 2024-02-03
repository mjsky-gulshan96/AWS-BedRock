import boto3
import os
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

# document upload
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import CSVLoader


from langchain.vectorstores import FAISS
from langchain.indexes.vectorstore import VectorStoreIndexWrapper


region=os.environ.get("AWS_DEFAULT_REGION", 'us-east-1')

# create bedrock client
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name=region
)

model_id="amazon.titan-embed-text-v1"

# - create the Anthropic Model
llm = Bedrock(model_id="anthropic.claude-v2", client=bedrock_client, model_kwargs={'max_tokens_to_sample':200})
bedrock_embeddings = BedrockEmbeddings(model_id=model_id, client=bedrock_client)


loader = CSVLoader('./catalog_data/csv_catalog.csv')

documents = loader.load()

# - in our testing Character split works better with this PDF data set
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 1000,
    chunk_overlap  = 100,
)
docs = text_splitter.split_documents(documents)


avg_doc_length = lambda documents: sum([len(doc.page_content) for doc in documents])//len(documents)
avg_char_count_pre = avg_doc_length(documents)
avg_char_count_post = avg_doc_length(docs)
print(f'Average length among {len(documents)} documents loaded is {avg_char_count_pre} characters.')
print(f'After the split we have {len(docs)} documents more than the original {len(documents)}.')
print(f'Average length among {len(docs)} documents (after split) is {avg_char_count_post} characters.')

# sample embeddings
try:
    sample_embedding = np.array(bedrock_embeddings.embed_query(docs[0].page_content))
    print("Sample embedding of a document chunk: ", sample_embedding)
    print("Size of the embedding: ", sample_embedding.shape)

except ValueError as error:
    if  "AccessDeniedException" in str(error):
        print(f"\x1b[41m{error}\
        \nTo troubeshoot this issue please refer to the following resources.\
         \nhttps://docs.aws.amazon.com/IAM/latest/UserGuide/troubleshoot_access-denied.html\
         \nhttps://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html\x1b[0m\n")      
        class StopExecution(ValueError):
            def _render_traceback_(self):
                pass
        raise StopExecution        
    else:
        raise error

vectorstore_faiss = FAISS.from_documents(
    docs,
    bedrock_embeddings,
)

wrapper_store_faiss = VectorStoreIndexWrapper(vectorstore=vectorstore_faiss)

query = """Stick with classic looks this season and getting dressed will be a breeze.  
               Long sleeve knit cardigan with a button closure.  
             Add a button front top under your cardigan to create a polished look"""

query_embedding = vectorstore_faiss.embedding_function.embed_query(query)
np.array(query_embedding)

relevant_documents = vectorstore_faiss.similarity_search_by_vector(query_embedding)

print(f'{len(relevant_documents)} documents are fetched which are relevant to the query.')
print('----')
for i, rel_doc in enumerate(relevant_documents):
    print(f'## Document {i+1}: {rel_doc.page_content}.......')
    print('---')