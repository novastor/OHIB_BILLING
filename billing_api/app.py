import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain

# Initialize FastAPI app
app = FastAPI()

# Request model for input validation
class QueryRequest(BaseModel):
    text: str
    chat_history: Optional[List[List[str]]] = []

# Function to perform RAG-based search
def search_with_rag(input_text: str, chat_history: List[List[str]]):
   '''
    generalized search function with stateless rag method.
    inputs are string pinecone index name for target and string prompt text
    output is generated response. 
    '''
    input_return = input_text["text"]
    
    #add correct keys here
    api_key = os.getenv("OPENAI_API_KEY")
    pc_key = os.getenv("PINECONE_API_KEY")

    if not api_key or not pc_key:
        raise HTTPException(status_code=500, detail="API keys are missing. Set them as environment variables.")

    index_name = "billing-codes"

    # Initialize embeddings and vector store
    embeddings = OpenAIEmbeddings(api_key=api_key)
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings, pinecone_api_key=pc_key)

    # Initialize OpenAI chat model
    chat = ChatOpenAI(verbose=True, temperature=0, model_name="gpt-4o", api_key=api_key)

    # Create RAG chain
    qa = ConversationalRetrievalChain.from_llm(
        llm=chat, chain_type="stuff", retriever=vectorstore.as_retriever()
    )

    # concatenate prompt to instructions and add json formatting
    prompt = (
        "Reading the following text, identify the procedure performed from the document, "
        "and only return its billing code, price, and description. The output must be a valid JSON format  "
        "object: {\"code\": \"returned_code\", \"description\": \"returned_description\", \"unitPrice\": returned_price}."
        "\nText: " + input_text
    )

    # Perform query
    response = qa({"question": prompt, "chat_history": chat_history})

    # Update chat history
    chat_history.append([response["question"], response["answer"]])

    # Ensure response is valid JSON
    try:
        return json.loads(response["answer"])
    except json.JSONDecodeError:
        return {"error": "Invalid JSON response from model"}

# API route for billing query
@app.post("/bill")
def bill_query(request: QueryRequest):
    """
    API endpoint to process billing queries via RAG search.
    
    :param request: JSON containing input text and optional chat history
    :return: JSON response with billing code details
    """
    return search_with_rag(request.text, request.chat_history)
