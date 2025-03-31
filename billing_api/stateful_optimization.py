from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain
import json



def search_with_rag(input):
    '''
    generalized search function with stateless rag method.
    inputs are string pinecone index name for target and string prompt text
    output is generated response. 
    '''
    chat_history = []
    index_name = "billing-codes"
    api_key = 'OPENAI_API_KEY'  
    pc_key  = 'PINECONE_API_KEY'

    embeddings = OpenAIEmbeddings(api_key=api_key)

    vectorstore = PineconeVectorStore(
            index_name=index_name, embedding=embeddings,pinecone_api_key=pc_key
    )

    chat = ChatOpenAI(verbose=True, temperature=0, model_name="gpt-4o",api_key=api_key)

    qa = ConversationalRetrievalChain.from_llm(
        llm=chat, chain_type="stuff", retriever=vectorstore.as_retriever()
    )    
    prompt = "reading the following text, identify the procedure performed from the document, and return it's billing code, price, and description. the output should specifically be only the code and price with text other than the specific description, with the json format(do not include json in the output) : {code:'returned_code',description:'returned_description',unitPrice:returned_price,}:" 
    prompt += input
    res = qa({"question":prompt,"chat_history":chat_history})
    
    history  = (res["question"], res["answer"])
    chat_history.append(history)

    res = json.dumps(res["answer"])
    return res
