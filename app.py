import os
import streamlit as st
from keys import mongoUri
from keys import openAIapikey
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
from pymongo import MongoClient

DB_NAME = "LangChainTest"
COLLECTION_NAME = "test"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "default"

@st.cache_resource
def InitializeDocument():

    # initialize MongoDB python client
    client = MongoClient(mongoUri)
    MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]
    
     #Load the document
    loader = TextLoader("./constitution.txt")
    documents = loader.load()

    #Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    #Get the embeddings for the list of chunks and store in the vectordb
    embeddings = OpenAIEmbeddings()
    return MongoDBAtlasVectorSearch.from_documents(
        chunks,
        embeddings,
        collection=MONGODB_COLLECTION,
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME)

@st.cache_resource
def ConnectMongo():
    embeddings = OpenAIEmbeddings()
    return MongoDBAtlasVectorSearch.from_connection_string(
        mongoUri,
        DB_NAME + "." + COLLECTION_NAME,
        OpenAIEmbeddings(disallowed_special=()),
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME)

def main():
    #os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    os.environ["OPENAI_API_KEY"] = openAIapikey

    st.set_page_config(
        page_title="Chat With A Document",
        page_icon="😺")
    
    #setup the sidebar
    st.sidebar.title("Options")

    #Create the memory object
    if "memory" not in st.session_state:
        st.session_state["memory"]=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    memory=st.session_state["memory"]

    #add a button to the sidebar to start a new conversation
    clear_button = st.sidebar.button("New Conversation", key="clear")
    if (clear_button):
        print("Clearing memory")
        memory.clear()

    #only need to run InitializeDocument once
    #vector_store = InitializeDocument()

    #connect to mongo after document has been loaded
    vector_store = ConnectMongo()

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
    retriever=vector_store.as_retriever()
    crc = ConversationalRetrievalChain.from_llm(llm,
                                                retriever,
                                                memory=memory)

    container = st.container()
    with container:
        with st.form(key="my form", clear_on_submit=True):
            user_input  = st.text_area(label="Question: ", key="input", height = 100)
            submit_button = st.form_submit_button(label="Ask")

        if submit_button and user_input:

            with st.spinner("Thinking..."):
                question = {'question': user_input}
                response = crc.run(question)
            
            #write the ressponse
            st.subheader(response)

            #write the chat history
            variables = memory.load_memory_variables({})
            messages = variables['chat_history']
            for message in messages:
                if isinstance(message, AIMessage):
                    with st.chat_message('assistant'):
                        st.markdown(message.content)
                elif isinstance(message, HumanMessage):
                    with st.chat_message('user'):
                        st.markdown(message.content)
                else:
                    st.write(f"System message: {message.content}")
    
if __name__ == "__main__":
    main()