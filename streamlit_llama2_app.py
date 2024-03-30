import streamlit as st
import os
import torch
from transformers import BitsAndBytesConfig
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader, \
    StorageContext, load_index_from_storage
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
from dotenv import load_dotenv

# Parameter Initialization
LLM_NAME = "meta-llama/Llama-2-7b-chat-hf"              # LLM model path from huggingface
TOKENIZER_NAME = "meta-llama/Llama-2-7b-chat-hf"        # Tokenizer path from huggingface
EMBED_MODEL_ID = "BAAI/bge-small-en-v1.5"               # Embedding model path from huggingface
INPUT_DATA_PATH = './material'                          # Path to the input data
QUESTION_LIST_PATH = './data/question_context.csv'      # Path to the question list
VECTOR_STORE_DIR = "./vectorDB"                         # Path to the vector store directory
CHUNK_SIZE = 256                                        # Chunk size for the vector store
OVERLAP_SIZE = 10                                       # Overlap size for the vector store
SYSTEM_PROMPT = """                                     
You are an AI teaching Assistant for the SEP 775 course. 
You will provide an interactive platform for students to ask questions and receive guidance on course materials.
Your goal is to answer questions as accurately as possible based on the instructions and context provided.
If you found the answer based on the context provided, you should provide the answer first, then at the end, beginning a new sentence with the words "Source:", followed by the name of the lecture, or assignment, or paper if possible.
"""                                                     # create the instruction prompt


# create the LLM
def large_language_model(LLM_NAME, TOKENIZER_NAME, SYSTEM_PROMPT):
    
    query_wrapper_prompt = PromptTemplate(
        "[INST]<<SYS>>\n" + SYSTEM_PROMPT  + "<</SYS>>\n\n{query_str}[/INST]"
    )
    
    # quantize to save memory
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,                                  # load in 4 bit
        bnb_4bit_compute_dtype=torch.float16,               # compute in float16
        bnb_4bit_quant_type="nf4",                          # quantize to 4 bit
        bnb_4bit_use_double_quant=True,                     # use double quantization
    )
    
    llm = HuggingFaceLLM(
        model_name=LLM_NAME,                                        # model name
        tokenizer_name=TOKENIZER_NAME,                              # tokenizer name
        context_window=3900,                                        # max context window to store tokens
        max_new_tokens=256,                                         # max new tokens to generate
        model_kwargs={"quantization_config": quantization_config},  # quantization config
        generate_kwargs={
            "temperature": 0.3,                                    # randomness of the output
            "top_p": 0.95,                                          # sampling from the most likely p tokens
            "top_k": 50                                             # only sample from the top k tokens
            },
        query_wrapper_prompt=query_wrapper_prompt,                  # query wrapper prompt
        device_map="auto",                                          # use the default device
    )
    return llm


# Ingestion of data
def read_document(INPUT_DATA):
    documents = SimpleDirectoryReader(INPUT_DATA).load_data()
    return documents
    

# Embeddings
def get_embeddings(EMBED_MODEL_ID):
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_ID, max_length=1024)
    return embed_model


# Indexing and Storing to Vector Store
def index_and_store(VECTOR_STORE_DIR):
    # check if storage already exists
    if not os.path.exists(VECTOR_STORE_DIR):
        print("Creating New DB...")
        index = VectorStoreIndex.from_documents(read_document(INPUT_DATA_PATH))
        # store it for later
        index.storage_context.persist(persist_dir=VECTOR_STORE_DIR)
    else:
        print("Using Existing DB...")
        # load the existing index
        storage_context = StorageContext.from_defaults(persist_dir=VECTOR_STORE_DIR)
        index = load_index_from_storage(storage_context=storage_context)
    return index


# chatbot response
def chatbot_response(question, index):
    query_engine = index.as_query_engine()
    response = query_engine.query(question)
    return response


# run the streamlit app
def main():
    st.set_page_config(page_title="Chat with AI Teaching Assistant, powered by LlamaIndex & OpenAI", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)

    st.title("SEP 775 AI Teaching Assistant, powered by LlamaIndex & OpenAI 💬🦙")
    st.info("Ask me a question about the [SEP 775 Computational Natural Langurage Processing](https://drive.google.com/file/d/1YpVo9bxj2aYcwEjEIce3YZXt5dny0869/view?usp=sharing) Course!", icon="📃")
            
    if "messages" not in st.session_state.keys(): # Initialize the chat messages history
        st.session_state.messages = [{
            "role": "assistant", 
            "content": "Hi! How can I help you today?"
        }]

    @st.cache_resource(show_spinner=False)
    def load_data():
        with st.spinner(text="Loading and document vectors – hang tight! This should take few seconds."):
            load_dotenv()   # Load the environment variables
            Settings.llm = large_language_model(LLM_NAME, TOKENIZER_NAME, SYSTEM_PROMPT)
            Settings.embed_model = get_embeddings(EMBED_MODEL_ID)
            Settings.chunk_size = CHUNK_SIZE
            Settings.chunk_overlap = OVERLAP_SIZE
            index = index_and_store(VECTOR_STORE_DIR)
            return index
    index = load_data()

    if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        st.session_state.chat_engine = index.as_chat_engine(
            chat_mode="context", #"condense_question", 
            verbose=True,
            system_prompt=(SYSTEM_PROMPT),
            )

    if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

    for message in st.session_state.messages: # Display the prior chat messages
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # If last message is not from assistant, generate a new response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chat_engine.chat(prompt)
                # response = chatbot_response(prompt, index)
                st.write(response.response)
                message = {"role": "assistant", "content": response.response}
                st.session_state.messages.append(message) # Add response to message history

if __name__ == '__main__':
    main()