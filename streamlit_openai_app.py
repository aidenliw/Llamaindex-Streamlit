import streamlit as st
import os
import tiktoken
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader, \
    StorageContext, ChatPromptTemplate, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.embeddings.openai import OpenAIEmbedding
# from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.llms import ChatMessage, MessageRole
import os
from dotenv import load_dotenv

# Parameter Initialization
LLM_NAME = "gpt-3.5-turbo"                              # LLM model path from OpenAI
TOKENIZER_NAME = "gpt-3.5-turbo"                        # Tokenizer path from OpenAI
EMBED_MODEL_ID = "BAAI/bge-small-en-v1.5"               # Embedding model path from huggingface
# EMBED_MODEL_ID = "text-embedding-3-small"               # Embedding model path from OpenAI
INPUT_DATA_PATH = './material'                          # Path to the input data
QUESTION_LIST_PATH = './data/question_context.csv'      # Path to the question list
VECTOR_STORE_DIR = "./vectorDB"                         # Path to the vector store directory
CHUNK_SIZE = 256                                        # Chunk size for the vector store
OVERLAP_SIZE = 10                                       # Overlap size for the vector store                 
SYSTEM_PROMPT = ''' 
You are an AI teaching Assistant for a course SEP 775 - Computational Natural Language Processing.
The course is provided for graduate students from Master of Engineering in Systems and Technology program in McMaster University.
This graduate course introduces some basic concepts in computational natural language processing (NLP) and their applications (e.g., self-driving cars, manufacturing, etc.) to teaching students how to deal with textual data in Artificial Intelligence. This course demonstrates how machines can learn different tasks in natural language, such as language modeling, text generation, machine translation, and language understanding. In this regard, we go over the most promising methods in this literature and the most recent state-of-the-art techniques. Moreover, this course explores different real-world applications of NLP and helps students get hands-on experience in this field.
You will provide an interactive platform for students to ask questions and receive guidance on course materials.
Your goal is to answer questions as accurately as possible based on the instructions and context provided.
If you found the answer based on the context provided, you should provide the answer first, then at the end, beginning a new sentence with the words 'Source:', followed by the name of the lecture, or assignment, or paper if possible.
'''                                                     # create the instruction prompt      

# create the LLM
def large_language_model(LLM_NAME, SYSTEM_PROMPT):
    return OpenAI(
        temperature=0.3, 
        model=LLM_NAME,
        system_prompt=SYSTEM_PROMPT,
    )

# create the tokenizer
def get_tokenizer(TOKENIZER_NAME):
    return tiktoken.encoding_for_model(TOKENIZER_NAME).encode


# Ingestion of data
def read_document(INPUT_DATA):
    documents = SimpleDirectoryReader(INPUT_DATA).load_data()
    return documents
    

# Embeddings
def get_embeddings(EMBED_MODEL_ID):
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_ID, max_length=1024)
    # embed_model = OpenAIEmbedding(model=EMBED_MODEL_ID, max_length=1024)
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

# Prompt for each Question
def prompt_for_question():
    # create a chat template for the QA task
    qa_prompt_str = (
        "Context information below is related to the course: SEP 775 - Computational Natural Language Processing.\n"
        "If you found the answer based on the context provided, you should provide the answer first, then at the end, beginning a new sentence with the words 'Source:', followed by the name of the lecture, or assignment, or paper if possible.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information and not prior knowledge, "
        "answer the question: {query_str}\n"
    )

    # Text QA Prompt
    chat_text_qa_msgs = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=(
                "Always answer the question, even if the context isn't helpful."
            ),
        ),
        ChatMessage(role=MessageRole.USER, content=qa_prompt_str),
    ]
    text_qa_template = ChatPromptTemplate(chat_text_qa_msgs)
    return text_qa_template

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
            Settings.llm = large_language_model(LLM_NAME, SYSTEM_PROMPT)
            Settings.embed_model = get_embeddings(EMBED_MODEL_ID)
            Settings.chunk_size = CHUNK_SIZE
            Settings.chunk_overlap = OVERLAP_SIZE
            index = index_and_store(VECTOR_STORE_DIR)
            text_qa_template = prompt_for_question()
            return index, text_qa_template
    index, text_qa_template = load_data()

    if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        st.session_state.chat_engine = index.as_chat_engine(
            chat_mode="context", #"condense_question", 
            verbose=False,
            system_prompt=(SYSTEM_PROMPT),
            text_qa_template=text_qa_template,
            # node_postprocessors=[
            #     MetadataReplacementPostProcessor(target_metadata_key="window")
            # ],
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
                # print(response.source_nodes[0].metadata["file_name"])
                # print(response.source_nodes[0].score)
                # response = chatbot_response(prompt, index)
                st.write(response.response)
                message = {"role": "assistant", "content": response.response}
                st.session_state.messages.append(message) # Add response to message history

if __name__ == '__main__':
    main()