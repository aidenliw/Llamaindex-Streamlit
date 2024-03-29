# 📦 Retrieval Augmented Generation (RAG) System Using LlamaIndex with OpenAI/Llama2 on Streamlit App

## Overview of the App
![app_screenshot](https://github.com/aidenliw/llamaindex-streamlit/blob/main/img/app_screenshot.png?raw=true)
- Simple Chat App using LlamaIndex's RAG system with OpenAI GPT 3.5 turbo / Llama 2 large language model.
- It can answer questions related to the SEP 775 Computational Natural Language Processing Course from Master of Engineering in System and Technology Program at McMaster University.

## Demo App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://llamaindex1.streamlit.app/) <- Click the Button on the left.
```
https://llamaindex1.streamlit.app/
```

## Local Execution

If you want to execute the application locally, please make sure you have done the following steps.
 1. Install Python Enviornment version 3.9 or above on your operating system.
 2. Install all the Python libraries fron the requirements.txt by using the following command in your terminal window:
    - pip install -r /path/to/requirements.txt
 4. Create an .env file on the root directory, save your secret keys into it.
    -  We need an OPENAI_API_KEY from [OpenAI](https://platform.openai.com/api-keys) and a HUGGINGFACE_API_KEY from [Hugging Face](https://huggingface.co/settings/tokens).
    -  The secret keys saved in your .env file should be as follows:
       -  OPENAI_API_KEY='sk-xxx'
       -  HUGGINGFACE_API_KEY='hf_xxx'
 5. Execute the application by using the following command in your terminal window:
    - For OpenAI app: python -m streamlit run streamlit_openai_app.py
    - For Llama2 app: python -m streamlit run streamlit_llama2_app.py
      - Caution! It may takes a while to download the 12 Gigabyte LLM to your computer.
 6. If you want to create your own vector embeddings, you could just simply delete the vectorDB file, and run the index_and_store() functin in the code to generate your own vector stores. 

## Further Reading

Interested in developing the app from scratch? Check out the documentations!
- [LlamaIndex](https://docs.llamaindex.ai/en/stable/)
- [Streamlit](https://blog.streamlit.io/build-a-chatbot-with-custom-data-sources-powered-by-llamaindex/)
- [OpenAI](https://platform.openai.com/docs/overview)
- [HuggingFace](https://huggingface.co/)
- [HuggingFace Llama-2-chat-hf Large Language Model](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
- [HuggingFace BAAI General Embedding (BGE) Model](https://huggingface.co/BAAI/bge-small-en-v1.5)
