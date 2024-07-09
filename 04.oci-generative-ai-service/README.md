# Chatbot Architecture
1. Ask Chatbot a question
2. Relevant documents from storage are retrieved and used as context
3. Prior questions and answers are also used as context
4. LLM answers using context and question

## OCI Generative AI and LangChain Integration
* Using the OCI Generative AI service we can access pretrained models or create and host our own fine-tuned custom models based on our own data on dedicated AI clusters
* langchain_community provides a wrapper class for using OCI Generative AI service as an LLM in LangChain Applications - **langchain_community.llms.OCIGenAI**

## LangChain Components
* **LangChain** is a framework for developing applications powered by language models
* It offers a multitude of components that help us build LLM-powered applications
    * LLMs
    * Prompts
    * Memory
    * Chains
    * Vector Stores
    * Document Loaders

# Models, Prompts and Chains
## LangChain Models
The core element of any language model application is **the model**

### LLM
* LLm in LangChain refer to pure text completion models
* They take a string prompt as input and output a string completion

### Chat Models
* Chat models are often backed by LLMs but are tuned specifically for having conversations
* They take a list of chat messages as input and return an AI message as output

## LangChain Prompt Templates
* Prompt templates are predefined recipes for generating prompts for language models
* Typically, language models expect the prompt to either be a string or else a list of chat messages

### Examples
**PromptTemplate**
> "Tell me a {adjective} joke about {content}."

**ChatPromptTemplate**
> ("human", "Hello, how are you doing?"),
> ("ai", "I'm doing well, thanks!")

## LangChain Chains
LangChain provides frameworks for creating chains of components, including LLMs and other types of components

### Using LangChainExpressionLanguage (LCEL)
* Create chains declaratively using LCEL
* LangChain Expression Language, or LCEL, is a declarative way to easily compose chains together

### Legacy
* Create chains using Python classes like LLM Chain and others

## Setting Up a Development Environment
Steps to set up a Development Environment:
1. Install PyCharm IDE
2. Creae Project
3. Install Packages
4. Copy Config and Key Files
5. Write Code
6. Run Code

# Extending Chatbot by Adding Memory
## LangChain Memory
* Ability to store information about past interactions is "memory"
* Chain interacts with the memory twice in a run
    * After User Input but Before Chain Execution
        * Read from Memory
    * After Core Logic but Before Output
        * Write to Memory
* Various types of memory are available in LangChain
* Data structures and algorithms built on top of chat messages decide what is returned from the memory, e.g., memory might return a succint summary of the past K messages

# Extending Chatbot by Adding RAG
## RAG with LangChain
* **Training Data**: LLMs can reason about a variety of topics, but their knowledge is limited to the data that they are trained on
* **Custom Data**: For building AI applications that can reason about private data, the custom data needs to be given to the model
> The process of fetching the custom information and inserting it into the model prompt is known as Retrieval Augmented Generation (RAG)

## Retrieval Augmented Generation (RAG) with LangChain
LLM has limited knowledge and needs to be augmented with custom data

### Indexing
* Load documents
* Split documents
* Embed and store

## Retrieval and Generation
* Retrieve
* Generate

# Extending Chatbot by Adding RAG + Memory
## RAG Plus Memory
In addition to RAG, we need out Chatbot to be conversational too

### RAG
* Get relevant documents and insert these in prompts

### Memory
* Get the prior chat history and insert it in the prompt
* For that we use a type of a Chain that supports retriever plus memory

# Chatbot Architecture
1. Load and split documents
2. Embed and store documents in Vector DB and persist
3. Load persisted Vector DB
4. Create a retrieval chain using LLM, Retriever, and Memory
5. Invoke chain with a question

## Indexing
1. Doc-loader
2. Text-splitter
3. Embedding
4. Chroma/FAISS DB
5. File Store

## Retrieval and Generation
6. 
    * Load vector store
    * embedding
7. DB
8. 
    * LLM
    * Retriever
    * Memory
9. Chain
10. Streamlit client

# Deploy Chatbot to OCI Compute Instance
We will deployt our chatbot code to a VM
1. Create VM
2. Connect to VM
3. Copy code and document files
4. Install Python
5. Create Virtual Environment and Activate it
6. Install libraries
Create and Run Chroma server
7. Run Chatbot app

# Deploy Chatbot to OCI Data Science
## Deploy LangChain Application to Data Science as Model
We will deployt out Chatbot code to Data Science as a Model
1. Create OCI Gen AI LLM
2. Create LangChain Application
3. Use ChainDeployment class
4. Prepare Model Artifacts
5. Verify the deployment
6. Save Artifacts
7. Deploy the Model
8. Invoke the Model

> Exam tip: read about LangSmith
