# Retrieval Augemented Generation (RAG)
* Retrieval Augmented Generation (RAG) is a method for generating text using additional information fetched from an external data source
* RAG models retrieve documents and pass them to a **seq2seq** model

## RAG Framework
### Retriever
* **Function**: Sources relevant information from a large corpus or database
* **Working**: Uses retrieval techniques
* **Purpose**: Provides the generation system with contextually relevant, accurate, and up-to-date information that might not be present in the model's pre-trained knowledge

### Ranker
* **Function**: Evaluates and prioritizes the information retrieved by the retriever
* **Working**: Uses various algorithms to evaluate the quality of the retrieved content
* **Purpose**: Ensures that the generation system receives the most pertinent and high-quality input

### Generator
* **Function**: Generates human-like text based on the retrieved and ranked information, along with the user's original query it receives
* **Working**: Uses generative models to craft human-like text that is contextually relevant, coherent, and informative
* **Purpose**: Ensures that the final response is factually accurate, and also coherent, fluent, and styled typically like human language

## RAG Techniques
### RAG Sequence
* For each input query (like a chapter topic), the model retrieves a set of relevant documents or information
* It then considers all these documents together to generate a single, cohesive response (the entire chapter) that reflects the combined information

### RAG Token
* For each part of the response (like each sentence or even each word), the model retrieves relevant documents
* The response is constructed incrementally, with each part reflecting the information form the documents retrieved for that specific part

## RAG Pipeline
### 1. Ingestion
1. Documents
2. Chunks
3. Embedding
4. Index (Database)

### 2. Retrieval
1. Query
2. Index
3. Top K Results

### 3. Generation
1. Top K Results
2. Response to User

## RAG Evaluation
### RAG Triad
1. Context
    * **Context Relevance**: Is the retrieved context relevant to the query?
2. Response
    * **Groundedness**: Is the response supported by the context?
3. Query
    * **Answer Relevance**: Is the answer relevant to the query?

# Vector Databases
## LLM VS LLM + RAG
* LLM without RAG rely on internal knwoledge learned during pre-trained on a large corpus of text. It may or may not use Fine-tuning
* LLMs with RAG use an external database, which is a Vector Database

## Vector
* A vector is a sequence of numbers called dimensions used to capture the important "features" of the data
* Embeddings in LLMs are essentially high-dimensional vectors
* Vectors are generated using deep learning embedding models and represent the semantic content of data, not the underlying words or pixels

> Optimized for multidimensional spaces where the relationship is based on distnaces and similarities in a high-dimensional vector space

## Embedding Distance
* Dot Product - Magnitud and Direction
* Cosine Distance - Distance difference (angle)

## Similar Vectors
* **K-Nearest Neighbors (KNN)** algorithm can be used to perform a vector or semantic search to obtain nearest vectors in embedding space to a query vector
* **Approximate Nearest Neighbors (ANN)** algorithms are designed to find near-optimanl neighbors much faster than exact KNN searches
* **ANN** methods such as **HNSW**, **FAISS**, **Annoy** are often preferred for large-scale similarity search tasks in embedding spaces due to their efficiency

## Vector Database Workflow
1. Vectors
2. Indexing
3. Vector Database
4. Querying
5. Post Processing

## Vector Databases Advantages
* Accuracy
* Latency
* Scalability

## Role of Vector Database with LLMs
* Address the hallucinatino (i.e., inaccuracy) problem inherent in LLM responses
* Augment prompt with enterprise-specific content to produce better responses
* Avoid exceeding LLM token limits by using most relevant content
* Cheaper than fine-tuning LLMs, which can be expensive to update
* Real-time updated knowledge base
* Cache previous LLM prompts/responses to improve performance and reduce costs

# Keyword Search
## Keyword Search
* Keywords are words used to match with the terms people are searching for, when looking for products, services, or general information
* Simplest form of search baed on exact matches of the user-provided keywords in the database or index
* Evaluates documents based on the presence and frecuency of the query term.

# Semantic Search
## Search by meaning
* Retrieval is done by understanding intent and context, rather than matching keywords
* Ways to do this:
    * **Dense Retrieval**: Uses text embeddings
    * **Reranking**: Assigns a relevance score

## Embeddings:
* Embeddings represent the meaning of text as a list of numbers
* Capture the essence of the data in a lower-dimensional space while maintaining the semantic relationships and meaning

## Dense Retrieval
* Relies on embeddings of both queries and documents to identify and rank relevant documents for a given query
* Enables the retrieval system to understand and match based on the contextual similarities between queries and documents

## Rerank
* Assigns a relevance score to (query, response) pairs from initial search results
* High relevance score pairs are more likely to be correct
* Implemented through a trained LLM

## Hybrid Search
Handles both Sparse and Dense Vectors
