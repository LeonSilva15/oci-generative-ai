# Core Capabilites
* Pretrained Foundational Models
* Prompt Engineering and LLM Customization
* Fine-tuning and Inference
* Dedicated AI Clusters
* Generative AI Security Architecture

# OCI Generative AI
**Fully managed service** that provides a set of customizable Large Language Models (LLMs) available via a single API to build generative AI applications

### Choice of Models
High performing pretrained foundational models from Meta and Cohere

### Flexible Fine-tuning
Create custom models by fine-tuning foundational models with your own data set

### Dedicated AI Clusters
GPU based compute resources that host your fine-tuning and inference workloads

## How does OCI Generative AI service work?
Text Input -> OCI Generative AI Service -> Text Output
* Built to understand, generate, and process human language at a massive scale
* Use cases:
    * Text Generation
    * Summarization
    * Data Extraction
    * Classification
    * Conversation

## Pretrained Foundational Models
### Text Generation
* Generate text
* Instruction-following Models

| Model | Vendor |
|-|-|
| Command | Cohere |
| Command-light | Cohere |
| Llama 2-70b-chat | Meta |

### Text Summarization
* Summarize text with your instructed format, length, and tone

| Model | Vendor |
|-|-|
| Command | Cohere |

### Embedding
* Convert text to vector embeddings
* Semantic Search
* Multilingual Models

| Model | Vendor |
|-|-|
| embed-english-v3.0 embed-multilingual-v3-0 | Cohere |
| embed-english-light-v3.0 embed-multilingual-light-v3-0 | Cohere |
| embed-english-light-v2-0 | Cohere |

## Fine-tuning
* Optimizing a pretrained foundational model on a smaller domain-specific dataset
    * Improve Model Performance on specific tasks
    * Improve Model Efficiency
* Use when a pretrained model doesn't perform your task well or you want to teach it something new
* OCI Generative AI uses the T-Few fine-tuning to enable fast and efficient customizations

## Dedicated AI Clusters
* Dedicated AI clusters GPU based compute resources that host the customer's fine-tuning and inference workloads
* Gen AI service establishes a dedicated AI cluster, which includes dedicated GPUs and an exclusive RDMA cluster network for connecting the GPUs
* The GPUs allocated for a customer's generative AI tasks are isolated from other GPUs

# Generation Models
## Tokens
* Language models undersand "tokens" rather than characters
* One token can be a part of a word, an entire word, or punctuation
    * A common word such as "apple" is a token
    * A word such as "friendship" is made up of two tokens - "friend" and "ship"
* Number of Tokens/Word depend on the complexity of the text
    * Simple text: 1 token/word (Avg.)
    * Complex text (less common words): 2-3 tokens/word (Avg.)

> Many words map to one token, but some don't: indivisible.

## Pretrained Generation Models in Generative AI
### command - Cohere
* Highly performant, instruction-following conversational model
* Model Parameters: 52B, context window: 4096 tokens
* Use cases: text generation, chat, text summarization

### command-light - Cohere
* Smaller, faster version of Command, but almost as capable
* Model Paramaters: 6B, context window: 4096 tokens
* Use when speed and cost are important (give clear instructions for the best results)

### llama-2-70b-chat - Meta
* Highly performant, open-source model optimized for dialog use cases
* Model Parameters: 70B, context window: 4097 tokens
* Use cases: chat, text generation

## Generation Model Parameters
### Maximum Output Tokens
Max number of tokens model generates per response

### Temperature
Determines how creative the model should be; close second to prompt engineering in controlling the ouput of generation models

### Top p, Top k
Two additional ways to pick the output tokens besides temperature

### Presence/Frequency Penalty
Assign a penalty when a token appears frequently and produces less repetitive text

### Show likelihoods
Determines how likely it would be for a token to follow the current generated token

## Temperature
Temperature is a (hyper) parameter that controls the randomness of the LLM output
* Temperature of **0** makes the model deterministic (limits the model to use the word with the highest probability)
* When temperature is **increased**, the distribution is _flattended_ over all words
* With increased temperature, model uses words with lower probabilities

## Top k
Top **k** tells the model to pick the next token from the top '**k**' tokens in its list, sorted by probability
* If **Top k** is set to _3_, model will only pick from the top 3 options and ignore all others

## Top p
Top **p** is similar to Top k but picks from the top tokens based on the sum of their probabilities
* If **p** is set as .15, then it will only pick from the options that their probabilities add up to the nearest below 15%
* If **p** is set to .75, the bottom 25% of probable outputs are excluded

## Stop Sequences
* A stop sequence is a string that tells the model to stop generating more content
* It is a way to control your model output
* If a period (.) is used as a stop sequence, the model stops generating text once it reaches the end of the first sentence, even if the number of tokens limit is much higher

## Frequency and Presence Penalties
* These are useful if you want to get rid of repetition in your outputs
* **Frequency penaly** penalizes tokens that have already appeared in the preceding text (including the prompt), and scales based on how many times that token has appeard
* So a token that has already appeard 10 times gets a higher penalty (wich reduces its probability of appearing) that a token that has appeard only once
* **Presence penalty** applies the penalty regardless of frequency. As long as the token appeard once before, it will get penalized

## Show Likelihooods
* Every time a new token is to be generated, a number between -15 and 0 is assigned to all tokens
* Tokens with higher numbers are more likely to follow the current token

# Summarization Model
### command - Cohere
* Generates a succinct version of the original text that relays the most important information
* Same as one of the pretrained text generation models, but with parameters that you can specify for text summarization
* Use cases include, but not limited to:
    * News articles
    * Blogs
    * Chat transcripts
    * Scientific articles
    * Meetings notes
    * any text that we should like to see a summary of

## Summarization Model Parameters
### Temperature
Determines how creative the model should be; Default temperature is 1 and maximum temperature is 5

### Length
Approximate length of the summary. Choose from
* Short
* Medium
* Long

### Format
Whether to display the summary in a free-form paragraph or in bullet points

### Extractiveness
How much to reuse the input in the summary. Summarizes with high extractiveness with low extractiveness tend to paraphrase

# Embedding Models
## Embeddings
* Embeddings are **numerical representations** of a piece of text converted to numer sequences
* Apiece of text vould be a word, phrase, paragraph or one or more paragraph
* Embeddings make it easy for computers to understand the relationship between pieces of text

## Word Embeddings
* Word Embeddings capture properties of the word
* Embeddings represent more properties (coordinates) than just two
* These rows of coordinates are called vectores and represented as numbers

## Semantic Similarity
* Cosine and Dot Product Similarity can be used to compute **numerical similarity**
* Embeddings that are **numerically similar** are also **semantically similar**

## Sentence Embeddings
* A sentence embedding associates every sentence with a vector of numbers
* Similar sentences are assigned to similar vectors, different sentences are assigned to different vectors
* The eembedding vector of "canine companions say" will be more similar to the embedding vector of "woof" than that of "meow"

## Embedding Models in Generative AI
* Cohere.embed-english converts English text into vector embeddings
* Cohere.embed-english-lights is the smaller and faster version of embed-english
* Cohere.embed-multilingual is the state-of-the-art multilingual embedding model that can convert text in over 100 languages into vector embeddings

### Use cases
* Semantic search
* Text classification
* Text clustering

## Embedding Models in Generative AI
### Cohere embed-english-v3.0
### Cohere embed-multilingual-v3.0
* English and Multilingual
* Model creates a 1024-dimensional vector for each embedding
* Max 512 tokens per embedding

### Cohere embed-english-light-v3.0
### Cohere embed-multilingual-light-v3.0
* Smaller, faster version; English and Multilingual
* Model creates a 384-dimensional vector for each embedding
* Max 512 tokens per embedding

### Cohere embed-english-light-v2.0
* Previous generation models, English
* Model creates a 1024-dimensional vector for each embedding
* Max 512 tokens per embedding

# Prompt Engineering
## Prompt
The input or initial text provided to the model

## Prompt Engineering
The process of iteratively refining a prompt for the purpose of eliciting a particular style of response

## LLMs as next word predictors
1. Text prompts are how users interact with Large Language Models
2. LLM models attempt to produce the next series of words that are most likely to follow from the previous text

## Aligning LLMs to follow instructions
* Completion LLMs are trained to predict the next word on a large dataset of Internet text, rather than to safely perform the language task that the user wants
* Cannot give instructions or ask questions to a completion LLM
* Instead, need to formulate your input as a prompt whose natural continuation is your desired output

## In-context Learning and Few-shot Prompting
### In-context learning
Conditional (prompting) and LLM with instructions and or demonstrations of the task it is meant to complete

### k-shot prompting
Explicitly providing _k_ examples of the intended task in the prompt

> Few-shot prompting is widely belived to improve results over 0-shot prompting

## Prompt Formats
Large Language Models are trained on a specific prompt format. If we format prompts in a different way, our way gets odd/inferior results

## Advanced Prompting Strategies
### Chain-of-Tought
Provide examples in a prompt is to show responses that include a reasoning step

### Zero Shot Chain-of-Tought
Apply chain-of-thought prompting without providing examples

# Customize LLMs with your data
## Training LLMs from scratch with my data?
* **Cost**: Expensive - $1M per 10B parameters to train
* **Data**: A lot of data is needed - E.g. Meta's Llama-2 7B model was trained on 2 trillion tokens (1T tokens ~ 20M novels ~ 1B legal briefs). And we need a lot of **annotated data**
* **Expertise**: Pretraining models is hard - requires a thorough understanding of model performance, how to monitor for it, detect and mitigate hardware failures, and understand the limitations of the model

## In-context Learning / Few shot Prompting
User provides demonstrations in the prompt to teach the model how to perform certain tasks
* Popular techniques include **Chain of Thought Prompting**
* Main limitation: Model Context Length

## Fine-tuning a pretrained model
Optimize a model on a smaller domain-specific dataset
* Recommended when a pretrained model doesn't perform your task well or when you want to teach it something new
* Adapt to specific style and tone, and learn human preferences

### Fine-tuning Benefits
* Improve Model Performance on specific tasks
    * More effective mechanism of improving model performance than Prompt Engineering
    * By customizing the model to domain-specific data, it can better understand and generate contextually relevant responses
* Improve Model Efficiency
    * Reduce the number of tokes needed for your model to perform well on your tasks
    * Condense the expertise of a large model into a smaller, more efficient model


## Retrieval Augmente Generation (RAG)
* Language model is able to query enterprise knowledge bases (databases, wikis, vector databases, etc.) to provide grounded responses
* RAGs do not require custom models

## Customize LLMs with your data
<table>
    <tr>
        <th>Method</th>
        <th>Description</th>
        <th>When to use?</th>
        <th>Pros</th>
        <th>Cons</th>
    </tr>
    <tr>
        <th>Few shot Prompting</th>
        <td>Provide examples in the prompt to steer the model to better performance</td>
        <td>LLM already understands topics that are necessary for the text generation</td>
        <td>
            <ul>
                <li>Very simple</li>
                <li>No training cost</li>
            </ul>
        </td>
        <td>Adds latency to each model request</td>
    </tr>
    <tr>
        <th>Fine-tuning</th>
        <td>Adapt a pretrained LLM to perform a specifc task on private data</td>
        <td>
            <ul>
                <li>LLM does not perform well on a particular task</li>
                <li>Data required to adapt the LLM is too large for prompt engineering</li>
                <li>Latency with the current LLM is too high</li>
            </ul>
        </td>
        <td>
            <ul>
                <li>Increase in model performance on a specific task</li>
                <li>No impact on model latency</li>
            </ul>
        </td>
        <td>Requires a labeled dataset which can be expensive and time-consuming to acquire</td>
    </tr>
    <tr>
        <th>RAG</th>
        <td>Optimize the output of a LLM with targeted information without modifying the underlying model itself</td>
        <td>
            <ul>
                <li>When the data changes rapidly</li>
                <li>When we want to mitigae hallucinations by grounding answers in enterprise data (improve auditing)</li>
            </ul>
        </td>
        <td>
            <ul>
                <li>Access the latest data</li>
                <li>Ground the result</li>
                <li>Does not require fine-tuning jobs</li>
            </ul>
        </td>
        <td>
            <ul>
                <li>More complex to setup</li>
                <li>Requires a compatible data source</li>
            </ul>
        </td>
    </tr>
</table>

* Prompt Engineering is the easiest to start with; test and learn quickly
* If we need more context, then use Retrieval Augmented Generation (RAG)
* If we need more instruction following, then use Fine-tuning

> Sometimes we might need to use more than one

1. Start with a simple Prompt
2. Add Few shot Prompting
3. Add simple retrieval using RAG
4. Fine-tune the model
5. Optimize the retrieval on fine-tuned model

# Fine-Tuning and Inference in OCI Generative AI
## Fine-tuning and Inference
* A model is fine-tuned by taking a pretrained foundational model and providing additional training using custom data
* In Machine Learning, Inference referes to the process of using a trained ML model to make predictions or decisions based on new input data
* With language models, inference refers to the model receiving new text as input and generated output text based on what it has learned during training and fine-tuning

## Fine-tuning workflow in OCI Generative AI
**Custom Model**: A model that we can create by using a **Pretrained Model** as a base and using our own **dataset** to fine-tune that model

1. Create a **Dedicated AI Cluster** (Fine-tuning)
2. Gather Training Data
3. Kickstart Fine-tuning
4. **Fine-tuned (custom) Model** gets created

## Inference workflow in OCI Generative AI
**Model Endpoint**: A designed point on a **Dedicated AI Cluster** where a large language model can accept user requests and send back responses such as the model's generated text

1. Create a **Dedicated AI Cluster** (Hosting)
2. Create Endpoint
3. **Serve Model**

## Dedicated AI Clusters
* Effectively a single-tenant deployment where the GPUs in the cluster only host our custom models
* Since the model endpoint isn't shared with other customers, the model throughput is consistent
* The minimum cluster size is easier to estimate based on the expected throughput
* Cluster Types
    * **Fine-tuning**: used for _training_ a pretrained foundational model
    * **Hosting**: used for hosting a custom model endpoint for _inference_

## T-Few Fine-tuning
* Traditionally, Vanilla fine-tuning involves updating the weights of all (most) the layers in the model, requiring longer training time and higher serving (inference) costs
* T-Few fine-tuning selectively updates **only a fraction of the model's weights**
    * T-Few fine-tuning is an additive Few-Shot Parameter Efficient Fine Tuning (PEFT) technique that inserts additional layers, comprising ~0.01% of the baseline model's size
    * The weight updates are localized to the T-Few layers during the fine-tuning process
    * Isolating the weight updates to these T-Few layers significantly reduces the overall training time and cost compared to updating all layers

### T-Few fine-tuning process
* T-Few fine-tuning process being by utilizing the initial weights of the base model and an annotated training dataset
* Annotated data comprises of input-ouput pairs employed in supervised training
* Supplementary set of model weights is generated (~0.01% of the baseline model's size)
* Updates to the weights are confined to a specific group of transformer layers, (T-Few transformer layers), saving substantial training time and cost

## Reducing Inference costs
* Inference is computationally expensive
* Each Hosting cluster can host one Base Model Endpoint and up to N Fine-tuned Custom Model Endpoints serving requests concurrently
* This approach of models sharing the same GPU resources reduces the expenses associated with inference
* Endpoints can be deactivated to stop serving requests and re-activated later

## Inference serving with minimal overhead
* GPU memory is limited, so switching between models can incur significant overhead due to reloading the full GPU memory
* These models share the majority of weights, with only slight variations; can be efficiently deployed on the same GPUs in a dedicated AI cluster
* This architecture results in minimal overhead when switching between models derived from the same base model

# Dedicated AI Cluster Units
| Unit Size | Base Model | Description | Limit Name |
|---|---|---|---|
| Large Cohere | cohere.command | Dedicated AI cluster unit, either for **hosting** or **fine-tuning** the cohere.command model | dedicated-unit-large-cohere-count |
| Small Cohere | cohere.command-light | Dedicated AI cluster unit, either for **hosting** or **fine-tuning** the cohere.command-light model | dedicated-unit-small-cohere-count |
| Embed Cohere | cohere.embed | Dedicated AI cluster unit, either for **hosting** the cohere.embed models | dedicated-unit-embed-cohere-count |
| Llama2-70 | llama2-70b-chat | Dedicated AI cluster unit, either for **hosting** the Llama2 models | dedicated-unit-large-llama2-70-count |

# Dedicated AI Cluster Units Sizing
| Capability | Base Model | Fine-tuning Dedicated AI Cluster | Hosting Dedicated AI Cluster |
|---|---|---|---|
| Text Generation | cohere.command | Unit size: Large Cohere - **Required units: 2** | Unit size: Large Cohere - Required units: 1 |
| Text Generation | cohere.command-light | Unit size: Small Cohere - **Required units: 2** | Unit size: Small Cohere - Required units: 1 |
| Text Generation | llama2_70b-chat | - | Unit size: Llama2-70 - Required units: 1 | Unit size: Small Cohere - Required units: 1 |
| Summarization | cohere.command | - | Unit size: Cohere - Required units: 1 | Unit size: Small Cohere - Required units: 1 |
| Embedding | cohere.command | - | Unit size: Cohere - Required units: 1 | Unit size: Small Cohere - Required units: 1 |

## Dedicated AI Cluster Sizing
* **Fine-tuning** Dedicated AI Cluster
    * Requires **two units** for the base model chosen
    * Fine-tuning a model requires more GPUs than hosting a model (therefore, two units)
    * The same fine-tuning cluster can be used to fine-tune several models
* **Hosting** Dedicated AI Cluster
    * Requires **one unit** for the base model chosen
    * Same cluster can host up to 50 different fine-tuned models (using T-Few fine tuning)
    * Can create up to 50 endpoints that point to the different models hosted on the same hosting cluster

## Pricing
* Minimum Commitment
    * Min Hosting commitment: **744 unit-hours/cluster**
    * Min Fine-tuning commitment: **1 unit-hour/fine-tuning job**

### Example:
* Unit Hours for each Fine-tuning
    * Each fine-tuning cluster requires **two units** and each cluster is active for **five hours**
    * fine tuning per cluster = **10 unit-hours**

* Fine-tuning Cost
    * **Fine-tuning** cost/month = (10 unit-hours)/week x (4 weeks) x $_large-cohere-dedicated-unit-per-hour-price_

* Hosting Cost
    * **Hosting** cost/month = (744-unit-hours) x $_large-cohere-dedicated-unit-per-hour-price_

* Total Cost
    * Total cost/month = (40 + 744 unit-hours) x $_large-cohere-dedicated-unit-per-hour-price_

# Fine-tuning Configuration
* Training Methods
    * Vanilla: Traditional fine-tuning method
    * T-Few: Efficient fine-tuning method
* Hyperparameters
    * Total Training Epochs
    * Learning Rate
    * Training Batch Size
    * Early Stopping Patience
    * Early Stopping Threshold
    * Log Model metrics interval in steps
    * Number of last layers (Vanilla)

## Fine-tuning Parameters (T-Few)
| Hyperparameter | Description | Default value |
|---|---|---|
| Total Training Epochs | The number of iterations through the entire training dataset | Default (3) |
| Batch Size | The number of samples processed before updating model parameters | 8 (cohere.command), an integer between 8-16 for cohere.command-light |
| Learning Rate | The rate at wich parameters are updated after each batch | Default (0.1, T-Few) |
| Early Stopping Threshold | The minimum improvement in loss required to prevent premature termination of the training process | Default (0.01) |
| Early Stopping Patience | The tolerance for stagnation in the loss metric before stopping the training process | Default (6) |
| Log Model metrics interval in steps | Determines how frequently to log model metrics. Every step is logged for the first 20 steps and then follows this parameter for log frequency | Default (10) |

## Understanging Fine-tuning Results
### Accuracy
* Accuracy is a measure of how many predictions the model made correctly out of all the predictions in an evaluation
* To evaluate generative models for accuracy, we ask it to predict certain words in the user-uploaded data

### Loss
* Loss is a measure that describes how bad or wrong a prediction is
* Accuracy may tell you how many predictions the model got wrong, but it will not describe how incorrect the wrong predictions are
* To evaluate generative models for loss, we ask the model to predict ceratin words in the user-provided data and evaluate how wrong the incorrect predictions are
* Loss should decrease as the model improves

# OCI  Generative AI Security
## Dedicated GPU and RDMA Netword
* Security and privacy of customer workloads is an essential design tenet
* GPUs allocated for a customer's generative AI tasks are isolated form other GPUs

## Model Endpoints
* For strong data privacy and security, a dedicated GPU cluster only hanldes fine-tuned models of a single customer
* Base model + fine-tuned model endpoints share the same cluster resources for the most efficient utilization of underlying GPUs in the dedicated AI cluster

## Customer Data and Model Isolation
* Customer data access is restricted within the customer's tenancy, so that one customer's data can't be seen by another customer
* Only a customer's application can access custom models created and hosted from within that customer's tenancy

## Generative AI leverages OCI Security Services
* Leverages OCI IAM for Authentication and Authorization
* OCI Key Management Service stores base model keys securely
* The fine-tuned customer models weights are stored in OCI Storage buckets (encrypted by default)
