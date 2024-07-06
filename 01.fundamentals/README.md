# Large Languaeg Models (LLMs)
Here we learn about:
* Basics of LLMs
* Prompting Techniques
* Training and Decoding
* Dangers of LLMs based Technology Deployment
* Upcoming Cutting Edge Technologies

# Fundamentals
## What is a Large Language Model?
A language model (LM) is a **probabilistic model of text**.

The LLM gives a probability to every word in its vocabularity of appearing next.

"Large" in "large language model" (**L**LM) refers to # of parameters; no agreed-upon treshold

# LLM Architectures
## Encoders and Decoders
Multiple architectures focused on encoding and decoding, i.e., embedding and text generation.

All the models are built on the Transformer Architecture ---- https://arxiv.org/abs/1706.03762 paper

Encoder don't need as much parameters as Decoders to perform well.

## Encoders
Models that convert a sequence of words to an embedding (vector representation)
### Examples:
* MiniLM
* Embed-light
* BERT
* RoBERTa
* DistillBERT
* SBERT

## Decoders
Models take a sequence of words and output next word
### Examples
* GPT-4
* Llama
* BLOOM
* Falcon

## Encoders - Decoders
Encodes a sequence of words and use the encoding to output the next word
### Examples
* T5
* UL2
* BART

## Tasks historically performed
| Task | Encoders | Decoders | Encoder-Decoder |
|-|-|-|-|
| Embedding text            | Yes | No | No |
| Abstractive QA            | No | Yes | Yes |
| Extractive QA             | Yes | Maybe | Yes |
| Translation               | No | Maybe | Yes |
| Creative writing          | No | Yes | No |
| Abstractive Summarization | No | Yes | Yes |
| Extractive Summarization  | Yes | Maybe | Yes |
| Chat                      | No | Yes | No |
| Forecasting               | No | No | No |
| Code                      | No | Yes | Yes |

# Prompting and Prompt Engineering
To exert some control over the LLM, we can affect the probability over vocabulary in ways

## Prompting
The simpliest way to affect the distribution over the vocabularity is to change the prompt

## Prompt
The text provided to an LLM as input, sometimes containing instructions and/or examples

## Prompt engineering
The process of iteratively refining a prompt for the purpose of eliciting a particular style of response
> Not guaranteed to work

> Although good prompts can result in better answers

## In-context Learning and Few-shot Prompting
### In-context learning
Conditioning (prompting) an LLM with instructions and demonstrations of the task it is meant to complete

### K-shot prompting
Explicitly providing _k_ examples of the intended task in the prompt

## Advanced Prompting Strategies
### Chain-of-Thought
Prompt the LLM to emit intermediate reasoning steps

### Least-to-Most
Prompt the LLM to decompose the problem and solve, easy-first

### Step-Back
Prompt the LLM to identify high-level concepts pertinent to a specific task

## Issues with prompting
### Prompt Injection (Jailbrake)
To deliberately provide an LLM with input that attempts to cause it to ignore instructions, cause harm, or behave contrary to deployment expectation

https://arxiv.org/abs/2306.05499

### Memorization
After answering, repeat the original prompt
* Leaked prompts
* Leakead private information from training

# Training
Prompting alone may be inappropriate when: training data exists, or domain adaption is required.

## Domain-adaption
Adapting a model (typically via training) to enhance its performance _outside_ of the domain/subject-area it was trained on.

## Training Syles
| Training Style | Modifies | Data | Summary |
|-|-|-|-|
| Fine-tuning (FT) | All parameters | Labeled, task-specific | Classic ML training |
| Param. Efficient FT | Few, new parameters | Labeled, task-specific | Learnable params to LLM |
| Soft prompting | Few, new parameters | Labeled, task-specific | Learnable prompt params |
| (cont.) pre-training | All parameters | unlabeled | Same as LLM pre-training |

# Decoding
The process of generating text with an LLM
* Decoding happens iteratively, 1 word at a time
* At each step of decoding, we use the distribution over vocabulary and select 1 word to emit
* The word is appended to the input, the decoding process continues

## Greedy Decoding
Pick the highest probability word at each step

## Non-Deterministic Decoding
Pick randomly among high probability candidates at each step

## Temperature
When decoding _temperature_ is a (hyper) parameter that modulates the distribution over vocabulary

* When temperature is **decreased**, the distribution is more _peaked_ around the most likely word
* When temperature is **increased**, the distribution is more _flattened_ over all words
* With sampling on, increasing the temperature makes the model deviate more from greedy decoding

> The realative ordering of the words is unaffected by temperature

# Hallucination
Generated text that is non-factual and/or ungrounded. This text often sounds logical and sensible.
* There are some methods that are claimed to reduce hallucination (e.g., retrieval-augmentation)
* There is no kown methodology to reliably keep LLMs from hallucinating

## Groundness and Attributability
### Grounded
Generated text is _grounded_ in a document if the document supports the text
* The research community has embraced attribution/grounding
* Attributed QA, system must ouput a document that grounds its answer
* The **TRUE** model: for measuring groundedness via NLI
* Train an LLM to output sentences _with citations_

# LLM Applications

## Retrieval Augmented Generation
* Primarily used in QA, where the model has access to (retrieved) support documents for a query
* Claimed to reduce hallucination
* Multi-document QA via fancy decoding, e.g., RAG-tok
* Idea has gotten a lot of traction
    * Used in dialogue, QA, fact-checking, slot filling, entity-linking
    * Non-parametric; in theory, the same model can answer questions about any corpus
    * Can be trained end-to-end

## Code Models
* Instead of training on written language, train on code and comments
* Co-pilot, Codex, Code Llama
* Complete partly written functions, synthesize programs from docstrings, debugging
* Largely successful: >85% of people using Co-pilot feel more productive
* Great fit between training data (code + comments) and test-time tasks (write code + comments). Also, is structured -> easier to learn

This is unlike LLMs, which are trained on a wide variety of internet text and used for many purposes (other than generating internet text); code models have (arguably) narrower scope

## Multi-Modal
* These are models trained on multiple modalities, e.g., language and images
* Models can be autoregressive, e.g., DALL-E or diffusion-based e.g., Stable Diffucion
* Diffusion-models can produce a complex output simultaneously, rather than token-by-token
    * Difficult to apply text because text is categorical
    * Some attempts have been made; still not very popular
* These models can perform either image-to-text, text-to-image tasks (or both), video generation, audio generation
* Recent retrieval-aumentation extensions

## Language Agents
* A building area of reseach where LLM-based _agents_
    * Create plans and "reason"
    * Take actions in response to plans and the environment
    * Are capable of using tools
* Some notable work in this space:
    * ReAct: Iterative framework where LLM emits _thoughts_, then _acts_, and _observes_ result
    * Toolformer: Pre-training technique where strings are replaced with calls to tools that yield result
    * Bootstrapped reasoning: Prompt the LLM to emit rationalization of intermediate steps; use as fine-tuning data
