# Eriss: Reflective Causal Language Model (Unreleased Prototype)

This repository contains an **unreleased, experimental prototype** from several years ago.  
It was an attempt to **refactor `LlamaForCausalLM`** into a reflective, memory-augmented model called **`ErissForCausalLM`**, using the **default Llama training dataset and tokenizer** at the time.  

It was never cleaned up, never productionized, and never intended for release.  
This README captures the full architecture and design exactly as it existed during the prototype stage.

---

## Overview

`ErissForCausalLM` is a modified causal language model built on top of a Llama-style transformer.  
It behaves like a normal `CausalLM` model externally, but internally performs a multi-step reasoning cycle:

1. Generate **internal thoughts** before responding  
2. Maintain long-term memory via **SQLite**  
3. Maintain a persistent **sense of self**, **sense of user**, and **set of objectives**  
4. Maintain a multi-turn **context window** outside of the active prompt  
5. Optionally infer **next actions**  

This was an exploratory research direction: embedding “reflection” directly into the forward pass of a causal LM.

Everything described below comes **from the original prototype**, preserved as-is.

---

## File Location

The implementation lived here:

transformers/models/eriss/modeling_eriss.py

markdown
Copy code

The class inherits from an internal `ErissPreTrainedModel` and wraps a base `ErissModel` transformer.

---

## High-Level Interaction Flow

A single user interaction runs through the following stages:

### 1. User Query Arrives
- Model receives raw `input_ids`
- Embeds the query  
- Retrieves relevant memories by cosine similarity from a SQLite `memory_buffer` table

### 2. Construct Thought Prompt
A long token sequence is assembled using **pre-tokenized fragments stored as NumPy arrays**:

- System prompt (first-person or third-person variant)
- Suffix indicating “generate thoughts”
- Sense of self
- Sense of user
- Current objectives
- Context window from previous interactions
- Retrieved relevant memories
- Current user query
- A dedicated “thought prompt”

### 3. Generate Internal Thoughts
- `self.generate(...)` is run  
- Thought tokens are post-processed  
- Thoughts are appended to the context window  
- Thoughts are logged token-by-token into a dedicated `thoughts` table  
- Thoughts are saved into the active interaction record

### 4. Construct Response Prompt
A second token sequence is built:

- System response prompt  
- The generated thoughts  
- A bridge sequence  
- The response prompt  
- The user’s query  
- An end-of-turn marker

### 5. Generate Visible Response
- Another `generate(...)` call produces the user-visible answer  
- Tokens accumulate in `self.interaction_ai_response` during streaming

### 6. Update Memory and Model State
After the visible response:

- Embed the combined interaction (query + thoughts + response)  
- Insert the interaction row into `memory_buffer`  
- Update sense of self  
- Update sense of user  
- Recompute objectives  
- Insert everything into the context window for long-term history  
- Optionally infer an “action” from recent context

---

## SQLite Memory System

This prototype stores everything in a local SQLite file:

eriss-prime.db

yaml
Copy code

Tables created:

### `thoughts`
Stores individual thought tokens as they stream.

### `sense_of_self`
Stores versions of the model’s evolving self-description.

### `sense_of_user`
Parallel table for the model’s evolving understanding of the user.

### `objectives`
Each row contains one objective as a JSON-encoded token list.

### `memory_buffer`
Stores all prior interactions, each containing:
- embedding  
- user query tokens  
- thought tokens  
- response tokens  
- timestamp tokens  

Similarity search is done by embedding the current query, then performing cosine similarity against all stored rows using NumPy.

### `context_window`
Stores compact token sequences describing:
- timestamp  
- user query  
- thoughts  
- ai response  
- sense of self  
- sense of user  
- objectives  
- action  
- token count  

This table is used to reconstruct a small rolling history for future prompts.

### `actions`
Intended to log tool or action decisions (not fully implemented in the prototype).

---

## Prompt Architecture

All prompts are stored as **pre-tokenized NumPy arrays**, not strings.

These arrays include:

- System prompts  
- Thought suffix prompts  
- Response suffix prompts  
- Headers (for user queries, thoughts, objectives, actions)  
- Sense of self prompts  
- Sense of user prompts  
- Objective prompts  
- Action prompts  
- Bridge tokens  
- End-of-turn tokens  
- Separator tokens  

Special chat tokens include IDs like:
`128000`, `128001`, `128006`, `128007`, `128009`, etc.

Helper functions manipulate these arrays:

- `split_by_last_occurrence`  
- `clean_special_tokens`  
- `timestamp_to_tokens`  
- Grouping and filtering logic for stripping separators  

---

## Sense of Self

The model maintains an evolving internal self-description.

### Updating Sense of Self
1. Build a prompt with system + self prompt + thoughts  
2. Generate new self-description  
3. Clean and embed the result  
4. Store both tokens and embedding in `sense_of_self`  
5. Insert the self tokens into the active context window

### Loading Sense of Self
- Load all embeddings  
- Compute cosine similarity with new query  
- Return top matches as raw tokens  

---

## Sense of User

Same mechanism as sense of self, but driven by:

- The user query  
- The user’s linguistic patterns  
- Embeddings updated via generation

Stored in `sense_of_user` and included in prompts whenever thinking or responding.

---

## Objectives System

Objectives are maintained in the `objectives` table.

### Setting Objectives:
1. Load last N interactions from the `context_window`  
2. Build an objectives prompt  
3. Generate new objectives  
4. Split them on custom separators  
5. Clear the old objective rows  
6. Insert each objective token list as its own row  

### Using Objectives:
All objective rows are concatenated at runtime into a single token sequence that is attached to both thought prompts and response prompts.

---

## Action System

`determine_action()` attempts to infer an internal “action” (also via generation):

1. Extract relevant context  
2. Build action prompt  
3. Generate an action token sequence  
4. Clean the sequence  
5. Insert it into the `context_window` under the `action` field  

A separate `actions` SQL table exists for storing structured actions, but the prototype primarily stores them as raw tokens in the context window.

---

## Streaming and Cache Logic

The prototype customizes generation internals:

- `prepare_inputs_for_generation` rewrites:
  - `input_ids`
  - `position_ids`
  - `attention_mask`
  - `inputs_embeds`  
  depending on whether the model is streaming a single token.

- `forward_intercept` contains special handling when:
  - `context_position == 3` (responding)
  - `past_key_values` exist  
  - manually increments cache position and masks  

This ensures that thought tokens and response tokens stream correctly one at a time.

---

## Device Handling

Device selection:

- Prefer Apple Silicon (`mps`)
- Else prefer CUDA
- Else fallback to CPU

The base transformer model, cached embeddings, and pre-tokenized arrays are moved to the detected device.

---

## Conceptual Usage

Even though the internals are complex, usage looks standard:

```python
prompt = "Explain retrieval-augmented generation."
input = tokenizer(prompt, return_tensors="pt")
output = model.generate(**input, max_new_tokens=256)
Under the hood, this triggers:

Memory retrieval

Thought generation

Response generation

Memory update

Objective update

Sense-of-self update

Sense-of-user update

Context window update

But from the outside, it behaves just like LlamaForCausalLM.

Historical Context
This code is exactly what it was:

An experimental research prototype,

A refactor of LlamaForCausalLM,

Using the default dataset and tokenizer from the Llama 3.1-era base model,

With a large amount of logic hacked directly into the forward path,

Designed to explore integrated reflection, memory, and self-modeling.

It was never meant for release and never finished.
