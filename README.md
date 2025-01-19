# Automated Supervised Data Creation and QLoRA Fine-Tuning of LLaMA 2

This repository demonstrates an automated pipeline for creating supervised datasets using GCP BigQuery warehouse and fine-tuning Meta's LLaMA 2 model using the QLoRA technique. The pipeline includes fetching and preprocessing data, preparing it for LLaMA 2's specific input format, and performing efficient fine-tuning using quantized adapters.

## Key Features
- **Automated Data Pipeline**: Leverages GCP BigQuery to query, join, and preprocess data with SQL for supervised learning tasks.
- **Instruction-Based Dataset Creation**: Enhances model generalization by adding task-specific instructions to training samples.
- **Efficient Fine-Tuning**: Uses QLoRA for low-rank adaptation on the LLaMA 2 model, enabling cost-effective training on consumer-grade GPUs.

## Pipeline Overview

### 1. Data Retrieval and Preprocessing
- The pipeline queries data from the publicly available Stack Overflow dataset on GCP BigQuery.
- Filters and joins:
  - **Questions**: Extracts Python-specific questions from `posts_questions`.
  - **Answers**: Matches questions with their accepted answers from `posts_answers`.
- Applies constraints:
  - Includes only questions with accepted answers.
  - Filters by tags (`python`) and date range (`>= 2020-01-01`).
- Prepares a dataset with `input_text` (questions) and `output_text` (answers).

#### Code Snippet
```sql
SELECT
    CONCAT(q.title, q.body) AS input_text,
    a.body AS output_text
FROM
    `bigquery-public-data.stackoverflow.posts_questions` q
JOIN
    `bigquery-public-data.stackoverflow.posts_answers` a
ON
    q.accepted_answer_id = a.id
WHERE
    q.accepted_answer_id IS NOT NULL
    AND REGEXP_CONTAINS(q.tags, "python")
    AND a.creation_date >= "2020-01-01"
LIMIT
    10000;
```

### 2. Instruction-Based Dataset Creation
- Enhances input data with instructions to guide the model on task expectations.
- Example transformation:
  - **Before**:
    ```text
    How to efficiently insert key-value pairs into a dictionary in Python?
    ```
  - **After**:
    ```text
    <s>[INST] Please answer the following Stackoverflow question on Python. Answer it like you are a developer answering Stackoverflow questions.

    Stackoverflow question:
    How to efficiently insert key-value pairs into a dictionary in Python? [/INST] Use the `dict.update()` method for efficiency. </s>
    ```

### 3. Data Transformation for LLaMA 2
- Converts the dataset to LLaMA 2-compatible supervised learning format.
- Saves the output as `transformed_data.json`.

#### Transformation Function
```python
def transform_to_llama_format(example):
    input_text = example['input_text_instruct']
    output_text = example['output_text']
    return {'text': f"<s>[INST] {input_text.strip()} [/INST] {output_text.strip()} </s>"}
```

### 4. Fine-Tuning with QLoRA
- **QLoRA (Quantized Low-Rank Adapters)**: Efficiently fine-tunes LLaMA 2 by introducing low-rank matrices to adapter layers, reducing memory requirements.
- Uses the [Hugging Face Transformers](https://huggingface.co/docs/transformers/) library for implementation.
- Fine-tuned model output is saved in `qlora_llama2`.

#### Tools Used
- **Libraries**: `transformers`, `peft`, `datasets`, `torch`, and `bitsandbytes`.
- **Configuration**:
  - `r = 64`: Low-rank matrix size.
  - `lora_alpha = 32`: Scaling factor.
  - Dropout: `0.1`.

#### Fine-Tuning Example
```python
from peft import LoraConfig, get_peft_model

# Define QLoRA Configuration
lora_config = LoraConfig(
    r=64,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    task_type="CAUSAL_LM"
)

# Apply QLoRA to the model
model = get_peft_model(model, lora_config)

# Train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    tokenizer=tokenizer,
)

trainer.train()
```

## Directory Structure
```
rajdip-i-qlora-finetuned/
├── Preprocess.ipynb                 # Notebook for data retrieval and preprocessing
├── QLoRA Fine-Tuning of Llma-2-7b.ipynb  # Notebook for fine-tuning using QLoRA
├── qlora-finetuning-cce64209d0bb.json   # GCP credentials for BigQuery access
├── transformed_data.json            # Preprocessed data for fine-tuning
├── transformed_data1.json           # Transformed data in LLaMA 2 format
└── tune_data_stack_overflow_python_qa.jsonl  # Raw data before transformation
```


