Task 2 Medical Fine-Tuning with QLoRA Using Unsloth

A Complete, Step-by-Step README (Final Version)

This README explains the full process I followed to fine-tune a medical language model using QLoRA, Unsloth, Google Colab, and Hugging Face. The document includes:

A complete project overview

Tools used

Dataset details

Training workflow

Full Colab code

Saving & evaluation

Recommended hyperparameters

Troubleshooting

Ethics guidelines

1. Project Overview

In this task, the objective was to fine-tune a medical domain LLM using QLoRA through Unsloth’s optimized training pipeline in Google Colab.
The chosen base model was a modern LLM such as Llama-3 or DeepSeek-R1.

My goals:

Load and prepare a medical dataset

Tokenize it properly

Apply QLoRA to reduce GPU memory usage

Train using Unsloths optimized methods

Save the adapter

Test the model on medical questions

This helped me understand PEFT (Parameter Efficient Fine-Tuning), quantization, and low-VRAM training in practice.

 2. Tools & Libraries Used

Google Colab GPU

Unsloth (fast QLoRA implementation)

QLoRA (4-bit training)

Transformers

PEFT

Datasets

Bitsandbytes

Hugging Face Hub

Optional: Weights & Biases

3. Dataset Description

I used a medical-specific dataset, mainly composed of:

Clinical Question-Answer pairs

Short medical consultations

Condition-specific responses

Medical terminology

Dataset was formatted into:

{
  "prompt": "Medical question",
  "response": "Correct medical answer"
}


Data cleaning and formatting was done inside Colab.

 4. Training Workflow (What I Actually Did)
Step 1 Install Unsloth & Dependencies

In Colab, install all libraries:

pip install -q transformers accelerate datasets peft bitsandbytes safetensors sentencepiece evaluate huggingface_hub unsloth

Step 2 Load the Base Model

Used Llama-3 or DeepSeek-R1 in 4/8-bit quantized mode:

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

MODEL_NAME = "your-base-model-or-deepseek-r1"
OUTPUT_DIR = "/content/drive/MyDrive/deepseek-medical-qlora"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_8bit=True,   
    device_map="auto"
)

model = prepare_model_for_kbit_training(model)

Step 3 Configure QLoRA Adapter
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj","v_proj"],
    lora_dropout=0.05,
    bias="none"
)

model = get_peft_model(model, lora_config)

Step 4 Load & Prepare Medical Dataset
from datasets import load_dataset

dataset = load_dataset('json', data_files={
    'train': 'data/medical_qa_train.jsonl',
    'validation': 'data/medical_qa_eval.jsonl'
})

def preprocess(example):
    full = example['prompt'] + tokenizer.eos_token + example['response'] + tokenizer.eos_token
    return tokenizer(full, truncation=True, max_length=1024)

tokenized = dataset.map(
    preprocess, 
    remove_columns=dataset['train'].column_names
)

Step 5 Training
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=50,
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized['train'],
    eval_dataset=tokenized['validation']
)

trainer.train()

Step 6  Monitor GPU Usage
!nvidia-smi --query-gpu=memory.total,memory.used,utilization.gpu --format=csv

Step 7  Save the Adapter
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)


Upload to Hugging Face:

from huggingface_hub import login
login("YOUR_HF_TOKEN")

model.push_to_hub("username/deepseek-medical-qlora")
tokenizer.push_to_hub("username/deepseek-medical-qlora")

Step 8  Evaluation
from transformers import AutoModelForCausalLM
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_8bit=True,
    device_map="auto"
)

adapter = PeftModel.from_pretrained(base, OUTPUT_DIR)

prompt = "What are the symptoms of iron deficiency?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

output = adapter.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(output[0]))

5. Results (My Observations)

After fine-tuning:

Medical reasoning improved significantly

Responses became more consistent and accurate

Training fit easily into Colab VRAM

I learned deeply how QLoRA + PEFT works with real medical data

6. Files Produced
notebooks/deepseek_finetune_task2.ipynb
data/medical_qa_train.jsonl
data/medical_qa_eval.jsonl
models/deepseek-medical-qlora/
  adapter_model.safetensors
  adapter_config.json
  tokenizer.json
  special_tokens_map.json

7. Recommended Hyperparameters
Parameter	Recommended
learning_rate	1e-4 2e-4
epochs	2 -5
micro batch	1- 4
grad accumulation	4 - 16
LoRA rank	4 - 16
LoRA alpha	16 - 32
8. Safety & Ethics

This model is not a medical device.
Do NOT use it for real medical decisions.

Always ensure:

Human review by medical professionals

No PHI (patient data) in training

Transparency about model limitations

9. Troubleshooting
OOM Errors

Reduce batch size

Lower max sequence length

Use smaller base model

Tokenizer Errors

Ensure tokenizer = model checkpoint

Unsloth Errors

Reinstall:
pip install --upgrade unsloth