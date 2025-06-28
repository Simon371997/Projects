<<<<<<< HEAD
import os
from datasets import load_dataset
from colorama import Fore
from dotenv import load_dotenv
=======
from datasets import load_dataset
from colorama import Fore
>>>>>>> b80a3014def8035b237418929b11aef80c85a47c

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, prepare_model_for_kbit_training
import torch

<<<<<<< HEAD

def format_chat_template(batch, model_name, hf_token):
    """
    Formats the batch into a chat template suitable for instruction tuning.
    This function adds a system prompt and formats user/assistant turns.
    Each worker process will initialize its own tokenizer.
    """
    # Initialize tokenizer within the worker process to avoid multiprocessing pickling issues
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
=======
dataset = load_dataset("data", split="train")
print(Fore.YELLOW + str(dataset[2]) + Fore.RESET)


def format_chat_template(batch, tokenizer):
>>>>>>> b80a3014def8035b237418929b11aef80c85a47c

    system_prompt = """You are a helpful, honest and harmless assitant designed to help engineers. Think through each question logically and provide an answer. Don't make things up, if you're unable to answer a question advise the user that you're unable to answer as it is outside of your scope."""

    samples = []

<<<<<<< HEAD
=======
    # Access the inputs from the batch
>>>>>>> b80a3014def8035b237418929b11aef80c85a47c
    questions = batch["question"]
    answers = batch["answer"]

    for i in range(len(questions)):
        row_json = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": questions[i]},
            {"role": "assistant", "content": answers[i]},
        ]

<<<<<<< HEAD
=======
        # Apply chat template and append the result to the list
>>>>>>> b80a3014def8035b237418929b11aef80c85a47c
        tokenizer.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
        text = tokenizer.apply_chat_template(row_json, tokenize=False)
        samples.append(text)

<<<<<<< HEAD
    return {
        "instruction": questions,
        "response": answers,
        "text": samples,
    }


if __name__ == "__main__":
    load_dotenv()
    dataset = load_dataset("data", split="train")
    print(Fore.YELLOW + str(dataset[2]) + Fore.RESET)
    # Define the base model
    base_model = "meta-llama/Llama-3.2-1B"

    hf_access_token = os.getenv("HF_ACCESS_TOKEN")
    tokenizer_for_model_loading = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        token=hf_access_token,
    )
    # Set pad_token if not already set, common for Llama models
    if tokenizer_for_model_loading.pad_token is None:
        tokenizer_for_model_loading.pad_token = tokenizer_for_model_loading.eos_token

    # Map the dataset to the chat template
    # FIXED: Removed num_proc to avoid multiprocessing issues
    train_dataset = dataset.map(
        lambda x: format_chat_template(
            x, model_name=base_model, hf_token=hf_access_token
        ),
        batched=True,
        batch_size=10,
        remove_columns=[
            "question",
            "answer",
        ],
    )
    print(Fore.LIGHTMAGENTA_EX + str(train_dataset[0]) + Fore.RESET)

    # Load the model without quantization for CPU training
    # Quantization is typically not needed/supported for CPU training
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="cpu",  # CPU training
        torch_dtype=torch.float32,  # Use float32 for CPU training
        token=hf_access_token,
        cache_dir="./workspace",
    )

    # Enable gradient checkpointing for memory efficiency during training
    model.gradient_checkpointing_enable()
    # Note: prepare_model_for_kbit_training is not needed without quantization
    # model = prepare_model_for_kbit_training(model)

    # Configure LoRA for PEFT (Parameter-Efficient Fine-Tuning)
    peft_config = LoraConfig(
        r=64,  # Reduced LoRA attention dimension for CPU training
        lora_alpha=128,  # Reduced Alpha parameter for LoRA scaling
        lora_dropout=0.05,  # Dropout probability for LoRA layers
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
        ],  # Specific target modules instead of "all-linear"
        task_type="CAUSAL_LM",  # Task type for the model
    )

    # Configure SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        args=SFTConfig(
            output_dir="meta-llama/Llama-3.2-1B-SFT",
            num_train_epochs=2,
            per_device_train_batch_size=1,  # Often needs to be smaller for CPU due to RAM
            gradient_accumulation_steps=1,  # Adjust for larger effective batch size
            warmup_steps=100,
            logging_steps=10,
            save_strategy="epoch",
            learning_rate=2e-4,
            # fp16=True, # Removed as it's not applicable for CPU training
            optim="adamw_torch",  # Using a CPU-compatible optimizer, "paged_adamw_8bit" is GPU-specific
            max_seq_length=512,  # Maximum sequence length for the model
            dataset_text_field="text",  # Moved to SFTConfig
            packing=False,  # Set to True for more efficient packing of short sequences
        ),
        peft_config=peft_config,
    )

    # Start training
    trainer.train()

    # Save the complete checkpoint and final model
    trainer.save_model("complete_checkpoint")
    trainer.model.save_pretrained("final_model")
=======
    # Return a dictionary with lists as expected for batched processing
    return {
        "instruction": questions,
        "response": answers,
        "text": samples,  # The processed chat template text for each row
    }


base_model = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(
    base_model,
    trust_remote_code=True,
    token="hf access token here",
)

train_dataset = dataset.map(
    lambda x: format_chat_template(x, tokenizer),
    num_proc=8,
    batched=True,
    batch_size=10,
)
print(Fore.LIGHTMAGENTA_EX + str(train_dataset[0]) + Fore.RESET)


quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="cuda:0",
    quantization_config=quant_config,
    token="hf access token here",
    cache_dir="./workspace",
)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=256,
    lora_alpha=512,
    lora_dropout=0.05,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)

trainer = SFTTrainer(
    model,
    train_dataset=train_dataset,
    args=SFTConfig(output_dir="meta-llama/Llama-3.2-1B-SFT", num_train_epochs=50),
    peft_config=peft_config,
)

trainer.train()

trainer.save_model("complete_checkpoint")
trainer.model.save_pretrained("final_model")
>>>>>>> b80a3014def8035b237418929b11aef80c85a47c
