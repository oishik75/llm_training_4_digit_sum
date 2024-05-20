import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, TaskType
from peft import get_peft_model
import torch

def tokenize(batch, tokenizer, max_length):
    training_input = [batch['prompts'][i] + str(batch['answers'][i]) for i in range(len(batch['prompts']))]
    tokenized_input = tokenizer(training_input, truncation=True, max_length=max_length, padding="max_length")
    return tokenized_input

def main():
    parser = argparse.ArgumentParser("Training Parser")
    parser.add_argument("--model", default="google/gemma-2b", help="Mddel name or checkpoint path for base model")
    parser.add_argument("--save_model_dir", default="models/", help="Directory for storing model checkpoints.")
    parser.add_argument("--log_dir", default="logs/", help="Directory for storing logs.")
    parser.add_argument("--data_path", default="data/", help="Directory containing train test and valid csvs.")
    # Tokeinizer Arguments
    parser.add_argument("--max_length", type=int, default=64, help="Max length for tokenization.")
    # Training Arguments
    parser.add_argument("--no_cuda", action="store_true", help="Flag for not using gpu.")
    parser.add_argument("--train_batch_size", type=int, default=2, help="Training batch size.")
    parser.add_argument("--eval_batch_size", type=int, default=2, help="Evaluation batch size.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Number of steps for gradient accumulatation.")
    parser.add_argument("--n_epochs", type=int, default=4, help="No of training epochs.")
    parser.add_argument("--logging_steps", type=int, default=50, help="Logging steps interval")
    parser.add_argument("--eval_steps", type=int, default=50, help="Evaluation steps interval")
    parser.add_argument("--save_steps", type=int, default=50, help="Model saving steps interval")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight Decay Parameter")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Steps for learning rate warmup.")
    parser.add_argument("--lr_scheduler_type", default="cosine", help="Type of lr scheduler")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning Rate")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    # Load Dataset
    dataset = load_dataset(path=args.data_path)

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir="cache")

    # Tokenize Data
    tokenized_dataset = dataset.map(
        lambda x: tokenize(x, tokenizer, args.max_length), 
        remove_columns=dataset['train'].column_names,
        batched=True,
        batch_size=512
    )

    # Load Model
    model = AutoModelForCausalLM.from_pretrained(args.model, cache_dir="cache")
    model.to(device)
    model_size = sum(t.numel() for t in model.parameters())
    print(f"Model size: {model_size/1000**2:.1f}M parameters")
    # Peft configuration
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=4, lora_alpha=32, lora_dropout=0.1)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=args.save_model_dir,
        logging_dir=args.log_dir,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.n_epochs,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        learning_rate=args.learning_rate,
        save_steps=args.save_steps,
        save_total_limit=10,
        fp16=True,
        report_to="tensorboard"
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"]
    )

    trainer.train()


if __name__ == "__main__":
    main()