import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, TaskType
from peft import get_peft_model
import torch
import numpy as np

# Tokenize dataset for training
def tokenize(batch, tokenizer, max_length):
    training_input = [batch['prompts'][i] + str(batch['answers'][i]) for i in range(len(batch['prompts']))]
    tokenized_input = tokenizer(training_input, truncation=True, max_length=max_length, padding="max_length")
    return tokenized_input

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]

    return logits.argmax(dim=-1)

def extract_answer_from_prediction(prediction):
    # Assumes that the answer number is the first word after "Answer: "
    # If "Answer: " is not present, or the next word is not a number assign answer to be 0
    if "Answer: " not in prediction:
        return 0
    else:
        try:
            answer = int(prediction.split("Answer: ")[1].strip().split(" "))
        except:
            answer = 0

    return answer

# Function for computing metric for logging during evaluation
def compute_metrics(eval_preds, metric, tokenizer):
    pred_ids = eval_preds.predictions
    label_ids = eval_preds.label_ids
    # Hugginface uses -100 as the id for padding for some reason. Setting the correct padding id here that could be used to decode.
    pred_ids[pred_ids == -100] = tokenizer.pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    # Decode token_ids to text
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    pred_answers = list(map(extract_answer_from_prediction, pred_str))
    true_answers = list(map(lambda x: int(x.split(" ")[-1]), label_str))
    # Compute metric
    accuracy = np.mean(np.abs(np.array(pred_answers) == np.array(true_answers)))
    mean_absolute_error = np.mean(np.abs(np.array(pred_answers) - np.array(true_answers)))
    return {"accuracy": accuracy, "mae": mean_absolute_error}

def main():
    parser = argparse.ArgumentParser("Training Parser")
    parser.add_argument("--model", default="google/gemma-2b", help="Model name or checkpoint path for base model")
    parser.add_argument("--save_model_dir", default="models/", help="Directory for storing model checkpoints.")
    parser.add_argument("--log_dir", default="logs/", help="Directory for storing logs.")
    parser.add_argument("--data_path", default="data/", help="Directory containing train test and valid csvs.")
    # Tokeinizer Arguments
    parser.add_argument("--max_length", type=int, default=64, help="Max length for tokenization.")
    # Training Arguments
    parser.add_argument("--no_cuda", action="store_true", help="Flag for not using gpu.")
    parser.add_argument("--train_batch_size", type=int, default=2, help="Training batch size.")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="Evaluation batch size.")
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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
        compute_metrics=lambda x: compute_metrics(x, metric="", tokenizer=tokenizer),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"]
    )

    trainer.train()


if __name__ == "__main__":
    main()