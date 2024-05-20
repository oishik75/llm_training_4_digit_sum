import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.utils.data import TensorDataset, DataLoader
from peft import PeftModel
from tqdm import tqdm
import numpy as np
import pandas as pd

def evaluate(model, tokenizer, dataset, device, max_length=64):
    # Tokenize dataset
    tokenized_dataset = dataset.map(lambda x: tokenize(x, tokenizer, max_length), 
                                        remove_columns=['prompts'], 
                                        batched=True, 
                                        batch_size=512)
    
    # Create torch tensors
    input_ids = torch.tensor(tokenized_dataset["input_ids"]).squeeze().to(device)
    attention_masks = torch.tensor(tokenized_dataset["attention_mask"]).squeeze().to(device)

    # Create dataset and dataloader
    dataset = TensorDataset(input_ids, attention_masks)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    # Move model to device
    model.to(device)

    # Inference
    output_nums = []
    predictions = []
    for batch in tqdm(dataloader):
        input_ids, attention_mask= batch
        with torch.no_grad():
            output_ids = model.generate(
                input_ids, 
                attention_mask=attention_mask, 
                max_new_tokens=50, 
                pad_token_id=tokenizer.eos_token_id
            )
        output_ids = output_ids[:, input_ids.shape[1]:]
        output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        predictions += output
        output = [text.split("\n")[0] for text in output]
        try:
            output = list(map(int, output))
            output_nums += output
        except:
            output_nums += [0 for _ in range(len(output))]

    accuracy = np.mean(np.abs(np.array(output_nums) == np.array(tokenized_dataset['answers'])))
    mean_absolute_error = np.mean(np.abs(np.array(output_nums) - np.array(tokenized_dataset['answers'])))
    print(predictions[:5])
    return accuracy, mean_absolute_error, predictions

def tokenize(batch, tokenizer, max_length):
    tokenized_input = tokenizer(batch['prompts'], truncation=True, max_length=max_length, padding="max_length")
    answers = list(map(int, batch['answers']))
    return {"input_ids": tokenized_input["input_ids"], "attention_mask": tokenized_input["attention_mask"], "answers": answers}

def main():
    parser = argparse.ArgumentParser("Evaluation Parser")
    parser.add_argument("--model", default="google/gemma-2b", help="Model name or path")
    parser.add_argument("--lora_model", default=None, help="Lora model name/path")
    parser.add_argument("--split", choices=["train", "valid", "test", "all"], default="test", help="Split to evaluate on")
    parser.add_argument("--dataset_train", default="data/train.csv", help="Train Dataset")
    parser.add_argument("--dataset_valid", default="data/valid.csv", help="Valid Dataset")
    parser.add_argument("--dataset_test", default="data/test.csv", help="Test Dataset")
    parser.add_argument("--no_cuda", action="store_true", help="Flag for evaluating without gpu")
    parser.add_argument("--save_predictions", action="store_true", help="Flag for saving the original model output.")
    parser.add_argument("--save_prediction_file_train", default="output/predictions_train.csv", help="File for storing the model output for train.")
    parser.add_argument("--save_prediction_file_valid", default="output/predictions_valid.csv", help="File for storing the model output for valid.")
    parser.add_argument("--save_prediction_file_test", default="output/predictions_test.csv", help="File for storing the model output for test.")

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir="cache")
    model = AutoModelForCausalLM.from_pretrained(args.model, cache_dir="cache")
    if args.lora_model is not None:
        model = PeftModel.from_pretrained(model, args.lora_model, cache_dir="cache")

    DEVICE = "cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu"

    if args.split == "train" or args.split == "all":
        print("Evaluating on Train Dataset...")
        dataset = load_dataset('csv', data_files=args.dataset_train, split="train")
        print("No of instances: ", len(dataset))
        
        accuracy, mae, predictions = evaluate(model, tokenizer, dataset, DEVICE)
        if args.save_predictions:
            dataset = dataset.add_column("predictions", predictions)
            dataset.to_csv(args.save_prediction_file_train)

        print(f"Accuracy: {accuracy}\t Mean Absolute Error: {mae}")
        print("-------------------------------------------------------")

    if args.split == "valid" or args.split == "all":
        print("Evaluating on Validation Dataset...")
        dataset = load_dataset('csv', data_files=args.dataset_valid, split="train")
        print("No of instances: ", len(dataset))
        
        accuracy, mae, predictions = evaluate(model, tokenizer, dataset, DEVICE)
        if args.save_predictions:
            dataset = dataset.add_column("predictions", predictions)
            dataset.to_csv(args.save_prediction_file_valid)

        print(f"Accuracy: {accuracy}\t Mean Absolute Error: {mae}")
        print("-------------------------------------------------------")
    
    if args.split == "test" or args.split == "all":
        print("Evaluating on Test Dataset...")
        dataset = load_dataset('csv', data_files=args.dataset_test, split="train")
        print("No of instances: ", len(dataset))
        
        accuracy, mae, predictions = evaluate(model, tokenizer, dataset, DEVICE)
        if args.save_predictions:
            dataset = dataset.add_column("predictions", predictions)
            dataset.to_csv(args.save_prediction_file_test)

        print(f"Accuracy: {accuracy}\t Mean Absolute Error: {mae}")
        print("-------------------------------------------------------")

if __name__ == "__main__":
    main()