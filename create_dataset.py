import random
import pandas as pd

def main():
    n_train = 5000
    n_valid = 1000
    n_test = 1000

    n_total = n_train + n_valid + n_test

    prompts = []
    answers = []

    # Create Prompts and Answers
    for _ in range(n_total):
        num1 = random.randint(1000, 9999)
        num2 = random.randint(1000, 9999)
        # prompts.append(f"The sum of {num1} and {num2} is")
        prompts.append(f"Question: What is the sum of {num1} and {num2}?\n\nAnswer: ")
        answers.append(f"{num1+num2}")

    # Create Dataset Split    
    train_prompts = prompts[:n_train]
    train_answers = answers[:n_train]

    valid_prompts = prompts[n_train:n_train+n_valid]
    valid_answers = answers[n_train:n_train+n_valid]

    test_prompts = prompts[n_train+n_valid:]
    test_answers = answers[n_train+n_valid:]

    # Create Dataframes
    df_train = pd.DataFrame({"prompts": train_prompts, "answers": train_answers})
    df_valid = pd.DataFrame({"prompts": valid_prompts, "answers": valid_answers})
    df_test = pd.DataFrame({"prompts": test_prompts, "answers": test_answers})

    # Save Dataframes
    df_train.to_csv("data/train.csv", index=False)
    df_valid.to_csv("data/valid.csv", index=False)
    df_test.to_csv("data/test.csv", index=False)


if __name__ == "__main__":
    main()
