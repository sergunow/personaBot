import transformers
import datasets
import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, \
    DataCollatorForSeq2Seq
import torch

max_input_length = 128
max_target_length = 128
source_lang = "en"
target_lang = "en"
prefix = 'Rick: '


def preprocess_function(examples):
    inputs = [prefix + ex for ex in examples["context"]]
    targets = [ex for ex in examples["response"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, padding=True, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, padding=True, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--prefix", type=str, default='')

    # Data, model, and output directories
    parser.add_argument("--output_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--dataset", type=str, default='rick_subs.csv', required=True)

    input_args = parser.parse_args()
    global prefix
    prefix = input_args.prefix

    data = pd.read_csv('./datasets/{0}'.format(input_args.dataset))
    data = data.dropna()
    data = data.reset_index()

    df = {}
    n = 3
    for i in range(n, len(data) - 1):
        row = []
        prev = i - 1 - n  # we additionally substract 1, so row will contain current responce and 7 previous responces
        for j in range(i, prev, -1):
            row.append(data['content'][j])
        context_text = ' '.join([w for w in row])
        df[i] = {
            'response': data['content'][i + 1],
            'context': context_text
        }
    df = pd.DataFrame.from_dict(df, 'index')

    train, eval = train_test_split(df, test_size=0.3, random_state=42)
    train = train.reset_index()
    eval = eval.reset_index()

    train_dataset = Dataset.from_pandas(train[['response', 'context']])
    eval_dataset = Dataset.from_pandas(eval[['response', 'context']])

    tokenizer = AutoTokenizer.from_pretrained(input_args.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(input_args.model_name_or_path)

    args = Seq2SeqTrainingArguments(output_dir=input_args.output_dir,
                                    do_train=True,
                                    do_eval=True,
                                    evaluation_strategy="epoch",
                                    per_device_train_batch_size=input_args.per_device_train_batch_size,
                                    per_device_eval_batch_size=input_args.per_device_train_batch_size,
                                    learning_rate=5e-5,
                                    num_train_epochs=input_args.epochs,
                                    logging_dir="/logs")

    tokenized_train_dataset_datasets = train_dataset.map(preprocess_function, batched=True)
    tokenized_eval_dataset_datasets = eval_dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # defining trainer using 🤗
    trainer = Seq2SeqTrainer(model=model,
                             args=args,
                             data_collator=data_collator,
                             tokenizer=tokenizer,
                             train_dataset=tokenized_train_dataset_datasets,
                             eval_dataset=tokenized_eval_dataset_datasets)
    trainer.train()