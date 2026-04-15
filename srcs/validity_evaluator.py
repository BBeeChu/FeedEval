import os
import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, TrainerCallback, Trainer, DataCollatorForSeq2Seq, EarlyStoppingCallback
from trl import SFTTrainer
from datasets import Dataset
import pandas as pd
import pickle
import numpy as np
import json
import argparse
import random
import torch as th
from tqdm import tqdm
import re
from sklearn.metrics import accuracy_score


def set_seed(args):
    """
    Ensure reproducibility by setting the seed for random number generation.
    """
    np.random.seed(args.seed)
    random.seed(args.seed)
    if th.cuda.is_available():
        th.manual_seed(args.seed)
        th.cuda.manual_seed(args.seed)
        th.cuda.manual_seed_all(args.seed)  # if use multi-GPU
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False

def prepare_dataset(examples, tokenizer, args, test=False):
    
    
    system_prompt = "Judge whether the hypothesis is entailed by the premise. Answer with 'entailment' or 'contradiction'.\n\n" 

    result_dic = {}

    if not test:
        
        prompts = [
            tokenizer.apply_chat_template([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_text},
                {"role": "assistant", "content": output_text}
            ], tokenize=False, add_generation_prompt=False)
            for input_text, output_text in zip(examples["input_text"], examples["output_text"])
        ]

        result_dic["text"] = prompts


    
    else:
       
        prompts = [
            tokenizer.apply_chat_template([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_text},
            ], tokenize=False, add_generation_prompt=True)
            for input_text in examples["input_text"]
        ]

        tokenized = tokenizer(prompts, padding="max_length", max_length=512, truncation=True, return_tensors="pt")
        labels = examples["output_text"]

        result_dic["text"] = prompts
        result_dic["label_texts"] = labels
        result_dic["input_ids"] = tokenized["input_ids"]
        result_dic["attention_mask"] = tokenized["attention_mask"]
    
    

    return result_dic


def llama_train(args, model, train_dataset, dev_dataset, tokenizer):
    """
    Train the model using the provided training dataset and evaluation dataset.
    """
    
    eval_steps = args.eval_steps
    training_args = TrainingArguments(
        output_dir=args.checkpoint_dir,
        report_to="none",
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=eval_steps,
        eval_steps=eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        max_grad_norm=1.0,
        weight_decay=0.05,
        deepspeed="deepspeed_config.json",
    )

    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )


    trainer.train()

    best_path = trainer.state.best_model_checkpoint
    with open(f"{args.checkpoint_dir}/best_checkpoint_path.txt", "w") as f:
        f.write(best_path)
    
    return model



def llama_test(model, tokenizer, test_dataset, args):

    
    result_dic = {
        "pred": [],
        "true": [],
    }

    model.eval()
    batch_size = args.eval_batch_size
    expected_answer=["entailment", "contradiction"]
    expected_bos_token = tokenizer(
                expected_answer,
                padding=True,
                return_tensors="pt"
            ).input_ids[:, 1]
    with th.no_grad():
        for i in tqdm(range(0, len(test_dataset), args.eval_batch_size)):
            test = test_dataset[i:i+batch_size]
            
            
            outputs = model(
                input_ids=th.tensor(test["input_ids"]).to(model.device),
                attention_mask=th.tensor(test["attention_mask"]).to(model.device))

            output_prob = torch.softmax(outputs.logits[:, -1, expected_bos_token], dim=-1).flatten()
            output_list = []
            if output_prob[0].item() >= output_prob[1].item():
                output_list.append("entailment")
            else:
                output_list.append("contradiction")
            
            for i, (pred, true_result) in enumerate(zip(output_list, test["label_texts"])):
                
                try:
                    result_dic["pred"].append(pred)
                    result_dic["true"].append(true_result)
                    with open(f"{args.model_saving_dir}/tmp_results.json", "w") as f:
                        json.dump(result_dic, f, indent=4)
                    
                except Exception as e:
                    print(f"Error processing prediction: {e}")
                    continue
    
           
    log = "Test Result"
    accuracy = accuracy_score(result_dic["true"], result_dic["pred"])
    log += f"\n\n| Accuracy: {accuracy} |"
    print(log)

    return result_dic




def extract_step_num(name):
    match = re.search(r"checkpoint-(\d+)", name)
    return int(match.group(1)) if match else -1    

def main(args):
    set_seed(args)
    
    args.device = "cuda" if th.cuda.is_available() else "cpu"
    

    if not os.path.exists("results"):
        os.makedirs("results")
    model_saving_dir = os.path.join("results", "mnli")
    if not os.path.exists(model_saving_dir):
        os.makedirs(model_saving_dir)
    model_saving_dir = os.path.join(model_saving_dir, args.model_name.split("/")[-1])
    if not os.path.exists(model_saving_dir):
        os.makedirs(model_saving_dir)

    args.model_saving_dir = model_saving_dir
    

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                cache_dir="./"
            )
    model.config.use_cache = False

    
    df = pd.read_csv("../data/validity_dataset.csv")
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    train_df = df[:int(len(df) * 0.8)].reset_index(drop=True)
    dev_df = df[int(len(df) * 0.8):int(len(df) * 0.9)].reset_index(drop=True)
    test_df = df[int(len(df) * 0.9):].reset_index(drop=True)

    if args.debug:
        train_df = train_df.sample(10).reset_index(drop=True)
        dev_df = dev_df.sample(10).reset_index(drop=True)
        test_df = test_df[:10]

        args.epochs = 2
    train_dataset = Dataset.from_pandas(train_df)
    dev_dataset = Dataset.from_pandas(dev_df)
    test_dataset = Dataset.from_pandas(test_df)

    train_dataset = train_dataset.map(lambda x: prepare_dataset(x, tokenizer, args), batched=True)
    dev_dataset = dev_dataset.map(lambda x: prepare_dataset(x, tokenizer, args), batched=True)
    test_dataset = test_dataset.map(lambda x: prepare_dataset(x, tokenizer, args, test=True), batched=True)

    

    args.checkpoint_dir = f"./{args.model_name.split('/')[-1]}"
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, "mnli")
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    

    if not args.test:
        model = llama_train(args, model, train_dataset, dev_dataset, tokenizer)
        
        best_result = llama_test(model, tokenizer, test_dataset, args)

        
    else:
        checkpoint_dirs = [
            f for f in os.listdir(args.checkpoint_dir) 
            if f.startswith("checkpoint") and os.path.isdir(os.path.join(args.checkpoint_dir, f))
        ]
        best_checkpoint = max(checkpoint_dirs, key=extract_step_num)
        best_model_path = os.path.join(args.checkpoint_dir, best_checkpoint)
        
        best_model = AutoModelForCausalLM.from_pretrained(best_model_path, torch_dtype=th.bfloat16)
        best_model = best_model.to(args.device)
        
        best_result = llama_test(best_model, tokenizer, test_dataset, args)

    
    with open(f"{args.model_saving_dir}/best_result_dict.pkl", "wb") as f:
        pickle.dump(best_result, f)
    

            
    return best_result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune LLaMA model for essay scoring")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="Model name or path")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--eval_batch_size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--eval_steps", type=int, default=4000, help="Evaluation steps during training")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode with reduced dataset and epochs")
    parser.add_argument("--patience", type=int, default=2, help="Early stopping patience")
    parser.add_argument("--test", action="store_true", help="Run in test mode without training")
    args = parser.parse_args()
    
    
    
    print(args.model_name)
    result = main(args)
