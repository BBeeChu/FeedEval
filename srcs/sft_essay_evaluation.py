import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, TrainerCallback, Trainer, EarlyStoppingCallback, AutoModelWithLMHead, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
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
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from trl import SFTTrainer, SFTConfig

trait_map = {
    1: ["sentence fluency", "word choice", "conventions", "organization", "content"],
    2: ["sentence fluency", "word choice", "conventions", "organization", "content"],
    3: ["narrativity", "language", "prompt adherence", "content"],
    4: ["narrativity", "language", "prompt adherence", "content"],
    5: ["narrativity", "language", "prompt adherence", "content"],
    6: ["narrativity", "language", "prompt adherence", "content"],
    7: ["style", "conventions", "organization", "content"],
    8: ["voice", "sentence fluency", "word choice", "conventions", "organization", "content"]
    }


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

    
    system_prompt = """
    You are an essay evaluator.
    You will receive an essay and you will need to evaluate the essay of prompt {prompt_id}, focusing on the following traits: {trait_list}.
    Output only the literal evaluation in JSON format, using the trait names as keys, without any additional text.
    """
    
    
    
    
    result_dic = {}
    if not test:
        prompts = []
        
        for essay, prompt_id, output_text in zip(examples["essay"], examples["essay_set"], examples["score_feedback_labels"]):
            if "qwen" in args.model_name.lower():
                prompts.append(
                    [
                    {"role": "user", "content": system_prompt.format(trait_list=trait_map[prompt_id], prompt_id=prompt_id) + "\n" + f"Essay: {essay}"},
                    {"role": "assistant", "content": output_text}
                    ]
                )
            else:
                prompts.append(
                [
                {"role": "system", "content": system_prompt.format(trait_list=trait_map[prompt_id], prompt_id=prompt_id)},
                {"role": "user", "content": f"Essay: {essay}"},
                {"role": "assistant", "content": output_text}
                ]
            )
            
        result_dic["messages"] = prompts

        
    
    else:
        prompts = []
        labels = []
        
        for essay, prompt_id, output_text in zip(examples["essay"], examples["essay_set"], examples["score_feedback_labels"]):
            
            if "qwen" in args.model_name.lower():
                prompt_full = tokenizer.apply_chat_template([
                    {"role": "user", "content": system_prompt.format(trait_list=trait_map[prompt_id], prompt_id=prompt_id) + "\n" + f"Essay: {essay}"}
                ], tokenize=False, add_generation_prompt=True)
            else:
                prompt_full = tokenizer.apply_chat_template([
                {"role": "system", "content": system_prompt.format(trait_list=trait_map[prompt_id], prompt_id=prompt_id)},
                {"role": "user", "content": f"Essay: {essay}"},
            ], tokenize=False, add_generation_prompt=True)
            prompts.append(prompt_full)
            labels.append(output_text)
        
        tokenized = tokenizer(prompts, truncation=True, padding='max_length', max_length=1024)
        result_dic["messages"] = prompts
        result_dic["label_texts"] = labels
        result_dic["input_ids"] = tokenized["input_ids"]
        result_dic["attention_mask"] = tokenized["attention_mask"]

    return result_dic


def llama_train(args, model, train_dataset, dev_dataset, tokenizer):
    
    
    eval_steps = args.eval_steps
    if args.debug:
        eval_steps = 2
    print("Evaluation steps:", eval_steps)
    """
    Train the model using the provided training dataset and evaluation dataset.
    """
    
   
    training_args = TrainingArguments(
        output_dir=args.checkpoint_dir,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=eval_steps,
        eval_steps=eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=False,
        bf16=True,
        max_grad_norm=1.0,
        weight_decay=0.05,
        deepspeed="deepspeed_config.json",
        report_to="none",
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
    
    pred_dic = dict()
    true_dic = dict()
    model.eval()
    batch_size = args.test_batch_size
    with th.no_grad():
        for i in tqdm(range(0, len(test_dataset), args.test_batch_size)):
            test = test_dataset[i:i+batch_size]
            
            essay_ids = test["essay_id"]
            
            if args.with_score:
                outputs = model.generate(
                input_ids=th.tensor(test["input_ids"]).to(model.device),
                attention_mask=th.tensor(test["attention_mask"]).to(model.device),
                max_new_tokens=712,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
                num_beams=1)
            else:
                outputs = model.generate(
                    input_ids=th.tensor(test["input_ids"]).to(model.device),
                    attention_mask=th.tensor(test["attention_mask"]).to(model.device),
                    max_new_tokens=512,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                    num_beams=1)

            generated_texts = tokenizer.batch_decode(outputs[:, th.tensor(test["input_ids"]).shape[1]:], skip_special_tokens=True)
            for i, (pred, true_result) in enumerate(zip(generated_texts, test["label_texts"])):
                pred = pred.replace("assistant", "").strip()
                essay_id = essay_ids[i]
                pred_dic[essay_id] = pred
                true_dic[essay_id] = true_result

                with open(f"{args.fold_model_saving_dir}/tmp_rationale_predictions.json", "w") as f:
                    json.dump(pred_dic, f, indent=4)
                with open(f"{args.fold_model_saving_dir}/tmp_rationale_true_labels.json", "w") as f:
                    json.dump(true_dic, f, indent=4)
                

    return pred_dic, true_dic
                


def main(args):
    
    set_seed(args)

    if not os.path.exists("results"):
        os.makedirs("results")
    model_saving_dir = os.path.join("results", args.model_name.split("/")[-1])
    if not os.path.exists(model_saving_dir):
        os.makedirs(model_saving_dir)
    for fold in range(5):
        print(f"Processing fold {fold}...")
        fold_model_saving_dir = os.path.join(model_saving_dir, f"{fold}")
        if not os.path.exists(fold_model_saving_dir):
            os.makedirs(fold_model_saving_dir)
    
        args.fold_model_saving_dir = fold_model_saving_dir

        args.checkpoint_dir = f"./{args.model_name.split('/')[-1]}"
        if not os.path.exists(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            
        args.checkpoint_dir = os.path.join(args.checkpoint_dir, f"fold_{fold}")
        if not os.path.exists(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir, exist_ok=True)
        


        if not args.test:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left", use_fast=False)
            tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(args.model_name, 
                                                    cache_dir=".", 
                                                    torch_dtype=th.bfloat16, 
                                                    pad_token_id=tokenizer.pad_token_id,
                                                    )
            lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],  # for LLaMA
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            )

            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left", use_fast=False)
            tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(args.model_name, 
                                                    cache_dir=".", 
                                                    torch_dtype=th.bfloat16, 
                                                    pad_token_id=tokenizer.pad_token_id,
                                                    device_map="auto",
                                                    )
            
            with open(f"{args.checkpoint_dir}/best_checkpoint_path.txt", "r") as f:
                best_model_checkpoint = f.read().strip()
            print("Best model checkpoint:", best_model_checkpoint)
            tokenizer = AutoTokenizer.from_pretrained(best_model_checkpoint, padding_side="left", use_fast=False)
            tokenizer.pad_token = tokenizer.eos_token
            model = PeftModel.from_pretrained(model, best_model_checkpoint, is_trainable=False)
            model.print_trainable_parameters()
        

        train_df = pd.read_csv(f"../data/fold_{fold}/train.csv")
        dev_df = pd.read_csv(f"../data/fold_{fold}/dev.csv")
        test_df = pd.read_csv(f"../data/fold_{fold}/test.csv")
        

        if args.debug:
            train_df = train_df.sample(10).reset_index(drop=True)
            dev_df = dev_df.sample(10).reset_index(drop=True)
            test_df = test_df[:10]

            args.epochs = 1
        train_dataset = Dataset.from_pandas(train_df)
        dev_dataset = Dataset.from_pandas(dev_df)
        test_dataset = Dataset.from_pandas(test_df)

        
        train_dataset = train_dataset.map(lambda x: prepare_dataset(x, tokenizer, args), batched=True)
        dev_dataset = dev_dataset.map(lambda x: prepare_dataset(x, tokenizer, args), batched=True)
        test_dataset = test_dataset.map(lambda x: prepare_dataset(x, tokenizer, args, test=True), batched=True)
        

        if not args.test:
            model = llama_train(args, model, train_dataset, dev_dataset, tokenizer)

            
            pred_dic, true_dic = llama_test(model, tokenizer, test_dataset, args)
        
        else:
            
            pred_dic, true_dic = llama_test(model, tokenizer, test_dataset, args)

      
        with open(f"{args.fold_model_saving_dir}/rationale_predictions.pkl", "wb") as f:
            pickle.dump(pred_dic, f)
        with open(f"{args.fold_model_saving_dir}/rationale_true_labels.pkl", "wb") as f:
            pickle.dump(true_dic, f)


            
    return pred_dic, true_dic

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune LLaMA model for essay scoring")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B", help="Model name or path")
    # parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model name or path")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size for training (8B)")
    parser.add_argument("--eval_batch_size", type=int, default=4, help="Batch size for evaluation (8B)")
    parser.add_argument("--test_batch_size", type=int, default=16, help="Batch size for testing (8B)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--eval_steps", type=int, default=100, help="Evaluation steps during training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for training")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode with reduced dataset and epochs")
    parser.add_argument("--patience", type=int, default=2, help="Early stopping patience")
    parser.add_argument("--test", action="store_true", help="Run in test mode without training")
    args = parser.parse_args()
    
   
   
    print(f"Running for model {args.model_name}")
    pred_result, true_result = main(args)

