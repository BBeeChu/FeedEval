from pathlib import Path
import torch
import torch.nn.functional as F
from torch.optim import AdamW
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from typing import Dict, List, Union
import json
import os
from tqdm import tqdm
from transformers import set_seed
import logging
from datasets import load_dataset, Dataset
import argparse
from peft import LoraConfig, get_peft_model, TaskType
from accelerate import Accelerator

logger = logging.getLogger(__name__)

accelerator = Accelerator()
device = accelerator.device


def rank_loss(score_chosen, score_rejected, margin=0.5):
    """
    Implements: L = -log(σ(score_chosen - score_rejected - margin))
    
    Args:
        score_chosen: Tensor of shape (batch,) - scores for preferred outputs
        score_rejected: Tensor of shape (batch,) - scores for less preferred outputs
        margin: float - optional margin to encourage separation
        
    Returns:
        loss: Tensor - scalar loss value
    """
    diff = score_chosen - score_rejected - margin
    loss = -torch.log(torch.sigmoid(diff) + 1e-8)  # small value added for numerical stability
    return loss.mean()


def disable_dropout_in_model(model):
    """Disables dropout in the model for inference."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0
    return model


class PreferenceDataLoader:
    def __init__(self, data_path: str, tokenizer: AutoTokenizer, args, test: bool):
        """
        Initialize preference data loader.
        Args:
            data_path: Path to the dataset file
            tokenizer: Tokenizer for processing text
        """
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.test = test
        self.args = args
        
        
        self.raw_data = self._load_raw_data()
        self.dataset = self._format_dataset()


    def _load_raw_data(self):
        """Load raw JSON data."""
        logger.info(f"Loading dataset from {self.data_path}")
        try:
            with open(self.data_path, 'r') as f:
                data = json.load(f)
                if self.test:
                    # Filter for test split if specified
                    return [item for item in data if item["split"] == "test"]
                    # return [item for item in data]
                else:
                    # Return all data if not test split
                    return [item for item in data if item["split"] != "test"]
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise

    def _format_conversation(self, prompt, feedback) -> list:
        """Format a conversation with dialog history and response."""
        conversation = []

        # Add system prompt
                        
        system_prompt = {"role": "system",
                            "content": "Judge the pedagogical helpfulness of the response provided by a teacher. Focus on the helpfulness of the scaffolding guidance, inclusion of revision points, and actionability of the feedback."}
        conversation.append(system_prompt)
        
        conversation.append({"role": "user",
                            "content": prompt + "\nFeedback: " + feedback})

        return conversation

    def _format_dataset(self):
        """Format the dataset into chosen/rejected pairs."""
        formatted_data = {
            'chosen': [],
            'rejected': []
        }

        for item in self.raw_data:
            # Format conversations for chosen (generated) and rejected (ground truth)
            chosen_conv = self._format_conversation(
                item["prompt"], item["chosen"]
            )
            rejected_conv = self._format_conversation(
                item["prompt"], item["rejected"]
            )
            

            formatted_data['chosen'].append(chosen_conv)
            formatted_data['rejected'].append(rejected_conv)

        # print one example
        print(formatted_data['chosen'][5])
        print(formatted_data['rejected'][5])

        return Dataset.from_dict(formatted_data)  # .shuffle(seed=42)

    def get_evaluation_pairs(self, batch_size: int = None):
        """Get evaluation pairs with optional batching."""
        if batch_size:
            return self.dataset.iter(batch_size=batch_size)
        return self.dataset


class RewardModel:
    def __init__(self, model_name: str):
        """Initialize reward model."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            num_labels=1,
            cache_dir="./"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir="./"
        )

        lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],  # for LLaMA
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        self.model = self.model.to(device)

    def get_scores(self, conversations: List[List[dict]], **kwargs) -> List[float]:
        """Get reward scores for a batch of conversations."""
        scores = []
        for conversation in conversations:
            inputs = self.tokenizer.apply_chat_template(
                conversation,
                tokenize=True,
                return_tensors="pt"
            ).to(self.device)
            outputs = self.model(inputs)
            score = outputs.logits[0][0]
            scores.append(score)
        return torch.stack(scores)


def evaluate_preference_accuracy(
        model,
        data_path: str,
        args,
        # batch_size: int = 64,
        # batch_size: int = 1,
        batch_size: int = 4,
        output_dir: str = "scaffolding_scores"
) -> Dict[str, Union[float, int]]:
    """Evaluate preference prediction accuracy."""
    data_loader = PreferenceDataLoader(data_path, model.tokenizer, args, test=True)

    total_correct = 0
    total_samples = 0
    all_scores = {'chosen': [], 'rejected': []}

    # Copy original data for enrichment


    model.model.eval()

    with torch.no_grad():

        for batch in tqdm(data_loader.get_evaluation_pairs(batch_size), desc="Evaluating", total=int(len(data_loader.dataset)/ batch_size)):
            chosen_scores = model.get_scores(batch['chosen'], batch_size=batch_size)
            rejected_scores = model.get_scores(batch['rejected'], batch_size=batch_size)

            print(f"Chosen scores: {chosen_scores}")
            print(f"Rejected scores: {rejected_scores}")

            # Calculate accuracy
            batch_correct = sum(1 for c, r in zip(chosen_scores, rejected_scores) if c > r)
            total_correct += batch_correct
            total_samples += len(chosen_scores)

            # Store scores
            all_scores['chosen'].extend(chosen_scores.detach().tolist())
            all_scores['rejected'].extend(rejected_scores.detach().tolist())


            logger.info(f"Current accuracy: {total_correct / total_samples:.4f}")

    # Calculate final metrics
    accuracy = total_correct / total_samples
    results = {
        'win_rate': accuracy,
        'score': float(np.mean(all_scores['chosen'])),
        'baseline_score': float(np.mean(all_scores['rejected'])),
        'mean_margin': float(np.mean(np.array(all_scores['chosen']) - np.array(all_scores['rejected']))),
        'total_samples': total_samples,
    }



    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    set_seed(42)

    parser = argparse.ArgumentParser(description='Run scaffolding score reward model on generations.')
    parser.add_argument('--data_path', type=str, help='Path to the data with generations.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for evaluation.')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for training.')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-3B-Instruct', help='Model name or path.')
    args = parser.parse_args()

    MODEL_NAME = args.model_name
    data_path = "../data/feedback_preference_dataset.json"
    save_path = "./llama_3b_helpfulness_reward_model/"

    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    
    model = RewardModel(MODEL_NAME)
    data_loader = PreferenceDataLoader(data_path, model.tokenizer, args, test=False)
    print(len(data_loader.dataset))
    
    optimizer = AdamW(model.model.parameters(), lr=1e-5)
    batch_size = args.batch_size
    loss_list = []

    model, optimizer, data_loader = accelerator.prepare(
        model,
        optimizer,
        data_loader
    )

    model.model.train()
    for epoch in range(args.epochs):
        for batch in tqdm(data_loader.get_evaluation_pairs(batch_size), desc="Evaluating", total=int(len(data_loader.dataset)/ batch_size)):
            chosen_scores = model.get_scores(batch['chosen'], batch_size=batch_size)
            rejected_scores = model.get_scores(batch['rejected'], batch_size=batch_size)
            
            loss = rank_loss(chosen_scores, rejected_scores, margin=0.5)
            
            optimizer.zero_grad()
            accelerator.backward(loss)
            
            loss_list.append(loss.item())
            optimizer.step()

        print(f"Loss: {np.mean(loss_list)}")

    model.model.save_pretrained(save_path)
    model.tokenizer.save_pretrained(save_path)


        
    results = evaluate_preference_accuracy(model, data_path, args, batch_size=args.batch_size)
    print(f"Final Results: {results}")
    