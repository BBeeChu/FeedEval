from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch as th
import os
import argparse
import pandas as pd
from tqdm import tqdm

def main(args):

    filt_data = pd.read_csv(f"../data/filtered_low_score_data/fold_{args.fold}_low_score_data.csv")
    data = pd.read_csv(f"../data/fold_{args.fold}/test.csv")
    
    feedback_data = {}
    for idx, row in data.iterrows():
        essay_id = row["essay_id"]
        feedback = row[f"{args.quality}_quality_score_feedback_labels"]
        tmp_dic = {}
        for trait, score_feedback in eval(feedback).items():
            if trait != 'overall':
                tmp_dic[trait] = score_feedback['rationale']
        feedback_data[essay_id] = tmp_dic

    if args.llm == "qwen":
        
        qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
        qwen_tokenizer.pad_token = qwen_tokenizer.eos_token
        qwen_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", device_map="auto", trust_remote_code=True,
                                                          pad_token_id=qwen_tokenizer.eos_token_id)
    elif args.llm == "llama":
        qwen_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
        qwen_tokenizer.pad_token = qwen_tokenizer.eos_token
        qwen_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", device_map="auto", trust_remote_code=True,
                                                          pad_token_id=qwen_tokenizer.eos_token_id)

    qwen_system_prompt = "You are a middle school student. You have received feedback on your essay. Please read the feedback carefully and revise your essay based on the feedback."
    qwen_excerpt_user_prompt = """
    Prompt:
    {prompt}
    Excerpt:
    {excerpt}
    Previous Essay:
    {essay}
    Feedback:
    {feedback}
    Strictly follow the feedback to revise the essay.
    Do not revise any part that is not present in the feedback. 
    Ensure that the revised essay reflects only the points mentioned in the feedback.
    Only output the revised essay without any additional commentary or explanation.
    Revised Essay:
    """
    
    qwen_no_excerpt_user_prompt = """
    Prompt:
    {prompt}
    Previous Essay:
    {essay}
    Feedback:
    {feedback}
    Strictly follow the feedback to revise the essay.
    Do not revise any part that is not present in the feedback. 
    Ensure that the revised essay reflects only the points mentioned in the feedback.
    Only output the revised essay without any additional commentary or explanation.
    Revised Essay:
    """
    revised_essay_dic = {}
    with th.no_grad():
        for i in tqdm(range(0, len(filt_data), args.batch_size)):
            batch = filt_data[i:i+args.batch_size]
            texts = []
            essay_ids = []
            for idx, row in batch.iterrows():
                essay = row["essay"]
                essay_id = row["essay_id"]
                if essay_id not in feedback_data:
                    continue
                essay_ids.append(essay_id)
                prompt_id = row["essay_set"]
                feedback = feedback_data[essay_id]
                with open(f"../data/rubric/prompt_{prompt_id}/prompt.txt", "r") as f:
                    prompt = f.read()
                
                if prompt_id in [3, 4, 5, 6]:

                    with open(f"../data/rubric/prompt_{prompt_id}/excerpt.txt", "r") as f:
                        excerpt = f.read()
                    qwen_user_prompt = qwen_excerpt_user_prompt.format(prompt=prompt, excerpt=excerpt, essay=essay, feedback=feedback)
                else:
                    qwen_user_prompt = qwen_no_excerpt_user_prompt.format(prompt=prompt, essay=essay, feedback=feedback)
                

                text_input = qwen_system_prompt + qwen_user_prompt

                messages = [
                    {"role": "user", "content": text_input}
                ]

                
                text = qwen_tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
                )
                
                texts.append(text)
            
            model_inputs = qwen_tokenizer(texts, return_tensors="pt", padding=True).to(qwen_model.device)

            # conduct text completion
            generated_ids = qwen_model.generate(
                **model_inputs,
                max_new_tokens=1000,
                temperature=0.7,
            )
            
            for j, idx in enumerate(range(i, min(i+args.batch_size, len(filt_data)))):
                try:
                    output_ids = generated_ids[j][len(model_inputs.input_ids[j]):].tolist() 


                    # parsing thinking content
                    try:
                        # rindex finding 151668 (</think>)
                        index = len(output_ids) - output_ids[::-1].index(151668)
                    except ValueError:
                        index = 0

                    thinking_content = qwen_tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
                    content = qwen_tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

                    revised_essay_dic[essay_ids[j]] = content
                except Exception as e:
                    print(e)
                    continue
        
            pd.to_pickle(revised_essay_dic, f"./revised_results/fold_{args.fold}_{args.llm}_{args.quality}_revised_essays.pkl")
    

    return revised_essay_dic

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing essays.")
    parser.add_argument("--llm", type=str, default="qwen", choices=["qwen", "llama"], help="LLM model to use for revision.")
    parser.add_argument("--quality", type=str, default="high", help="Quality of feedback to use for revision.")
    args = parser.parse_args()

    for fold in range(5):
        args.fold = fold
        print(f"Processing fold {fold} with feedback of {args.quality}.")
        revised_essays = main(args)