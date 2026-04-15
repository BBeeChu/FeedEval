import os
import pandas as pd
import json
from tqdm import tqdm
from gpt_model import GPTModel
import numpy as np
import pickle
import argparse
def instruction_construct(prompt, essay, objective_analysis, scores):
    instruction = f"""
        [Prompt]
        {prompt}
        (end of [Prompt])

        [Essay]
        {essay}
        (end of [Essay])

        [Scores]
        {scores}
        (end of [Scores])

        [Rubric descriptions]
        {objective_analysis}
        (end of [Rubric descriptions])
    
        Refer to the provided [Prompt], [Scores], and [Rubric descriptions] to evaluate the given essay.
        Your task is to analyze the reason why the essay got certain scores for each trait based on the analysis of the essay.

        
        [Note]
        I have made an effort to remove personally identifying information from the essays using the Named Entity Recognizer (NER). The relevant entities are identified in the text and then replaced with a string
        such as '@PERSON', '@ORGANIZATION', '@LOCATION', '@DATE', '@TIME', '@MONEY', '@PERCENT', '@CAPS' (any capitalized word) and '@NUM' (any digits). Please do not penalize the essay because of the anonymizations.
        (end of [Note])
        
        Q. Identify specific excerpts from the [Essay] that illustrate the strengths or weaknesses highlighted in the [Rubric descriptions] for each trait. Quote or summarize the relevant parts of the essay.
        Based on this analysis, rationalize the [Rubric descriptions] for each trait. If the [Rubric descriptions] for a given trait indicates that the writing is strong, provide only positive feedback. 
        If it identifies weaknesses, provide a detailed analysis of the issue and suggest specific ways to improve it. Keep your response for each trait within three sentences, and do not include any specific scores in your analysis. 
        Provide your answer in the following format:
        {{"trait 1": "evaluation for trait 1",
        "trait 2": "evaluation for trait 2", ...}}
        """
        
    return instruction

def instruction_construct_with_excerpt(prompt, essay, objective_analysis, excerpt, scores):
    instruction = f"""
        [Prompt]
        {prompt}
        (end of [Prompt])
        
        [Excerpt]
        {excerpt}
        (end of [Excerpt])

        [Essay]
        {essay}
        (end of [Essay])

        [Scores]
        {scores}
        (end of [Scores])

        [Rubric descriptions]
        {objective_analysis}
        (end of [Rubric descriptions])
    
        Refer to the provided [Prompt], [Excerpt], [Scores], and [Rubric descriptions] to evaluate the given essay.
        Your task is to analyze the reason why the essay got certain scores for each trait based on the analysis of the essay.
        
        [Note]
        I have made an effort to remove personally identifying information from the essays using the Named Entity Recognizer (NER). The relevant entities are identified in the text and then replaced with a string
        such as '@PERSON', '@ORGANIZATION', '@LOCATION', '@DATE', '@TIME', '@MONEY', '@PERCENT', '@CAPS' (any capitalized word) and '@NUM' (any digits). Please do not penalize the essay because of the anonymizations.
        (end of [Note])
        
        Q. Identify specific excerpts from the [Essay] that illustrate the strengths or weaknesses highlighted in the [Rubric descriptions] for each trait. Quote or summarize the relevant parts of the essay.
        Based on this analysis, rationalize the [Rubric descriptions] for each trait. If the [Rubric descriptions] for a given trait indicates that the writing is strong, provide only positive feedback. 
        If it identifies weaknesses, provide a detailed analysis of the issue and suggest specific ways to improve it. Keep your response for each trait within three sentences, and do not include any specific scores in your analysis. 
        Provide your answer in the following format:
        {{"trait 1": "evaluation for trait 1",
        "trait 2": "evaluation for trait 2", ...}}
        """
        
    return instruction

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

def main(args):
    asap_df = pd.read_csv("../data/entire_data.csv")
    
    asap_df = asap_df[asap_df["essay_set"].isin([1,2,3,4,5,6])].reset_index(drop=True)


    llm_model = GPTModel()

    scoring_criteria_dict = {}

    system_message = """
    You are a member of the English essay writing test evaluation committee. Please, evaluate given essay using following information.
    """

    for idx, row in tqdm(asap_df.iterrows(), total=len(asap_df)):
        essay_id = row["essay_id"]
        
        prompt_id = row["essay_set"]
        

        essay = row["essay"]
        score_rubric_map = json.load(open(f"../data/rubric/prompt_{prompt_id}/score_rubric_map.json"))
        trait_list = trait_map[prompt_id]
        
        if prompt_id in [3, 4, 5, 6]:
            with open(f"../data/rubric/prompt_{prompt_id}/excerpt.txt", "r") as f:
                excerpt = f.read().strip()
        
        with open(f"../data/rubric/prompt_{prompt_id}/prompt.txt", "r") as f:
            prompt = f.read().strip()
        trait_rubric_evaluation = ""
        trait_scores = ""
        for trait in trait_list:
            trait_scores += f"{trait}: {row[trait.replace('_', ' ')]}\n"
            score_rubric_text = ""
            for score, rubric in score_rubric_map[trait.replace(" ", "_")].items():
                score_rubric_text += f"Score {score}: {rubric}\n"
            trait_rubric_evaluation += f"""
            [Trait]
            {trait}
            (end of [Trait])
            
            The following is a rubric description in terms of '{trait}' trait.
            {score_rubric_text}

            """ 
        if prompt_id in [3, 4, 5, 6]:
            essay_prompt = instruction_construct_with_excerpt(prompt, essay, trait_rubric_evaluation, excerpt, trait_scores)
        else:
            essay_prompt = instruction_construct(prompt, essay, trait_rubric_evaluation, trait_scores)

        
        prompt_list = [
        {
            "role": "system",
            "content": system_message
        },
        {
            "role": "user",
            "content": essay_prompt
        }
        ]
        
        patience = 0
        while patience < 3:
            try:
                rationale_result = llm_model.ask_chatgpt(prompt_list, n=8, model="gpt-5.1", temperature=0.7)
                
                rationale_dic = {}
                for trait in trait_list:
                    rationale_dic[trait] = []
                for i, rationale_list in enumerate(rationale_result):
                    rationale_list = json.loads(rationale_list.replace("```", "").replace("json", ""))
                    for trait, rationale in rationale_list.items():
                        rationale_dic[trait].append(rationale) 
                scoring_criteria_dict[essay_id] = rationale_dic
                
                with open(f"../data/rationale.json", "w") as f:
                    json.dump(scoring_criteria_dict, f, indent=4)
                
                break
            except Exception as e:
                patience += 1
                print(f"Error processing essay {essay_id}, attempt {patience}: {e}")

                if patience >= 3:
                    print(f"Failed after 3 retries. Saving empty rationale for essay {essay_id}.")
                    scoring_criteria_dict[essay_id] = "None"
                    with open(f"../data/rationale.json", "w") as f:
                        json.dump(scoring_criteria_dict, f, indent=4)
                
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    main(args)    
        