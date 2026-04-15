import re

from openai import OpenAI


key = 'your_openai_api_key_here'

client = OpenAI(api_key = key)

class GPTModel():
    
    def ask_chatgpt(self, prompt, model="gpt-5.1", temperature=0.0, n=1):
        response = client.chat.completions.create(
                    model=model,
                    messages=prompt,
                    n=n
                )
        # response = client.chat.completions.create(
        #             model=model,
        #             temperature=temperature,
        #             messages=prompt,
        #             n=n
        #         )
        # return response.choices[0].message.content
        return [choice.message.content for choice in response.choices]
    
    def post_process(self, answer):
        answer = answer.replace('\n', ' ').replace('sql','').replace('```','')
        answer = re.sub('[ ]+', ' ', answer)
        answer = answer.strip()
        return answer