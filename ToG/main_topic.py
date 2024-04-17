from tqdm import tqdm
import json
import argparse
from utils import prepare_dataset, run_llm

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str,
                    default="webqsp", help="choose the dataset.")
parser.add_argument("--max_length", type=int,
                    default=256, help="the max length of LLMs output.")
parser.add_argument("--LLM_type", type=str,
                    default="llama", help="base LLM model.")
parser.add_argument("--openai_api_keys", type=str,
                    default="", help="if the LLM_type is gpt-3.5-turbo or gpt-4, you need add your own openai api keys.")
args = parser.parse_args()

datas, question_string = prepare_dataset(args.dataset)
print("Start Running ToG on %s dataset." % args.dataset)

topic_prompt = """Given a quesion, please extract any useful information about the topic of given question.
Given a example as below.
question: what character did natalie portman play in star wars
topic: Natalie Portman, Star Wars Episode I: The Phantom Menace
information: In "Star Wars Episode I: The Phantom Menace," Natalie Portman portrayed the character Queen Padmé Amidala of Naboo. She was a central figure in the prequel trilogy of the Star Wars saga and played a significant role in the political and military conflicts depicted in the films.
quesion: {}
topic: {}
information:
"""
question_prompt = """Given the background information of the topic, please the answer the question related to the topic.
Given a example as below.
question: what character did natalie portman play in star wars
topic: Natalie Portman, Star Wars Episode I: The Phantom Menace
information: In "Star Wars Episode I: The Phantom Menace," Natalie Portman portrayed the character Queen Padmé Amidala of Naboo. She was a central figure in the prequel trilogy of the Star Wars saga and played a significant role in the political and military conflicts depicted in the films.
quesion: {}
topic: {}
information: {}
answer: 
"""

for data in tqdm(datas):
    question = data[question_string]
    topic_entity = data['topic_entity']
    topic = str(list(topic_entity.values())).strip('[]').replace("'", "")
    prompt = run_llm(topic_prompt.format(question, topic), 0., args.max_length*2, args.openai_api_keys, args.LLM_type)
    print(question_prompt.format(question, topic, prompt))
    response = run_llm(question_prompt.format(question, topic, prompt), 0., args.max_length, args.openai_api_keys, args.LLM_type)
    dict = {"question": question, "result": response, 'prompt': prompt}
    with open("topic_example_{}_{}.jsonl".format(args.LLM_type, args.dataset), "a") as outfile:
        json_str = json.dumps(dict)
        outfile.write(json_str + "\n")
