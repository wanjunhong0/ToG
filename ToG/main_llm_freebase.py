from tqdm import tqdm
import argparse
from utils import *
from freebase_func import *
import random
from client import *


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str,
                    default="webqsp", help="choose the dataset.")
parser.add_argument("--max_length", type=int,
                    default=256, help="the max length of LLMs output.")
parser.add_argument("--LLM_type", type=str,
                    default="llama", help="base LLM model.")
parser.add_argument("--opeani_api_keys", type=str,
                    default="", help="if the LLM_type is gpt-3.5-turbo or gpt-4, you need add your own openai api keys.")
args = parser.parse_args()

datas, question_string = prepare_dataset(args.dataset)
print("Start Running ToG on %s dataset." % args.dataset)

for data in tqdm(datas):
    question = data[question_string]
    response = run_llm(question, 0., args.max_length, args.opeani_api_keys, args.LLM_type)
    dict = {"question":question, "results": response}
    with open("{}_{}.jsonl".format(args.LLM_type, args.dataset), "a") as outfile:
        json_str = json.dumps(dict)
        outfile.write(json_str + "\n")