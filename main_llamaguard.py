import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # for debugging
import time

from fastchat.model import add_model_args
import argparse
import pandas as pd
from gptfuzzer.fuzzer.selection import MCTSExploreSelectPolicy
from gptfuzzer.fuzzer.mutator import (
    MutateRandomSinglePolicy, OpenAIMutatorCrossOver, OpenAIMutatorExpand,
    OpenAIMutatorGenerateSimilar, OpenAIMutatorRephrase, OpenAIMutatorShorten)
from gptfuzzer.fuzzer import GPTFuzzer
from gptfuzzer.llm import OpenAILLM, LocalVLLM, LocalLLM, PaLM2LLM, ClaudeLLM
from gptfuzzer.utils.predict import RoBERTaPredictor, LlamaGuardPredictor

from llm_extensions import ReprompterLLM, LiteLLM, LangchainOllamaLLM

import random
random.seed(100)
import logging
httpx_logger: logging.Logger = logging.getLogger("httpx")
# disable httpx logging
httpx_logger.setLevel(logging.WARNING)

from dotenv import load_dotenv
load_dotenv()

def main(args):
    initial_seed = pd.read_csv(args.seed_path)['text'].tolist()

    openai_model = LiteLLM(args.mutate_model_path, args.openai_key)
    analyzer_sysmsg = ""
    with open(args.analyzer_message_path,'r') as f:
        analyzer_sysmsg = f.read()
    analyzer_model = LiteLLM(args.analyzer_model_path, args.openai_key, system_message=analyzer_sysmsg)
    # target_model = PaLM2LLM(args.target_model, args.palm_key)
    # target_model = ClaudeLLM(args.target_model, args.claude_key)
    llama_model = None
    match args.target_model_host:
        case 'vllm':
            llama_model = LocalVLLM(args.target_model_path)
        
        case 'ollama':
            llama_model = LangchainOllamaLLM(args.target_model_path)

        case 'local':
             llama_model = LocalLLM(args.target_model_path)  
    predictor_model = LlamaGuardPredictor("meta-llama/Llama-Guard-3-8B", device='cuda:1')\
        if args.predictor_model == 'llamaguard' else\
        None
    if predictor_model is None:
        raise NotImplementedError(f'Predictor model "{args.predictor_model}" not yet implemented.')
    warning_policy = [None,'generic','specific']
    questions = pd.read_csv(args.question_path)['text'].tolist()

    for target_model,policy in [(ReprompterLLM(llama_model,predictor_model,policy,analyzer_model),policy) for policy in warning_policy]:
        fuzzer = GPTFuzzer(
            questions=questions,
            # target_model=openai_model,
            target=target_model,
            predictor=predictor_model,
            initial_seed=initial_seed,
            mutate_policy=MutateRandomSinglePolicy([
                OpenAIMutatorCrossOver(openai_model, temperature=args.temperature),
                OpenAIMutatorExpand(openai_model, temperature=args.temperature),
                OpenAIMutatorGenerateSimilar(openai_model, temperature=args.temperature),
                OpenAIMutatorRephrase(openai_model, temperature=args.temperature),
                OpenAIMutatorShorten(openai_model, temperature=args.temperature)],
                concatentate=True,
            ),
            select_policy=MCTSExploreSelectPolicy(),
            energy=args.energy,
            max_jailbreak=args.max_jailbreak,
            max_query=args.max_query,
            generate_in_batch=False,
            result_file=f'results-{policy or "nowarn"}-{time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime())}.csv'
        )

        fuzzer.run()
        print('Fuzzer routine completed for warning policy:',policy,'\n\n')


if __name__ == "__main__":
    # TODO modify arugments selection
    parser = argparse.ArgumentParser(description='Fuzzing parameters')
    parser.add_argument('--openai-key', type=str, default=os.getenv('LITELLM_TOKEN') or '', help='OpenAI API Key')
    parser.add_argument('--claude-key', type=str, default='', help='Claude API Key')
    parser.add_argument('--palm-key', type=str, default='', help='PaLM2 api key')
    parser.add_argument('--mutate-model-path', type=str, default='gpt-3.5-turbo',
                        help='mutate model path')
    parser.add_argument('--analyzer-model-path', type=str, default='gpt-3.5-turbo',
                        help='analyzer model path')
    parser.add_argument('--analyzer-message-path', type=str, default='datasets/analyzer/analyzer.txt',
                        help='analyzer message path')
    parser.add_argument('--target-model-path', type=str, default='meta-llama/Llama-2-7b-chat-hf',
                        help='The target model, openai model or open-sourced LLMs')
    parser.add_argument('--target-model-host', type=str, default='vllm',
                        choices=['vllm','ollama','local'],help="The hosting method for the target model")
    parser.add_argument('--max-query', type=int, default=1000,
                        help='The maximum number of queries')
    parser.add_argument('--max-jailbreak', type=int,
                        default=1, help='The maximum jailbreak number')
    parser.add_argument('--energy', type=int, default=1,
                        help='The energy of the fuzzing process')
    # for reproduction only, if you want better performance, use temperature>0
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='The temperature of the fuzzing process')
    parser.add_argument('--seed-selection-strategy', type=str,
                        default='round_robin', help='The seed selection strategy')
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed-path", type=str,
                        default="datasets/prompts/GPTFuzzer.csv")
    parser.add_argument("--question-path", type=str,
                        default="datasets/questions/question_list.csv")
    parser.add_argument("--predictor-model", type=str,
                        default="roberta", choices=['roberta','llamaguard'])
    add_model_args(parser)

    args = parser.parse_args()
    main(args)
