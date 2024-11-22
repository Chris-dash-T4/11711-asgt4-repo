from gptfuzzer.llm import LLM
from gptfuzzer.utils.predict import Predictor
from openai import OpenAI
from langchain_ollama import ChatOllama
from ollama import Client as Ollama
from typing import Literal, Optional, List
import concurrent.futures
import logging
import time

class LiteLLM(LLM):
    def __init__(self,
                 model_path,
                 api_key=None,
                 system_message=None
                ):
        super().__init__()

        if not api_key.startswith('sk-'):
            raise ValueError('OpenAI API key should start with sk-')
        if model_path not in ["gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "davinci-002"]:
            raise ValueError(
                'OpenAI model path should be one of: "gpt-3.5-turbo","gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "davinci-002"')
        self.client = OpenAI(
            api_key = api_key,
            base_url="https://cmu.litellm.ai",
        )
        self.model_path = model_path
        self.system_message = system_message if system_message is not None else "You are a helpful assistant."

    def generate(self, prompt, temperature=0, max_tokens=512, n=1, max_trials=10, failure_sleep_time=5):
        for _ in range(max_trials):
            try:
                results = self.client.chat.completions.create(
                    model=self.model_path,
                    messages=[
                        {"role": "system", "content": self.system_message},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=n,
                )
                return [results.choices[i].message.content for i in range(n)]
            except Exception as e:
                logging.warning(
                    f"OpenAI API call failed due to {e}. Retrying {_+1} / {max_trials} times...")
                time.sleep(failure_sleep_time)

        return [" " for _ in range(n)]

    def generate_batch(self, prompts, temperature=0, max_tokens=512, n=1, max_trials=10, failure_sleep_time=5):
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.generate, prompt, temperature, max_tokens, n,
                                       max_trials, failure_sleep_time): prompt for prompt in prompts}
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())
        return results

class LangchainOllamaLLM(LLM):
    def __init__(self,
                 model_path,
                 system_message=None
                ):
        super().__init__()

        # TODO determine error handling for invalid Ollama models
        self.client = ChatOllama(
            model=model_path
        )
        self.model_path = model_path
        self.system_message = system_message if system_message is not None else "You are a helpful assistant."

    def make_client(self, temperature, max_tokens):
        # Keep model as-is if the parameters are unchanged
        if self.client.temperature == temperature and self.client.num_predict == max_tokens:
            return
        self.client = ChatOllama(
            model=self.model_path,
            temperature=temperature,
            num_predict=max_tokens
        )

    def generate(self, prompt, temperature=0.01, max_tokens=512):
        self.make_client(temperature,max_tokens)
        results = self.client.invoke(
            [
                ("system", self.system_message),
                ("human", prompt),
            ]
        )
        return results.content

    def generate_batch(self, prompts, temperature=0, max_tokens=512):
        self.make_client(temperature,max_tokens)
        results = self.client.batch(
            [ 
                [("system", self.system_message),("human", prompt)]
                for prompt in prompts
            ]
        )
        return [res.content for res in results]

class HostedOllamaLLM(LLM):
    def __init__(self,
                 model_path,
                 system_message=None,
                 hostname:str='localhost',
                 port:int=11711,
                ):
        super().__init__()

        # TODO determine error handling for invalid Ollama models
        assert 0 <= port < 65536, f"Invalid TCP/UDP port: {port}"
        if port < 1024:
            print(f'Warning: selected port ({port}) may interfere with other services.')
        self.client = Ollama(host=f'http://{hostname}:{port}')
        self.model_path = model_path
        self.system_message = system_message if system_message is not None else "You are a helpful assistant."

    def generate(self, prompt, temperature=0, max_tokens=512, n=1, max_trials=10, failure_sleep_time=5):
        raise NotImplementedError
        for _ in range(max_trials):
            try:
                results = self.client.chat(
                    model=self.model_path,
                    messages=[
                        {"role": "system", "content": self.system_message},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=n,
                )
                return [results.choices[i].message.content for i in range(n)]
            except Exception as e:
                logging.warning(
                    f"OpenAI API call failed due to {e}. Retrying {_+1} / {max_trials} times...")
                time.sleep(failure_sleep_time)

        raise NotImplementedError
        return [" " for _ in range(n)]

    def generate_batch(self, prompts, temperature=0, max_tokens=512, n=1, max_trials=10, failure_sleep_time=5):
        raise NotImplementedError
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.generate, prompt, temperature, max_tokens, n,
                                       max_trials, failure_sleep_time): prompt for prompt in prompts}
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())
        return results

class ReprompterLLM(LLM):
    WarningMethod = Optional[Literal['generic','specific']]

    def __init__(self,
                 base_llm : LLM,
                 predictor : Predictor,
                 warning : WarningMethod = None,
                 analyzer_llm : Optional[LLM] = None,
                ):
        super().__init__()
        self.base = base_llm
        self.predictor = predictor
        self.warning = warning
        self.analyzer_llm = analyzer_llm

    def generate(self, prompt, analyzer_args={}, **kwargs):
        output = self.base.generate(prompt,**kwargs)
        prediction = self.predictor.predict(output)[0]
        if prediction != 0:
            match self.warning:
                case 'generic':
                    new_prompt = "WARNING: a previous response to this prompt violated the provided content policy.\n"+\
                        "Do not generate content that is harmful, illegal, or unethical.\n\n"+prompt
                    output = self.base.generate(new_prompt,**kwargs)

                case 'generic':
                    # System message for analyzer should be set to instruct the model properly
                    analysis = self.analyzer_llm.generate(output,**analyzer_args)
                    new_prompt = "WARNING: a previous response to this prompt violated the provided content policy.\n"+\
                        "The following were identified as potential triggers: "+analysis+"\n"+\
                        "Do not generate content that is harmful, illegal, or unethical.\n\n"+prompt
                    output = self.base.generate(new_prompt,**kwargs)

                case _:
                    # run the LM again as-is
                    output = self.base.generate(prompt,**kwargs)

        return output

    def generate_batch(self, prompts, temperature=0, max_tokens=512):
        raise NotImplementedError
        self.make_client(temperature,max_tokens)
        results = self.client.batch(
            [ 
                [("system", self.system_message),("human", prompt)]
                for prompt in prompts
            ]
        )
        return [res.content for res in results]
