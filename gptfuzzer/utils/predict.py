from transformers import RobertaForSequenceClassification, RobertaTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class Predictor:
    def __init__(self, path):
        self.path = path

    def predict(self, sequences):
        raise NotImplementedError("Predictor must implement predict method.")


class RoBERTaPredictor(Predictor):
    def __init__(self, path, device='cuda'):
        super().__init__(path)
        self.device = device
        self.model = RobertaForSequenceClassification.from_pretrained(
            self.path).to(self.device)
        self.tokenizer = RobertaTokenizer.from_pretrained(self.path)

    def predict(self, sequences):
        inputs = self.tokenizer(sequences, padding=True, truncation=True,
                                max_length=512, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        _, predicted_classes = torch.max(predictions, dim=1)
        predicted_classes = predicted_classes.cpu().tolist()
        return predicted_classes
    

class LlamaGuardPredictor(Predictor):
    def __init__(self, path, device='cuda'):
        super().__init__(path)
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            self.path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.path)

    def predict(self, sequences):
        '''
        So far 2 different implementation attempts here
        In the first one, there was an error that said we had to define a padding token
        "self.tokenizer.pad_token = self.tokenizer.eos_token" is an attempt to do that
        but it keeps saying "tokenizer is not defined"
        '''
        self.tokenizer.pad_token = self.tokenizer.eos_token
        inputs = self.tokenizer(sequences, padding=True, truncation=True,
                                max_length=512, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        _, predicted_classes = torch.max(predictions, dim=1)
        predicted_classes = predicted_classes.cpu().tolist()
        return predicted_classes


        '''
        This is the second implementation.
        It is based on the sample code in the model card https://huggingface.co/meta-llama/Llama-Guard-3-8B
        Its problem is that apply_chat_template only wants an input formatted as user/assistant/user/assistant...
        '''
        # input_ids = self.tokenizer.apply_chat_template(sequences, return_tensors="pt").to(self.device)
        # with torch.no_grad():
        #     outputs = self.model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
        # prompt_len = input_ids.shape[-1]
        # return self.tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
