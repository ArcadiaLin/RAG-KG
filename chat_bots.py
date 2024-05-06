from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from transformers.generation.utils import GenerationConfig

device = "cuda" # the device to load the model onto
initialize_prompt = """你是一个金融领域助手，你接下来的回答将严格按照我所提供的资料和要求进行作答，并且无需包含任何多余的解释，是否理解，如果理解请返回1"""


def debug(func):
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        if self.debug_mode:
            print("Debugging Messages:\n")
            for message in self.messages:
                print(message)
        return result
    return wrapper


class TemplateChatBot():
    # for qwen1.5&llama3&Yi
    def __init__(self, checkpoint_dir: str, debug=True):
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint_dir,
            torch_dtype="auto",
            device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        self.messages = [{'role': 'system', 'content': 'You are a helpful assistant.'}]
        self.debug_mode=debug
    
    @debug
    def generate(self, message: list) -> str:
        text = self.tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(device)
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(
                model_inputs.input_ids,
                generated_ids
                )
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        self.messages.append({"role":"assistant", "content": f"{response}"})
        return response
    
    def initialize(self):
        self.messages.append({"role" :"user", "content": initialize_prompt})
        response = self.generate(message=self.messages)
        return response
    
    def chat(self, query: str) -> str:
        self.messages.append({"role": "user", "content": query})
        response = self.generate(self.messages)
        return response
    

class GLMChat():
    def __init__(self, checkpoint_dir: str, debug=False):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(checkpoint_dir, trust_remote_code=True).half().cuda()
        self.model = self.model.eval()
        self.messages = []
    
    @debug
    def generate(self, message: list) -> str:
        response, self.messages = self.model.chat(self.tokenizer, message, history=self.messages)
        return response
    
    def chat(self, query: str):
        response = self.generate(query)
        return response