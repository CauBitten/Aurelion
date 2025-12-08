import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class RobustGuard:
    def __init__(self, model_id='meta-llama/LlamaGuard-7b', hf_token=None):
        '''
        Inicializa o Llama Guard. 
        Requer GPU decente. Se não tiver, o código tentará rodar em CPU (lento) ou precisa de Quantização.
        '''
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_id = model_id
        
        print(f'Carregando Llama Guard ({self.device})... Isso pode demorar.')
        
        # Carregando com float16 para economizar memória se tiver GPU
        dtype = torch.float16 if self.device == 'cuda' else torch.float32
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=dtype, 
            device_map='auto' if self.device == 'cuda' else None,
            token=hf_token
        )

    def evaluate(self, prompt: str):
        '''
        Formata o prompt para o template do Llama Guard e avalia.
        Retorna: 'safe' ou 'unsafe'
        '''
        # Template específico do Llama Guard para interação simples
        chat = [
            {'role': 'user', 'content': prompt}
        ]
        
        input_ids = self.tokenizer.apply_chat_template(chat, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(input_ids, max_new_tokens=100, pad_token_id=0)
            
        prompt_len = input_ids.shape[-1]
        generated_output = self.tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
        
        # O Llama Guard retorna 'safe' ou 'unsafe' na primeira linha
        result = generated_output.strip().split('\n')[0]
        return result