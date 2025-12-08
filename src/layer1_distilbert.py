import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch.nn.functional as F

class FastDetector:
    def __init__(self, model_path='./models/distilbert_gaelion'):
        '''
        Carrega o modelo leve treinado localmente.
        Se não houver modelo treinado, carrega o base (para teste).
        '''
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
            self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        except OSError:
            print('Modelo treinado não encontrado. Usando \'distilbert-base-uncased\' sem fine-tuning (apenas para estrutura).')
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            self.model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
        
        self.model.to(self.device)
        self.model.eval()

    def predict(self, prompt: str):
        '''
        Retorna: (probabilidade_risco, label_predito)
        0 = Seguro, 1 = Malicioso
        '''
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
        
        risk_score = probs[0][1].item() # Probabilidade da classe 1 (Malicioso)
        return risk_score