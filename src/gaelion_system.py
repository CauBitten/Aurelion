from src.layer1_distilbert import FastDetector
# Importamos a camada 2 apenas se necessário ou inicializamos lazy, 
# mas aqui vamos instanciar para demonstração.
from src.layer2_llamaguard import RobustGuard

class Gaelion:
    def __init__(self, hf_token):
        print('🛡️ Inicializando Gaelion System...')
        # Inicializa a camada rápida
        self.fast_layer = FastDetector()
        
        # Inicializa a camada robusta (Llama Guard)
        # Nota: Em produção, você poderia carregar isso em outro serviço/API para economizar RAM
        self.robust_layer = RobustGuard(hf_token=hf_token)
        
        # Limiares de decisão
        self.safe_threshold = 0.10  # Abaixo disso, confia que é seguro
        self.danger_threshold = 0.90 # Acima disso, confia que é ataque

    def check_prompt(self, prompt: str):
        print(f'\n--- Analisando: \'{prompt}\' ---')
        
        # 1. Camada Rápida (DistilBERT)
        risk_score = self.fast_layer.predict(prompt)
        print(f'⚡ Camada 1 (DistilBERT) Risco: {risk_score:.4f}')

        # Lógica de Funil (Pipeline Híbrido)
        
        # Caso A: Ataque Óbvio (Fast Reject)
        if risk_score > self.danger_threshold:
            return {
                'status': 'BLOCKED',
                'layer': 'Layer 1 (Fast)',
                'reason': 'High confidence malware signature',
                'risk_score': risk_score
            }

        # Caso B: Seguro Óbvio (Fast Pass)
        if risk_score < self.safe_threshold:
            return {
                'status': 'ALLOWED',
                'layer': 'Layer 1 (Fast)',
                'reason': 'Low risk detected',
                'risk_score': risk_score
            }

        # Caso C: Incerteza -> Chama o Especialista (Llama Guard)
        print('Incerteza na Camada 1. Acionando Llama Guard...')
        guard_verdict = self.robust_layer.evaluate(prompt)
        
        if 'unsafe' in guard_verdict:
            return {
                'status': 'BLOCKED',
                'layer': 'Layer 2 (Llama Guard)',
                'reason': 'Semantic violation detected',
                'details': guard_verdict
            }
        else:
            return {
                'status': 'ALLOWED',
                'layer': 'Layer 2 (Llama Guard)',
                'reason': 'Verified safe by expert model',
                'details': guard_verdict
            }