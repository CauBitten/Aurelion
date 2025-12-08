# Gaelion 🛡️

Gaelion protects the language model from malicious prompts

## Sistema de Detecção Preventiva de Prompts Maliciosos

### Objetivo

- Detectar e bloquear prompts maliciosos antes que cheguem ao modelo de linguagem.
- Reduzir riscos de jailbreak, vazamento de conteúdo sensível e manipulação intencional.
- Aumentar a segurança de sistemas de IA usando uma arquitetura híbrida eficiente.

### Contribuições

- Arquitetura que combina detectores leves locais + classificadores robustos sob demanda.
- Redução de falsos positivos/negativos via ensemble de modelos avaliadores.
- Menor custo computacional graças à ativação seletiva de modelos maiores.
- Foco em detectar intenção maliciosa, não apenas filtrar conteúdo gerado.
- Uso de treinamento adversarial e análise contextual para identificar ataques sutis.
- Saídas estruturadas: escores de risco, motivos interpretáveis, tokens suspeitos.

## Como executar o projeto

Este projeto utiliza o **uv** para gerenciamento de dependências e ambientes virtuais, garantindo uma execução rápida e reprodutível.

### Pré-requisitos

- **Python 3.10+** instalado.
- **uv** instalado (se não tiver, instale via: <https://docs.astral.sh/uv/getting-started/installation/>).
- Uma conta no **Hugging Face** com token de acesso (necessário para baixar o *Llama Guard*).
- (Opcional, mas recomendado) GPU NVIDIA com drivers CUDA para melhor performance.

### 1. Instalação

Clone o repositório e instale as dependências automaticamente:

```bash
git clone https://github.com/CauBitten/Gaelion.git
cd gaelion
```

### 2. Cria o ambiente virtual e instala todas as libs (torch, transformers, etc.)

```bash
uv sync
```

### 3. Configuração do Token (Hugging Face)

Como o projeto utiliza o Llama Guard (Meta), é necessário autenticação.
Exporte o seu token como variável de ambiente ou crie um arquivo .env:

```bash
HF_TOKEN=your_huggingface_token_here
```

### 4. Treinar a Camada Leve (Obrigatório na 1ª vez)

Antes de rodar o sistema, é preciso gerar os pesos do modelo local (DistilBERT).
Execute o script de treino rápido:

```bash
uv run train_layer1.py
```

Isso criará a pasta ./models/distilbert_gaelion com o modelo treinado.

### 5. Executar o Gaelion

Para testar o sistema via terminal com prompts de exemplo:

```bash
uv run main.py
```

### 6. Executar Experimentos Visuais (Notebook)

Rodar cada célula do notebook após selecionar o virtual environment criado pelo uv.

## Roadmap de Implementação

- 28/11 — 30/11: Definição da arquitetura e organização do repositório.
- 01/12 — 03/12: Implementação do detector leve e calibração inicial.
- 04/12 — 05/12: Criação do medidor de risco e definição dos limiares.
- 06/12 — 07/12: Integração do classificador robusto + ensemble.
- 08/12: Construção da API e integração completa do pipeline.
- 09/12: Testes adversariais, ajustes e validação.
- 10/12: Documentação final e preparação para entrega.
