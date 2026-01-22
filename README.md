# Aurelion 🛡️

<p>Aurelion protege o modelo de linguagem de prompts maliciosos.</p>

## Sistema de Detecção Preventiva de Prompts Maliciosos

### Objetivo

- Detectar e bloquear prompts maliciosos antes que cheguem ao modelo de linguagem.
- Reduzir riscos de jailbreak, de vazamento de conteúdo sensível e de manipulação intencional.
- Aumentar a segurança dos sistemas de IA por meio de uma arquitetura híbrida eficiente.

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
git clone https://github.com/CauBitten/Aurelion.git
cd Aurelion
```

### 2. Cria o ambiente virtual e instala todas as libs (torch, transformers, etc.)

```bash
uv sync
```

### 3. Configuração do Token (Hugging Face)

Como o projeto utiliza o Llama Guard (Meta), é necessária autenticação.
Exporte o seu token como variável de ambiente ou crie um arquivo .env:

```bash
HF_TOKEN=your_huggingface_token_here
```
