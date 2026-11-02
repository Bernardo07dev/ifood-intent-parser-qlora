# Brazilian Food Delivery Intent Parser (Qwen 2.5 1.5B + QLoRA)

A practical Machine Learning project exploring **Small Language Model (SLM) fine-tuning** with **QLoRA (4-bit)** for conversational commerce in Brazilian Portuguese (`pt-BR`).

Inspired by real-world food delivery challenges (such as iFood's *Large Commerce Model* and conversational assistants), this model takes an open-ended, natural language food order (as a user would type on WhatsApp or in a conversational search bar) and extracts a **strict, structured JSON** containing cart items, quantities, preparation notes, price limits, delivery time constraints, and dietary restrictions.

## Hugging Face Links
- **Model (LoRA Adapter):** [`BernardoDev07/qwen2.5-1.5b-ifood-intent-ptbr`](https://huggingface.co/BernardoDev07/qwen2.5-1.5b-ifood-intent-ptbr)
- **Dataset:** [`BernardoDev07/ifood-pedidos-intent-ptbr`](https://huggingface.co/datasets/BernardoDev07/ifood-pedidos-intent-ptbr)

## Example

**User Input (`pt-BR`):**
> *"Oi, quero pedir 2 x-bacon (sem cebola) e 1 coca-cola zero lata, quero gastar no maximo 60 reais e que chegue em ate 30 minutos."*

**Model Output (JSON):**
```json
{
  "itens": [
    {"nome": "x-bacon", "quantidade": 2, "observacoes": ["sem cebola"]},
    {"nome": "coca-cola zero lata", "quantidade": 1, "observacoes": []}
  ],
  "preco_maximo": 60.0,
  "tempo_maximo_min": 30,
  "restricoes": []
}
```

## Training Results (QLoRA on Tesla T4)
- **Base Model:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Method:** 4-bit NormalFloat (`nf4`) + LoRA (`r=16`, `lora_alpha=32`, target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`)
- **Dataset Size:** 450 training examples / 50 test examples (3 epochs)
- **Training Time:** ~3m 38s on a single NVIDIA Tesla T4 GPU

| Step | Training Loss |
| :--- | :--- |
| 10 | 1.7198 |
| 20 | 0.7877 |
| 40 | 0.3173 |
| 60 | 0.2126 |
| 80 | **0.1489** |

## Project Structure

- **`create_dataset.py`**: Generates 500 conversational order examples in Brazilian Portuguese and uploads the `train`/`test` splits to Hugging Face Datasets.
- **`train.py`**: Fine-tunes `Qwen/Qwen2.5-1.5B-Instruct` using `bitsandbytes` (4-bit), `peft` (LoRA), and `trl` (`SFTTrainer`), then pushes the trained adapter weights to the Hugging Face Hub.
- **`app.py`**: Simple **Gradio** web interface to test the model interactively.

## How to Run

### 1. Generate and Upload Dataset (Local CPU with `uv`)
```bash
uv run python create_dataset.py
```

### 2. Fine-Tune Model (GPU / Google Colab T4)
```bash
pip install trl peft bitsandbytes datasets accelerate huggingface_hub
export HF_TOKEN="your_hf_token"
python train.py
```
