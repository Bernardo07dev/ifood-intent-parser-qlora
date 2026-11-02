import os
import torch
from datasets import load_dataset
from huggingface_hub import login
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

MODELO_BASE = "Qwen/Qwen2.5-1.5B-Instruct"
MEU_DATASET = "BernardoDev07/ifood-pedidos-intent-ptbr"
MEU_MODELO_SAIDA = "BernardoDev07/qwen2.5-1.5b-ifood-intent-ptbr"


def formatar_para_chat(exemplo, tokenizer):
    mensagens = [
        {"role": "system", "content": exemplo["instruction"]},
        {"role": "user", "content": exemplo["input"]},
        {"role": "assistant", "content": exemplo["output"]},
    ]
    texto = tokenizer.apply_chat_template(mensagens, tokenize=False)
    return {"text": texto}


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("GPU nao detectada! Ative a GPU T4 no Colab.")

    print("GPU detectada:", torch.cuda.get_device_name(0))

    token = os.getenv("HF_TOKEN")
    if token:
        login(token=token)

    print("1. Carregando dataset do Hugging Face...")
    dataset = load_dataset(MEU_DATASET)

    print("2. Carregando tokenizer e modelo base em 4-bit (QLoRA)...")
    tokenizer = AutoTokenizer.from_pretrained(MODELO_BASE)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODELO_BASE,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="cuda:0",
    )

    treino_formatado = dataset["train"].map(lambda x: formatar_para_chat(x, tokenizer))
    teste_formatado = dataset["test"].map(lambda x: formatar_para_chat(x, tokenizer))

    print("3. Configurando o LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = SFTConfig(
        output_dir="./outputs",
        dataset_text_field="text",
        max_length=512,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=10,
        fp16=False,
        bf16=False,
        save_strategy="no",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=treino_formatado,
        eval_dataset=teste_formatado,
        peft_config=lora_config,
        args=training_args,
    )

    print("4. Iniciando o treinamento (deve levar ~2 a 4 minutos na T4)...")
    trainer.train()

    print(f"5. Subindo o modelo treinado para https://huggingface.co/{MEU_MODELO_SAIDA} ...")
    trainer.model.push_to_hub(MEU_MODELO_SAIDA)
    tokenizer.push_to_hub(MEU_MODELO_SAIDA)
    print("Treinamento concluido e modelo publicado no Hugging Face!")


if __name__ == "__main__":
    main()
