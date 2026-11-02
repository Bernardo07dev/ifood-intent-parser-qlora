import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELO_BASE = "Qwen/Qwen2.5-1.5B-Instruct"
MEU_LORA = "BernardoDev07/qwen2.5-1.5b-ifood-intent-ptbr"

SYSTEM_PROMPT = (
    "Voce e um assistente de pedidos de delivery. "
    "Extraia os dados do pedido do usuario e retorne APENAS um JSON valido "
    "com as chaves: itens (lista com nome, quantidade, observacoes), "
    "preco_maximo, tempo_maximo_min e restricoes."
)

print("Carregando modelo e adaptador LoRA...")
tokenizer = AutoTokenizer.from_pretrained(MODELO_BASE)
base_model = AutoModelForCausalLM.from_pretrained(MODELO_BASE, torch_dtype=torch.float32)

try:
    model = PeftModel.from_pretrained(base_model, MEU_LORA)
    print("Adaptador LoRA carregado com sucesso!")
except Exception as e:
    print(f"Aviso: Adaptador LoRA ainda nao publicado ({e}). Usando modelo base.")
    model = base_model


def extrair_pedido(frase_usuario):
    mensagens = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": frase_usuario},
    ]

    texto_entrada = tokenizer.apply_chat_template(
        mensagens, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([texto_entrada], return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=False,
        )

    novos_tokens = outputs[0][inputs.input_ids.shape[1] :]
    resposta = tokenizer.decode(novos_tokens, skip_special_tokens=True)
    return resposta


demo = gr.Interface(
    fn=extrair_pedido,
    inputs=gr.Textbox(
        lines=3,
        label="Digite seu pedido (como falaria no WhatsApp/iFood)",
        placeholder="Ex: Mano, pede 2 x-bacon sem cebola e 1 coca zero lata, quero gastar ate 60 reais",
    ),
    outputs=gr.Code(language="json", label="JSON Estruturado Extraído pelo Modelo"),
    title="🍔 Extrator de Intenção de Pedidos (Qwen 2.5 1.5B + LoRA)",
    description=(
        "Projeto de estudo de Fine-Tuning (QLoRA) para transformar pedidos abertos em português "
        "em um JSON estruturado de carrinho e filtros de busca."
    ),
    examples=[
        ["Oi, quero pedir 2 x-bacon (sem cebola) e 1 coca-cola zero lata, quero gastar no maximo 60 reais e que chegue em ate 30 minutos."],
        ["To com fome, vê pra mim 1 acai de 500ml (sem granola), sou intolerante a lactose."],
        ["Mano, manda 1 pizza meia calabresa meia marguerita (borda recheada de catupiry) e 1 guarana antarctica 2 litros gelado."],
    ],
)

if __name__ == "__main__":
    demo.launch()
