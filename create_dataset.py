import json
import os
import random
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()
random.seed(42)

SYSTEM_PROMPT = (
    "Voce e um assistente de pedidos de delivery. "
    "Extraia os dados do pedido do usuario e retorne APENAS um JSON valido "
    "com as chaves: itens (lista com nome, quantidade, observacoes), "
    "preco_maximo, tempo_maximo_min e restricoes."
)

PRATOS = [
    ("x-bacon", ["sem cebola", "bem passado", "maionese a parte", "sem tomate"]),
    ("smash burger duplo", ["ponto da carne ao ponto", "sem picles", "com extra de cheddar"]),
    ("pizza meia calabresa meia marguerita", ["borda recheada de catupiry", "sem azeitona", "bem assada"]),
    ("marmita de frango grelhado", ["sem feijao", "so arroz integral", "salada separada"]),
    ("acai de 500ml", ["com leite ninho e morango", "sem granola", "banana a parte"]),
    ("combo 20 pecas de sushi", ["sem cream cheese", "mandar shoyu light", "bastante gengibre"]),
    ("pastel de carne com queijo", ["bem sequinho", "mandar vinagrete"]),
    ("esfiha de carne", ["aberta", "mandar limao"]),
    ("poke de salmao", ["base de arroz japones", "sem cebola roxa", "molho tarê a parte"]),
    ("parmegiana de carne", ["arroz soltinho", "batata frita separada pro nao murchar"]),
]

BEBIDAS = [
    ("coca-cola zero lata", ["bem gelada", "com gelo e limao"]),
    ("guarana antarctica 2 litros", ["gelado"]),
    ("suco de laranja natural 500ml", ["sem acucar", "com pouco gelo"]),
    ("agua com gas", ["gelada"]),
]

RESTRICOES_OPCOES = [
    ("sou intolerante a lactose", "sem lactose"),
    ("tenho alergia a amendoim", "alergia a amendoim"),
    ("sou vegetariano", "vegetariano"),
    ("nao como gluten", "sem gluten"),
]


def montar_exemplo():
    prato, obs_possiveis = random.choice(PRATOS)
    qtd_prato = random.choice([1, 1, 2, 3])
    obs_prato = [random.choice(obs_possiveis)] if random.random() < 0.7 else []

    itens = [{"nome": prato, "quantidade": qtd_prato, "observacoes": obs_prato}]

    if random.random() < 0.6:
        bebida, obs_beb_possiveis = random.choice(BEBIDAS)
        qtd_beb = random.choice([1, 2])
        obs_beb = [random.choice(obs_beb_possiveis)] if random.random() < 0.4 else []
        itens.append({"nome": bebida, "quantidade": qtd_beb, "observacoes": obs_beb})

    preco_max = random.choice([None, 35.0, 45.0, 50.0, 60.0, 80.0, 100.0])
    tempo_max = random.choice([None, 25, 30, 40, 45, 60])

    restricoes = []
    frase_restricao = ""
    if random.random() < 0.25:
        fala, tag = random.choice(RESTRICOES_OPCOES)
        restricoes.append(tag)
        frase_restricao = f", {fala}"

    abertura = random.choice(["Oi, quero pedir", "Mano, manda", "Gostaria de pedir", "To com fome, vê pra mim", "Pede ai"])
    texto_obs_prato = f" ({obs_prato[0]})" if obs_prato else ""
    frase = f"{abertura} {qtd_prato} {prato}{texto_obs_prato}"

    if len(itens) > 1:
        beb = itens[1]
        texto_obs_beb = f" {beb['observacoes'][0]}" if beb["observacoes"] else ""
        frase += f" e {beb['quantidade']} {beb['nome']}{texto_obs_beb}"

    if preco_max is not None:
        frase += f", quero gastar no maximo {int(preco_max)} reais"

    if tempo_max is not None:
        frase += f" e que chegue em ate {tempo_max} minutos"

    frase += frase_restricao + "."

    saida_json = {
        "itens": itens,
        "preco_maximo": preco_max,
        "tempo_maximo_min": tempo_max,
        "restricoes": restricoes,
    }

    return {
        "instruction": SYSTEM_PROMPT,
        "input": frase,
        "output": json.dumps(saida_json, ensure_ascii=False),
    }


def main():
    exemplos = [montar_exemplo() for _ in range(500)]

    treino = Dataset.from_list(exemplos[:450])
    teste = Dataset.from_list(exemplos[450:])

    dataset_final = DatasetDict({"train": treino, "test": teste})
    print("Exemplo gerado:")
    print("Frase:", treino[0]["input"])
    print("JSON esperado:", treino[0]["output"])

    token = os.getenv("HF_TOKEN")
    user = os.getenv("HF_USERNAME", "BernardoDev07")

    if token:
        login(token=token)
        repo_id = f"{user}/ifood-pedidos-intent-ptbr"
        print(f"\nSubindo dataset para https://huggingface.co/datasets/{repo_id} ...")
        dataset_final.push_to_hub(repo_id)
        print("Dataset publicado com sucesso!")
    else:
        print("\nHF_TOKEN nao encontrado no .env. Pulando upload.")


if __name__ == "__main__":
    main()
