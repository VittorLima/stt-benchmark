import re
from num2words import num2words


def numbers_to_words(texto: str) -> str:
    """Converte números em um texto para suas formas por extenso usando num2words.

    Args:
        texto (str): O texto de entrada que pode conter números.

    Returns:
        str: O texto com os números convertidos para palavras.
    """

    def replace_number(match):
        # pega o trecho do texto que foi encontrado pelo re.sub, ex: "1500,50"
        numero = match.group()

        try:
            # troca vírgula por ponto para o Python conseguir converter para float
            normalizado = numero.replace(",", ".")

            if "." in normalizado:
                # é decimal (ex: "3.14") → converte como float
                return num2words(float(normalizado), lang="pt_BR")
            else:
                # é inteiro (ex: "42") → converte como int
                return num2words(int(numero), lang="pt_BR")

        except Exception:
            # se algo der errado, devolve o número original sem modificar
            return numero

    # percorre o texto inteiro procurando números inteiros e decimais
    # para cada número encontrado, chama replace_number(match) automaticamente
    resultado = re.sub(r"\d+(?:[.,]\d+)?", replace_number, texto)

    return resultado
