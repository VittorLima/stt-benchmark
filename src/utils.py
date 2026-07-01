import re
from num2words import num2words


def numbers_to_words(texto: str) -> str:
    """Converte números em um texto para suas formas por extenso usando num2words.

    Args:
        texto: O texto de entrada que pode conter números.

    Returns:
        O texto com os números convertidos para palavras.
    """

    def replace_number(match: re.Match[str]) -> str:
        numero = match.group()

        try:
            # pt-BR usa vírgula como separador decimal; float() exige ponto
            normalizado = numero.replace(",", ".")

            if "." in normalizado:
                return num2words(float(normalizado), lang="pt_BR")
            else:
                return num2words(int(numero), lang="pt_BR")

        except (ValueError, OverflowError):
            return numero

    resultado = re.sub(r"\d+(?:[.,]\d+)?", replace_number, texto)

    return resultado
