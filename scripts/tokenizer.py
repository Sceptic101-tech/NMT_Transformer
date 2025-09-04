import regex as re

class SeparatorTokenizer:
    """
    Простая реализация алгоритма токенизации по разделителю
    """

    def __init__(self):
        pass

    def tokenize(self, text:str, separator:str = None) -> list[str]:
        """
        Разбивает строку на список токенов

        Args:
            text (str): исходный текст
            separator (str | None): символ/строка, по которой происходит
                разбиение. Если None – используется стандартное split() без
                аргументов (разделитель «пробел»)
        Returns:
            list[str]: список токенов
        """
        # Разделяем знаки препинания пробелами, чтобы они стали отдельными токенами
        text = re.sub(r'([^\w\s])', r' \1 ', text)
        text = re.sub(r'[\t\n\r\f\v]', r' ', text)
        return text.split(separator)
