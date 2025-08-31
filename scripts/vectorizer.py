import numpy as np

class Seq2Seq_Vectorizer:
    """
    Векторизатор для задач seq‑to‑seq
    Предоставляет методы преобразования списка токенов в числовой массив
    индексов, а также one‑hot представление
    """

    def __init__(self, source_vocab, target_vocab, max_source_len:int, max_target_len:int):
        """
        Args:
            source_vocab (Vocabulary): словарь исходного языка
            target_vocab (Vocabulary): словарь целевого языка
            max_source_len (int): максимальная длина исходной последовательности без BOS/EOS. При векторизации к ней добавляются 2 токена
            max_target_len (int): максимальная длина целевой последовательности без BOS/EOS. При векторизации к ней добавляется 1 токен
        """
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len

    def _vectorize(self, indices:list[int], forced_len:int = -1, mask_index:int = 0) -> np.array:
        """
        Заполняет массив до нужной длины, добавляя `mask_index`

        Args:
            indices (list[int]): список индексов токенов
            forced_len (int): желаемая длина массива. Если <=0 – берётся длина списка
            mask_index (int): индекс паддинга

        Returns:
            np.ndarray[int64]: массив фиксированной длины
        """
        if forced_len <= 0:
            forced_len = len(indices)
        result_vec = np.empty(forced_len, dtype=np.int64)
        result_vec[:len(indices)] = indices
        result_vec[len(indices):] = mask_index
        return result_vec

    def _get_indices(self, tokens:list[str], add_bos:bool = False, add_eos:bool = False, is_target:bool = False) -> list[int]:
        """
        Преобразует список токенов в индексы, добавляя BOS/EOS при необходимости

        Args:
            tokens (list[str]): исходный список токенов
            add_bos (bool): добавить <BOS>
            add_eos (bool): добавить <EOS>
            is_target (bool): если True – используется target_vocab, иначе source_vocab

        Returns:
            list[int]: индексы токенов с BOS/EOS
        """
        indices = []
        cw_vocab = self.target_vocab if is_target else self.source_vocab

        if add_bos:
            indices.append(cw_vocab._bos_index)

        for token in tokens:
            indices.append(cw_vocab.get_token_index(token))

        if add_eos:
            indices.append(cw_vocab._eos_index)
        return indices

    def vectorize_vector_onehot(self, tokens:list[str], is_target:bool = False) -> np.array:
        """
        Преобразует токены в one‑hot вектор

        Args:
            tokens (list[str]): список токенов
            is_target (bool): использовать target_vocab или source_vocab

        Returns:
            np.ndarray[float32]: one‑hot вектор длиной len(vocab)
        """
        cw_vocab = self.target_vocab if is_target else self.source_vocab
        onehot_vec = np.zeros(len(cw_vocab), dtype=np.float32)
        for token in tokens:
            onehot_vec[cw_vocab.get_token_index(token)] = 1.0
        return onehot_vec

    def vectorize(self, source_tokens:list[str], target_tokens:list[str] = None, use_dataset_max_len:bool = True):
        """
        Основной метод векторизации пары (source, target)

        Args:
            source_tokens (list[str]): токены исходного текста
            target_tokens (list[str] | None): токены целевого текста. Если None – возвращается только `source_vec`
            use_dataset_max_len (bool): если True, длина массивов будет ограничена max_source_len+2 / max_target_len+1

        Returns:
            dict: {
                'source_vec': np.ndarray[int64],
                'target_x_vec': np.ndarray[int64] | None,
                'target_y_vec': np.ndarray[int64] | None
            }
            target_x – target без EOS, с BOS (input для decoder)
            target_y – target без BOS, с EOS (labels)
        """
        max_source_len = self.max_source_len + 2 if use_dataset_max_len else -1
        max_target_len = self.max_target_len + 1 if use_dataset_max_len else -1

        source_indices = self._get_indices(source_tokens, add_bos=True, add_eos=True, is_target=False)
        source_vec = self._vectorize(source_indices, max_source_len)

        target_x_vec = target_y_vec = None
        if target_tokens is not None:
            target_x_indices = self._get_indices(target_tokens, add_bos=True, add_eos=False, is_target=True)
            target_x_vec = self._vectorize(target_x_indices, max_target_len)

            target_y_indices = self._get_indices(target_tokens, add_bos=False, add_eos=True, is_target=True)
            target_y_vec = self._vectorize(target_y_indices, max_target_len)

        return {
            'source_vec': source_vec,
            'target_x_vec': target_x_vec,
            'target_y_vec': target_y_vec}

    @classmethod
    def from_dataframe(cls, texts_df, threshold_freq:int = 10):
        """
        Создаёт объект Seq2Seq_Vectorizer из DataFrame

        Требует реализации
        """
        pass

    def to_serializable(self) -> dict:
        """
        Возвращает сериализуемое представление в виде словаря
        """
        return {
            'tokens_vocab': self.tokens_vocab.to_serializable(),
            'label_vocab': self.label_vocab.to_serializable()}
