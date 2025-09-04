class CustomDataset:
    """
    Dataset для Seq2Seq задач
    
    При инициализации класс делит исходный DataFrame на три раздела: train,
    validation и test, а также сохраняет ссылки на каждый из них
    Для дальнейшего обучения удобно переключаться между наборами через
    set_dataframe_split()
    """

    def __init__(self, dataframe, tokenizer, vectorizer):
        """
        Args:
            dataframe (pd.DataFrame): исходный набор данных с колонками
                ['source_text', 'target_text', 'split']
            tokenizer: объект токенизатора. (не используется в текущем коде,
                но оставлен для совместимости)
            vectorizer (Seq2Seq_Vectorizer): объект, отвечающий за преобразование
                текста в числовые представления

        Инициализирует внутренние DataFrame‑ы и их длины, а также
        словарь lookup_split, позволяющий быстро переключаться между
        набором train/validation/test
        """
        self._vectorizer = vectorizer
        self._tokenizer = tokenizer

        # Исходный датафрейм
        self._main_df = dataframe

        # Разделяем на три поднаборы
        self._train_df = self._main_df[self._main_df.split == 'train']
        self._train_len = len(self._train_df)

        self._valid_df = self._main_df[self._main_df.split == 'validation']
        self._valid_len = len(self._valid_df)

        self._test_df = self._main_df[self._main_df.split == 'test']
        self._test_len = len(self._test_df)

        # Словарь для быстрого доступа к нужному поднабору
        self._lookup_split = {
            'train': (self._train_df, self._train_len),
            'validation': (self._valid_df, self._valid_len),
            'test': (self._test_df, self._test_len)
        }

        # Устанавливаем «текущий» набор данных в train
        self.set_dataframe_split('train')

    def __getitem__(self, index:int):
        """
        Возвращает одну обучающую пару (source‑target) в виде словаря, готового к подаче в DataLoader

        Args:
            index (int): индекс строки внутри текущего поднабора данных

        Returns:
            dict: {
                'source_vec':  np.ndarray[int64]  # входной массив
                'target_x_vec': np.ndarray[int64] # target без EOS, с BOS
                'target_y_vec': np.ndarray[int64] # target без BOS, с EOS
            }
        """
        row = self._cw_dataframe.iloc[index]
        vector_dict = self._vectorizer.vectorize(
            source_tokens=row['source_text'],
            target_tokens=row['target_text'],
            use_dataset_max_len=True
        )
        return {
            'source_vec': vector_dict['source_vec'],
            'target_x_vec': vector_dict['target_x_vec'],
            'target_y_vec': vector_dict['target_y_vec']
        }

    def __len__(self):
        """
        Длина текущего поднабора данных
        """
        return self._cw_df_len

    def set_dataframe_split(self, split:str='train'):
        """
        Переключает внутренний DataFrame на нужный раздел

        Args:
            split (str): один из ['train', 'validation', 'test']. По умолчанию train
        """
        self._cw_dataframe, self._cw_df_len = self._lookup_split[split]
