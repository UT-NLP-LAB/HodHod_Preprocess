import json
import math
from tqdm import tqdm
from piraye import NormalizerBuilder
from piraye.normalizer_builder import Config
from piraye.nltk_tokenizer import NltkTokenizer
import re
import os
from multiprocessing import Pool, cpu_count
from spacy.lang.en import English
import spacy


class Preprocessor:
    def __init__(self):
        self.normalizer = NormalizerBuilder(
            [Config.PUNCTUATION_FA, Config.ALPHABET_FA, Config.DIGIT_FA, Config.ALPHABET_EN, Config.DIGIT_EN,
             Config.DIGIT_FA, Config.DIACRITIC_DELETE, Config.SPACE_NORMAL, Config.PUNCTUATION_FA,
             Config.PUNCTUATION_EN],
            remove_extra_spaces=True,
            tokenization=True).build()
        self.tokenizer = NltkTokenizer()
        self.nlp = English()
        self.spacy_tokenizer = self.nlp.tokenizer

    def preprocess_line(self, text: str):
        text = self.normalizer.normalize(text)
        text = re.sub(r'\b[A-Z]+\b', '', text)
        text = re.sub(r'<[^>]+>', '', text)
        sents = [sen for sen in self.tokenizer.sentence_tokenize(text)]
        list_of_sentences = [[str(token) for token in self.spacy_tokenizer(sen)] for sen in sents]
        tokens = [item for sublist in list_of_sentences for item in sublist]
        text = ' '.join(tokens)
        return text

    def preprocess_document(self, text: str):
        lines = text.splitlines()
        return '\n'.join([self.preprocess_line(text_line) for text_line in lines])

    def preprocess_file(self, file_path: str):
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        source = os.path.basename(os.path.dirname(file_path))
        file_type = os.path.splitext(file_path)[-1]
        normalizer_folder = f'../data/normalized/{source}/'
        res_path = f'{normalizer_folder}{file_name}.jsonl'
        if not os.path.exists(normalizer_folder):
            os.makedirs(normalizer_folder)
        with open(file_path, 'r', encoding='utf-8') as fh:
            with open(res_path, 'w', encoding='utf-8') as f:
                if file_type == '.jsonl':
                    for line in tqdm(fh):
                        json_data = json.loads(line)
                        json_data['text'] = self.preprocess_document(json_data['text'])
                        json_data['source'] = source
                        json.dump(json_data, f, ensure_ascii=False)
                        f.write('\n')
                elif file_type == '.json':
                    json_datas = json.load(fh)
                    for json_data in tqdm(json_datas):
                        json_data['text'] = self.preprocess_line(json_data['text'])
                        json_data['source'] = source
                        json.dump(json_data, f, ensure_ascii=False)
                        f.write('\n')
        return True

    def preprocess_files(self, data_dir: str):
        files = sorted(os.listdir(data_dir))
        files = [data_dir + "/" + file for file in files if os.path.isfile(data_dir + "/" + file)]
        n_proc = cpu_count()
        n_chunks = math.ceil(len(files) / n_proc)
        remain = len(files) % n_proc
        if n_chunks == 1 and remain:
            n_proc = remain
        print(f"resetting to {n_proc} for number of processes")
        # files = [files[i: i + n_chunks] for i in range(0, len(files), n_chunks)]

        with Pool(processes=n_proc) as pool:
            pbar = tqdm(
                pool.imap(
                    self.preprocess_file, files
                ),
                total=len(files),
            )
            for test in pbar:
                pbar.update()
                if test:
                    continue
