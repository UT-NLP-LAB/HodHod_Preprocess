import json
import time

from tqdm import tqdm
from piraye import NormalizerBuilder
from piraye.tasks.normalizer.normalizer_builder import Config
from piraye.tasks.tokenizer.nltk_tokenizer import NltkTokenizer
import re
import os
from multiprocessing import Pool, cpu_count
from spacy.lang.en import English
import csv
import string
from .utils import get_all_files
from collections import Counter


class Preprocessor:
    def __init__(self, threshold=30, char_threshold=35, min_threshold=50):
        self.log_path = None
        self.normalizer = NormalizerBuilder(
            [Config.PUNCTUATION_FA, Config.ALPHABET_FA, Config.DIGIT_FA, Config.ALPHABET_EN, Config.DIGIT_EN,
             Config.DIGIT_FA, Config.DIACRITIC_DELETE, Config.SPACE_NORMAL, Config.PUNCTUATION_FA,
             Config.PUNCTUATION_EN],
            remove_extra_spaces=True,
            tokenization=True).build()
        self.tokenizer = NltkTokenizer()
        self.nlp = English()
        self.spacy_tokenizer = self.nlp.tokenizer
        self.data_path = "data/"

        self.threshold = threshold
        self.min_threshold = min_threshold
        self.char_threshold = char_threshold
        self.normalized_folder = ""
        self.number_of_total_rows = 0
        self.number_of_filtered_rows = 0
        self.filtering = True

    def get_features(self, s: str):
        s = s.lower()
        s = s.translate(str.maketrans("", "", string.punctuation))
        s = s.translate(str.maketrans("", "", ":><؟!.،,?"))
        s = re.sub(r"\s+", " ", s.strip())
        return len(s.split()) > self.threshold, s

    def is_persian_text(self, text):
        persian_chars = sum(1 for char in text if '\u0600' <= char <= '\u06FF')
        total_chars = len(text)
        percentage = (persian_chars / total_chars) * 100 if total_chars > 0 else 0
        return percentage > self.char_threshold

    def most_repeated_word_over_threshold(self, text):
        # Tokenize the text into words
        words = re.findall(r'\b\w+\b', text)
        # Count the frequency of each word
        word_counts = Counter(words)
        if not word_counts:
            return False
        # Find the word with the maximum frequency
        most_common_word, max_count = word_counts.most_common(1)[0]
        # Calculate the percentage of the text occupied by this word
        percentage = (max_count / len(words)) * 100
        return percentage < self.min_threshold

    def check_short_lines(self, text):
        lines = text.split('\n')
        short_line_count = sum(1 for line in lines if len(line.strip()) < 15)
        total_line_count = len(lines)
        portion_short_lines = (short_line_count / total_line_count) * 100
        return portion_short_lines < self.min_threshold

    def preprocess_line(self, text: str):
        text = self.normalizer.normalize(text)
        text = re.sub(r'\b[A-Z]+\b', '', text)
        text = re.sub(r'<[^>]+>', '', text)
        text = text.translate(str.maketrans("", "", "‎‏‪‫ ‭‮"))
        sents = [sen for sen in self.tokenizer.sentence_tokenize(text)]
        list_of_sentences = [[str(token) for token in self.spacy_tokenizer(sen)] for sen in sents]
        tokens = [item for sublist in list_of_sentences for item in sublist]
        text = ' '.join(tokens)
        return text

    def preprocess_document(self, text: str):
        lines = text.splitlines()
        return '\n'.join([self.preprocess_line(text_line) for text_line in lines])

    def write_json(self, json_data, f):
        if self.filtering:
            is_clean_text, s = self.get_features(json_data['text'])
            if is_clean_text and self.is_persian_text(s) and self.check_short_lines(
                    s) and self.most_repeated_word_over_threshold(s):
                json.dump(json_data, f, ensure_ascii=False)
                f.write('\n')
        else:
            json.dump(json_data, f, ensure_ascii=False)
            f.write('\n')

    def preprocess_file(self, file_path: str):
        file_name = os.path.splitext(os.path.basename(file_path))[0].replace(" ", "")
        file_type = os.path.splitext(file_path)[-1]
        source = os.path.dirname(file_path.split(self.data_path)[1])
        self.normalized_folder = f'./result/normalized/{source}'
        res_path = f'{self.normalized_folder}/{file_name}.jsonl'
        if not os.path.exists(self.normalized_folder):
            os.makedirs(self.normalized_folder)
        with open(file_path, 'r', encoding='utf-8') as fh:
            with open(res_path, 'w', encoding='utf-8') as f:
                if file_type == '.jsonl':
                    for i, line in enumerate(fh):
                        try:
                            json_data = json.loads(line)
                            json_data['id'] = f"{source}-{file_name}-{i}"
                            preprocessed_text = self.preprocess_document(json_data['text'])
                            if preprocessed_text:
                                json_data['text'] = preprocessed_text
                                json_data['source'] = source
                                self.write_json(json_data, f)
                        except json.decoder.JSONDecodeError:
                            print("Error in reading file: ", file_path)
                elif file_type == '.json':
                    try:
                        json_datas = json.load(fh)
                        for i, json_data in enumerate(json_datas):
                            json_data['id'] = f"{source}-{file_name}-{i}"
                            json_data['text'] = self.preprocess_line(json_data['text'])
                            json_data['source'] = source
                            self.write_json(json_data, f)
                    except json.decoder.JSONDecodeError:
                        print("Error in reading file: ", file_path)
                elif file_type == '.csv':
                    try:
                        csv_reader = csv.reader(fh)
                        columns = next(csv_reader)
                        for i, row in enumerate(csv_reader):
                            json_data = {'id': f"{source}-{file_name}-{i}"}
                            for index in range(len(columns)):
                                json_data[columns[index]] = row[index]
                            self.write_json(json_data, f)
                    except:
                        print("Error in reading file: ", file_path)
        return self.normalized_folder

    def normalize_files(self, all_files: list[str]):
        n_proc = cpu_count()
        print(f"resetting to {n_proc} for number of processes")
        with Pool(processes=n_proc) as pool:
            pbar = tqdm(
                pool.imap(
                    self.preprocess_file, all_files
                ),
                total=len(all_files),
            )
            for test in pbar:
                pbar.update()
                if test:
                    continue

    def preprocess_files(self, sub_folder_name: str, filtering=True):
        start_time = time.time()
        data_dir = self.data_path + sub_folder_name
        all_files = get_all_files(data_dir)
        log_name = sub_folder_name
        log_path = f'./result/logs/{log_name}.txt'
        if not os.path.exists('./result/logs'):
            os.makedirs('./result/logs')
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"Total files: {len(all_files)}\n")
        self.filtering = filtering
        self.normalize_files(all_files)
        count_words_filtered = 0
        print("total : ", len(all_files))
        for file_path in tqdm(all_files):
            file_name = os.path.splitext(os.path.basename(file_path))[0].replace(" ", "")
            source = os.path.dirname(file_path.split(self.data_path)[1])
            self.normalized_folder = f'./result/normalized/{source}'
            res_path = f'{self.normalized_folder}/{file_name}.jsonl'
            with open(res_path, 'r', encoding='utf-8') as fh:
                for i, line in enumerate(fh):
                    json_data = json.loads(line)
                    self.number_of_filtered_rows += 1
                    count_words_filtered += len(json_data['text'].split())
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"Number of words after filtering: {count_words_filtered}\n")
            f.write(f"Number of rows after filtering: : {self.number_of_filtered_rows}\n")
            f.write(f"Normalizing Time: {(time.time() - start_time):.3f} s\n---------------------------\n")
