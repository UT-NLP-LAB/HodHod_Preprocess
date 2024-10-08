import csv
import json
import os
import re
import string
import time
from collections import Counter
from multiprocessing import Pool, cpu_count

import pandas as pd
from piraye import NormalizerBuilder
from piraye.tasks.normalizer.normalizer_builder import Config
from piraye.tasks.tokenizer.nltk_tokenizer import NltkTokenizer
from spacy.lang.en import English
from tqdm import tqdm
from transformers import AutoTokenizer

from .utils import get_all_files

wierd_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           u"\U0001f926-\U0001f937"
                           u'\U00010000-\U0010ffff'
                           u"\u200d"
                           u"\u2640-\u2642"
                           u"\u2600-\u2B55"
                           u"\u23cf"
                           u"\u23e9"
                           u"\u231a"
                           u"\u3030"
                           u"\ufe0f"
                           u"\u2069"
                           u"\u2066"
                           # u"\u200c"
                           u"\u2068"
                           u"\u2067"
                           u"\u0640"
                           "]+", flags=re.UNICODE)


class Preprocessor:
    def __init__(self, token_ratio_quality=False, threshold=100, char_threshold=35, min_threshold=50,
                 line_threshold=20, number_threshold=0.2):
        self.log_path = None
        self.normalizer = NormalizerBuilder(
            [Config.PUNCTUATION_FA, Config.ALPHABET_FA, Config.DIGIT_FA, Config.ALPHABET_EN, Config.DIGIT_EN,
             Config.DIGIT_FA, Config.DIACRITIC_DELETE, Config.SPACE_KEEP, Config.PUNCTUATION_FA,
             Config.PUNCTUATION_EN],
            remove_extra_spaces=True,
            tokenization=True).build()
        self.tokenizer = NltkTokenizer()
        self.nlp = English()
        self.spacy_tokenizer = self.nlp.tokenizer
        self.nlp.max_length = 15000000
        self.data_path = "data/"

        self.threshold = threshold
        self.min_threshold = min_threshold
        self.char_threshold = char_threshold
        self.line_threshold = line_threshold  # number of words in each line
        self.normalized_folder = ""
        self.number_of_total_rows = 0
        self.number_of_filtered_rows = 0
        self.filtering = True
        self.number_threshold = number_threshold
        csv.field_size_limit(1000000000)
        self.token_ratio_quality = token_ratio_quality
        if token_ratio_quality:
            self.base_model_id = 'FacebookAI/xlm-roberta-large'
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            self.bert_tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_id,
                add_eos_token=True,
                add_bos_token=True,
                use_fast=True,
                padding=False,
                truncation=False,
            )

    def custom_tokenize(self, text):
        doc = self.nlp(text)
        tokens = []
        is_hashtag = False
        for token in doc:
            if token.text == "#":
                is_hashtag = True
            else:
                cr_text = ''
                if is_hashtag:
                    cr_text += '#'
                cr_text += token.text
                tokens.append(cr_text)
                is_hashtag = False
        return tokens

    def get_features(self, s: str):
        s = s.lower()
        s = s.translate(str.maketrans("", "", string.punctuation))
        s = s.translate(str.maketrans("", "", ":><؟!.،,?..,?!%;:-()[]{}$@#^&*"))
        s = re.sub(r"\s+", " ", s.strip())
        return len(s.split()) > self.threshold, s

    def is_persian_text(self, text):
        persian_chars = sum(1 for char in text if '\u0600' <= char <= '\u06FF')
        total_chars = len(text)
        percentage = (persian_chars / total_chars) * 100 if total_chars > 0 else 0
        return percentage > self.char_threshold

    def token_ratio_quality_assessment(self, text, filter_th=3):
        tokens = len(self.bert_tokenizer.tokenize(text))
        text_len = len(text)
        if text_len / tokens >= filter_th:
            return True
        else:
            return False

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
        short_line_count = sum(1 for line in lines if len(line.split()) < self.line_threshold)
        total_line_count = len(lines)
        portion_short_lines = (short_line_count / total_line_count) * 100
        return portion_short_lines < self.min_threshold

    def check_count_numbers_line(self, line):
        num_punct_count = sum(
            1 for char in line if char.isdigit() or char in '=+؛٪:><؟!.،,?!%;:¥-()[]{}$@#^&*۰۱۲۳۴۵۶۷۸۹"\'')
        total_chars = len(line)
        if num_punct_count / total_chars >= self.number_threshold:
            return ""
        else:
            return line

    def preprocess_line(self, text: str, source: str):
        text = re.sub(r'\.\s([a-zA-Z])', r'.\1', text)
        text = re.sub(r'\b[A-Z]+\b', '', text)
        text = re.sub(r'<[^>]+>', '', text)  # removing html tags
        text = wierd_pattern.sub(r'', text)  # Deleting unicodes
        if 'socialMedia' in source:
            re.sub(r"https://t\.me/[\w/]+", '', text)
            re.sub(r"eitta\.me/[\w/]+", '', text)
        text = text.translate(str.maketrans("", "", "‎‏‪‫ ‭‮"))  # Deleting PDF special characters
        text = self.normalizer.normalize(text)
        text = re.sub(r"(.)\1{2,}", r"\1\1", text)  # Deleting repeated chars
        text = text.replace('ه . ش', 'ه.ش').replace('ه . ق', 'ه.ق')  # ه.ش و ه.ق
        if 'paper' in source:
            text = self.check_count_numbers_line(text)
        sents = [sen for sen in self.tokenizer.sentence_tokenize(text)]
        list_of_sentences = [[str(token) for token in self.custom_tokenize(sen)] for sen in sents]
        tokens = [item for sublist in list_of_sentences for item in sublist]
        text = ' '.join(tokens)
        return text.strip()

    def preprocess_document(self, text: str, source: str):
        text = re.sub(r'\n\s*\t*\n*', '\n', text)
        text = re.sub(r'<style.*?</style>', '', text, flags=re.DOTALL)  # delete css tags
        if 'madlad' in source:
            text = text.replace('/n', '\n').replace('n\\', '\n').replace('\\n', '\n')
            text = re.sub(r'\\+n', r'\n', text)
            text = re.sub(r'\\+t', r'\t', text)
        lines = text.splitlines()
        lines = ([self.preprocess_line(text_line, source) for text_line in lines])
        if 'baznashr' in source:
            delete_list = ['انتهای پیام', 'نظرات کاربران', 'به این مطلب امتیاز دهید', 'تبادل نظر کنید', 'بیشتر بخوانید',
                           'اعتمادآنلاین', 'پایان پیام ', 'گفتگو با شبکه تلویزیونی سی بی اس آمریکا']
            while len(lines) > 0 and ((len(lines[-1]) < 40 and any(ext in lines[-1] for ext in delete_list)) or
                                      lines[-1].startswith("http") or len(lines[-1]) < 25):
                lines.pop()
            while len(lines) > 0 and ((len(lines[0]) < 40 and any(ext in lines[0] for ext in delete_list)) or
                                      lines[0].startswith("http") or len(lines[0]) < 25):
                lines.pop(0)
            seen = set()
            lines = [x for x in lines if not (x in seen or seen.add(x))]
        if 'socialMedia' in source:
            for i, line in enumerate(lines[::-1]):
                words = line.strip().split(" ")
                while len(words) > 0 and ("#" in words[-1] or "@" in words[-1] or len(words[-1]) <= 1):
                    words.pop()
                if len(words) > 0:
                    lines[len(lines) - i - 1] = " ".join(words)
                    break
                else:
                    lines[len(lines) - i - 1] = ''
        text = '\n'.join(lines)
        text = re.sub(r'\n\s*\t*\n*', '\n', text)
        return text.strip()

    def write_json(self, json_data, f):
        if self.filtering:
            text = json_data['text']
            is_clean_text, s = self.get_features(text)
            if (is_clean_text and self.is_persian_text(s) and self.check_short_lines(text)
                    and self.most_repeated_word_over_threshold(text)):
                if not self.token_ratio_quality or self.token_ratio_quality_assessment(text):
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
                            if isinstance(json_data['text'], str):
                                preprocessed_text = self.preprocess_document(json_data['text'], source)
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
                            json_data['text'] = self.preprocess_line(json_data['text'], source)
                            json_data['source'] = source
                            self.write_json(json_data, f)
                    except json.decoder.JSONDecodeError:
                        print("Error in reading file: ", file_path)
                elif file_type == '.csv':
                    try:
                        csv_reader = csv.reader(fh)
                        columns = next(csv_reader)
                        for i, row in enumerate(csv_reader):
                            json_data = {}
                            for index in range(len(columns)):
                                json_data[columns[index]] = row[index]
                            json_data['id'] = f"{source}-{file_name}-{i}"
                            json_data['source'] = source
                            json_data['text'] = self.preprocess_line(json_data['text'], source)
                            self.write_json(json_data, f)
                    except Exception as e:
                        print("Error in reading file: ", file_path, str(e))
                elif file_type == '.parquet':
                    try:
                        df = pd.read_parquet(file_path)
                        for i, row in df.iterrows():
                            json_data = {}
                            for column in df.columns:
                                json_data[column] = row[column]
                            json_data['id'] = f"{source}-{file_name}-{i}"
                            json_data['source'] = source
                            json_data['text'] = self.preprocess_line(json_data['text'], source)
                            self.write_json(json_data, f)
                    except Exception as e:
                        print("Error in reading file: ", file_path, str(e))
        #        print("finished : ", file_path)
        return self.normalized_folder

    def normalize_files(self, all_files: list[str]):
        n_proc = cpu_count() - 1
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
        self.log_path = f'./result/logs/{sub_folder_name}.txt'
        self.count_files(sub_folder_name)
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
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(f"Number of words after filtering: {count_words_filtered}\n")
            f.write(f"Number of rows after filtering: : {self.number_of_filtered_rows}\n")
            f.write(f"Normalizing Time: {(time.time() - start_time):.3f} s\n---------------------------\n")

    def count_files(self, sub_folder_name: str):
        data_dir = self.data_path + sub_folder_name
        all_files = get_all_files(data_dir)
        log_name = sub_folder_name
        self.log_path = f'./result/logs/{log_name}.txt'
        count_words = 0  # total number of words
        number_of_rows = 0  # total number of rows
        if not os.path.exists('./result/logs'):
            os.makedirs('./result/logs')
        with open(self.log_path, 'w', encoding='utf-8') as f:
            f.write(f"Total files: {len(all_files)}\n")
        for file_path in tqdm(all_files, desc="counting"):
            file_type = os.path.splitext(file_path)[-1]
            with open(file_path, 'r', encoding='utf-8') as fh:
                if file_type == '.jsonl':
                    for i, line in enumerate(fh):
                        try:
                            json_data = json.loads(line)
                            number_of_rows += 1
                            if isinstance(json_data['text'], str):
                                count_words += len(json_data['text'].split())
                        except json.decoder.JSONDecodeError:
                            print("Error in reading file: ", file_path)
                elif file_type == '.json':
                    try:
                        json_datas = json.load(fh)
                        for i, json_data in enumerate(json_datas):
                            number_of_rows += 1
                            count_words += len(json_data['text'].split())
                    except json.decoder.JSONDecodeError:
                        print("Error in reading file: ", file_path)
                elif file_type == '.csv':
                    try:
                        csv_reader = csv.reader(fh)
                        columns = next(csv_reader)
                        for i, row in enumerate(csv_reader):
                            json_data = {}
                            for index in range(len(columns)):
                                json_data[columns[index]] = row[index]
                            number_of_rows += 1
                            count_words += len(json_data['text'].split())
                    except Exception as e:
                        print("Error in reading file: ", file_path, str(e))
                elif file_type == '.parquet':
                    try:
                        df = pd.read_parquet(file_path)
                        for i, row in df.iterrows():
                            number_of_rows += 1
                            count_words += len(row['text'].split())
                    except Exception as e:
                        print("Error in reading file: ", file_path, str(e))
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(f"Number of words before filtering: {count_words}\n")
            f.write(f"Number of rows before filtering: : {number_of_rows}\n")
