import os
import re
from collections import Counter
from multiprocessing import Pool

import pandas as pd
from tqdm import tqdm

tqdm.pandas()


def imap_unordered_bar(func, args, n_processes=72):
    p = Pool(n_processes)
    res_list = []
    with tqdm(total=len(args)) as pbar:
        for i, res in enumerate(p.imap_unordered(func, args)):
            pbar.update()
            res_list.append(res)
    p.close()
    p.join()
    return res_list


def count_words(text):
    # Use regular expression to a split text into words
    words = re.findall(r'\b\w+\b', text.lower())
    # Count occurrences of each word
    cnt.update(words)


def process_json_paper(f_path):
    df = pd.read_json(f_path, lines=True)
    df['text'].apply(count_words)


def count_words_in_text(text):
    # Split the text into words
    words = re.findall(r'\b\w+\b', text.lower())
    total_oov = 0
    # Iterate through the predefined list of words
    for word in words:
        # Increment the count for the word if it appears in the text
        if word in oov_set:
            total_oov += 1
        # total_oov += word_counter[word]
    return total_oov / len(words)


oov_th = 0.025


def process_json_paper_oov(f_path):
    df = pd.read_json(f_path, lines=True)
    df['oov_ratio'] = df['text'].apply(count_words_in_text)
    len_stat = len(df[df['oov_ratio'] >= oov_th])
    df = df[df['oov_ratio'] <= oov_th]
    df[['text', 'id', 'source']].to_json(f_path.replace('papers', 'papers_filter'), orient='records', lines=True)
    return len_stat


if __name__ == '__main__':
    file_list = []
    start_directory = '../../data/papers'
    cnt = 0
    for root, dirs, files in tqdm(os.walk(start_directory)):
        for file in files:
            path = root + "/" + file
            file_list.append(path)

    cnt = Counter()
    for file in tqdm(file_list):
        process_json_paper(file)

    oov = [key for key, value in cnt.items() if value <= 5]
    oov_set = set(oov)
    res = process_json_paper_oov(file_list[0])
    file_res = imap_unordered_bar(process_json_paper_oov, file_list)
    print(sum(file_res))
