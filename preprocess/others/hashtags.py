from collections import defaultdict, Counter

from preprocess.utils import get_all_files
from tqdm import tqdm
import os
import json
import re

hashtag_pattern = r'#(\w+)'
hashtag_counts = {}


def get_hashtags(sub_folder_name: str):
    if not os.path.exists('../../result/hashtags'):
        os.makedirs('../../result/hashtags')
    data_dir = '../../data/' + sub_folder_name
    all_files = get_all_files(data_dir)
    hashtag_counts = {}
    for file_path in tqdm(all_files, desc="counting"):
        file_type = os.path.splitext(file_path)[-1]
        with open(file_path, 'r', encoding='utf-8') as fh:
            if file_type == '.json':
                try:
                    json_datas = json.load(fh)
                    for i, json_data in enumerate(json_datas):
                        hashtags = re.findall(hashtag_pattern, json_data['text'])
                        for hashtag in hashtags:
                            hashtag_counts[hashtag] = hashtag_counts.get(hashtag, 0) + 1
                except json.decoder.JSONDecodeError:
                    print("Error in reading file: ", file_path)

    hashtag_counts = sorted(hashtag_counts.items(), key=lambda x: -x[1])
    with open(f'../../result/hashtags/hashtags_{sub_folder_name}.txt', 'w', encoding='utf-8') as f:
        for hashtag, count in hashtag_counts:
            f.write(f"{hashtag}: {count}\n")


def extract_keywords(tweet, stopwords):
    words = tweet.split()
    keywords = [word for word in words if word.lower() not in stopwords]
    return keywords


def get_keywords(sub_folder_name: str):
    with open(f'../../result/hashtags/hashtags_{sub_folder_name}.txt', 'r', encoding='utf-8') as file:
        lines = file.readlines()
    with open(f'./stopwords.txt', 'r', encoding='utf-8') as file:
        stop_words = [elem.strip() for elem in file.readlines()]
    hashtags = ['#' + line.strip().split(':')[0] for line in lines]
    data_dir = '../../data/' + sub_folder_name
    all_files = get_all_files(data_dir)
    hashtags_keywords = defaultdict(list)
    for file_path in tqdm(all_files, desc="counting_keywords"):
        file_type = os.path.splitext(file_path)[-1]
        with open(file_path, 'r', encoding='utf-8') as fh:
            if file_type == '.json':
                try:
                    json_datas = json.load(fh)
                    for i, json_data in enumerate(json_datas):
                        tweet = json_data['text']
                        for hashtag in hashtags:
                            if hashtag in tweet:
                                keywords = extract_keywords(tweet, stop_words)
                                hashtags_keywords[hashtag].extend(keywords)
                except json.decoder.JSONDecodeError:
                    print("Error in reading file: ", file_path)

    result_json = {}
    for hashtag, keywords in hashtags_keywords.items():
        keyword_counts = Counter(keywords)
        top_keywords = keyword_counts.most_common(200)
        result_json[hashtag] = [{"keyword": key, "count": count} for key, count in top_keywords if key != hashtag]

    with open(f'../../result/hashtags/hashtags_dict_{sub_folder_name}.json', 'w', encoding='utf-8') as output_file:
        json.dump(result_json, output_file, ensure_ascii=False, indent=4)


def remove_irrelevant_hashtags(tweet, hashtag_keywords):
    words = tweet.split()
    hashtags = []
    while len(words) > 0 and words[-1].startswith("#"):
        hashtags.append(words.pop())
    # Check if the last word is a hashtag
    for hashtag in hashtags:
        keyword_count = 0
        if hashtag in hashtag_keywords:
            keywords = [elem['keyword'] for elem in hashtag_keywords[hashtag]]
            for keyword in keywords:
                if keyword in words:
                    keyword_count += 1

            if keyword_count >= 3:
                words.append(hashtag)
            else:
                print(f"Removing hashtag: {hashtag} from tweet: {tweet}")
    updated_tweet = " ".join(words)
    return updated_tweet


def remove_hashtags(sub_folder_name: str):
    data_dir = '../../data/' + sub_folder_name
    all_files = get_all_files(data_dir)
    with open(f'../../result/hashtags/hashtags_dict_{sub_folder_name}.json', 'r', encoding='utf-8') as output_file:
        hashtag_keywords = json.load(output_file)
    with open(f'../../data/{sub_folder_name}/{sub_folder_name}.jsonl', 'w', encoding='utf-8') as f:
        for file_path in tqdm(all_files, desc="counting"):
            file_type = os.path.splitext(file_path)[-1]
            with open(file_path, 'r', encoding='utf-8') as fh:
                if file_type == '.json':
                    try:
                        json_datas = json.load(fh)
                        for i, json_data in enumerate(json_datas):
                            json_data['text'] = remove_irrelevant_hashtags(json_data['text'], hashtag_keywords)
                            if len(json_data['text'].split() > 0):
                                json.dump(json_data, f, ensure_ascii=False)
                                f.write('\n')
                    except json.decoder.JSONDecodeError:
                        print("Error in reading file: ", file_path)


if __name__ == '__main__':
    # get_hashtags('telegram')
    # get_keywords('telegram')
    remove_hashtags('telegram')
