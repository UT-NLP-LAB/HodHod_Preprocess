import gc
import json
import math
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from datasketch import MinHash
import re
import string
from nltk import ngrams


class Deduplication:
    def __init__(self):
        pass

    def get_features(self, s: str):
        # lower cased
        s = s.lower()
        # remove punctuation
        s = s.translate(str.maketrans("", "", string.punctuation))
        s = s.replace(":><؟!.،,?", "")
        # remove consecutive spaces, newlines, tabs in the middle and in the beginning / end
        s = re.sub(r"\s+", " ", s.strip())
        return map(lambda x: "".join(x), ngrams(s, 6))

    def preprocess_file(self, file_path: str):
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        source = os.path.basename(os.path.dirname(file_path))
        file_type = os.path.splitext(file_path)[-1]
        hash_folder = f'../result/minhash_nfc/{source}/'
        res_path = f'{hash_folder}{file_name}.jsonl'
        if not os.path.exists(hash_folder):
            os.makedirs(hash_folder)
        with open(file_path, 'r', encoding='utf-8') as fh:
            with open(res_path, 'w', encoding='utf-8') as f:
                if file_type == '.jsonl':
                    for line in tqdm(fh):
                        json_data = json.loads(line)
                        map_text = self.get_features(json_data['text'])
                        mini_hash = MinHash(num_perm=128)
                        hash_text = [mini_hash.update(x.encode('utf8')) for x in map_text]
                        json_data['hash'] = mini_hash
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
