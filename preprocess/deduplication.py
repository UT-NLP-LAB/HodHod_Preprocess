import json
import os
import queue
import re
import string
import time
from collections import defaultdict
from glob import glob
from multiprocessing import Queue, Process

import networkit as nk
from datasketch import MinHash
from datasketch.lean_minhash import LeanMinHash
from more_itertools import divide
from nltk import ngrams
from tqdm import tqdm

from .utils import get_all_files


def _h_bytes(hs):
    return bytes(hs.byteswap().data)


def construct_graph(set_of_duplicate_pairs):
    graph = nk.Graph()
    mapper = {}
    for pair in set_of_duplicate_pairs:
        node1_name, node2_name = pair
        if node1_name not in mapper:
            mapper[node1_name] = graph.addNode()
        if node2_name not in mapper:
            mapper[node2_name] = graph.addNode()
        graph.addEdge(mapper[node1_name], mapper[node2_name])
    return graph, mapper


def find_connected_components(graph):
    cc = nk.components.ConnectedComponents(graph)
    cc.run()
    return cc.getComponents(), cc.numberOfComponents()


def get_features(s: str, width: int):
    # lower cased
    s = s.lower()
    re.sub(r"[\d٠١٢٣٤٥٦٧٨٩]+", " ", s)
    s = s.replace('جمعه', '').replace('شنبه', '')
    persian_numbers = ["یک", "دو", "سه", "چهار", "پنج", "شش", "هفت", "هشت", "نه", "ده"]
    pattern = re.compile(f"({'|'.join(persian_numbers)})")
    s = pattern.sub("", s)
    # remove punctuation
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = s.translate(str.maketrans("", "", "/:><؟!.،,?"))
    # remove consecutive spaces, newlines, tabs in the middle and in the beginning / end
    s = re.sub(r"\s+", " ", s.strip())
    return map(lambda x: "".join(x), ngrams(s, width))


class Deduplication:
    def __init__(self):
        self.n_proc = 0
        self.lsh_folder = ""
        self.BAND = 9
        self.doc_queues = [Queue(1000000000) for _ in range(self.BAND)]
        self.lsh_dicts = [defaultdict(list) for _ in range(self.BAND)]
        self.width = 13
        self.range = 13
        self.lsh_out = ""
        self.duplicates = defaultdict()
        self.data_path = "result/normalized/"

    def generate_hash(self, file_paths: list[str]):
        for file_path in tqdm(file_paths, total=len(file_paths), desc='generate_hash'):
            with open(file_path, 'r', encoding='utf-8') as fh:
                for line in fh:
                    json_data = json.loads(line)
                    map_text = get_features(json_data['text'], self.width)
                    mini_hash = MinHash(num_perm=128)
                    [mini_hash.update(x.encode('utf8')) for x in map_text]
                    lean_minhash = LeanMinHash(mini_hash)
                    for i, doc_queue in enumerate(self.doc_queues):
                        h_bytes = _h_bytes(lean_minhash.hashvalues[
                                           i * self.range: min((i + 1) * self.range, len(lean_minhash.hashvalues))])
                        doc_queue.put((f'{file_path}@{json_data["id"]}', h_bytes))
                    del mini_hash
                    del lean_minhash
        for doc_queue in self.doc_queues:
            doc_queue.put(("Done", "Done"))
        # print("PROCESS DONE")

    def lsh(self, doc_queue, lsh_dict, idx):
        i = 0
        done_process = 0
        pbar = tqdm(desc=f'lsh{idx}: ')
        with open(f'{self.lsh_folder}/deduplication{idx}.txt', 'w', encoding='utf-8') as f:
            while True:
                try:
                    key, h_bytes = doc_queue.get(timeout=30)
                    if key == "Done":
                        done_process += 1
                        print(f'done processes for {idx}: {done_process}')
                        continue
                    cand = lsh_dict.get(h_bytes, "None")
                    if cand != "None":
                        f.write(f'{key} :: {cand}\n')
                    else:
                        lsh_dict[h_bytes] = key
                    pbar.update(1)
                except queue.Empty:
                    if done_process == self.n_proc:
                        lsh_dict = {}
                        doc_queue = Queue(10)
                        pbar.close()
                        break
        print(f"process {idx}: Done")
        print(f"Total number of documents {idx}: {i}")

    def generate_pairs(self, all_files: list[str]):
        self.n_proc = 4
        parts = divide(self.n_proc, all_files)
        print(f"resetting to {self.n_proc} for number of processes")
        processes = []

        for process_id in range(self.n_proc):
            p = Process(
                target=self.generate_hash,
                args=(list(parts[process_id]),),
            )
            processes.append(p)
            p.start()
        # [process.join() for process in processes]
        # processes = []
        for process_id in range(self.BAND):
            p = Process(
                target=self.lsh,
                args=(self.doc_queues[process_id], self.lsh_dicts[process_id], process_id,),
            )
            processes.append(p)
            p.start()
        [process.join() for process in processes]

        return self.lsh_dicts

    def generate_connected_components_mp(self, log_file):
        start = time.time()
        files = glob(f"{self.lsh_folder}/*")
        print("Started graph building")
        set_of_duplicate_pairs = set()
        for fp in files:
            with open(fp, "r", encoding='utf-8') as f:
                for line in f:
                    pair = tuple(line.strip().split(" :: "))
                    if pair[0] != pair[1]:
                        set_of_duplicate_pairs.add(pair)
        log_file.write(f"length of the set of duplicates: {len(set_of_duplicate_pairs)}\n")

        # generate a graph using id's as nodes and a pair of ids as an edge
        nk.setNumberOfThreads(60)
        graph, mapper = construct_graph(set_of_duplicate_pairs)
        components, n_components = find_connected_components(graph)
        log_file.write(f"number of connected components: {n_components}, {time.time() - start:.3f}s\n")

        reversed_mapper = {value: key for key, value in mapper.items()}
        log_file.write(f"Graph generated duplicates list!!!: {time.time() - start:.3f}s\n")

        duplicates = defaultdict(set)
        n_duplicate_docs = 0
        for component in components:
            for j in range(1, len(component)):
                doc = reversed_mapper[component[j]]
                file_name, doc_idx = doc.split("@")
                duplicates[file_name].add(str(doc_idx))
                n_duplicate_docs += 1

        log_file.write(f"number of duplicate documents that will be removed:{n_duplicate_docs}\n")
        return duplicates

    def preprocess_files(self, sub_folder_name: str):
        start_time = time.time()
        data_dir = self.data_path + sub_folder_name
        all_files = get_all_files(data_dir)
        self.lsh_folder = f'./result/lsh/{sub_folder_name}'
        if not os.path.exists(self.lsh_folder):
            os.makedirs(self.lsh_folder)
        res_folder = './result/deduplication'
        if not os.path.exists(res_folder):
            os.makedirs(res_folder)
        self.generate_pairs(all_files)
        log_name = data_dir.split(self.data_path)[1]
        log_path = f'./result/logs/{log_name}.txt'
        total_rows = 0
        total_words = 0
        idx = 0
        with open(log_path, 'a', encoding='utf-8') as log_file:
            duplicates = self.generate_connected_components_mp(log_file)
            for file_path in tqdm(all_files, desc='preprocess_files'):
                res_path = f'{res_folder}/{sub_folder_name}/{sub_folder_name}{str(idx)}.jsonl'
                if not os.path.exists(os.path.dirname(res_path)):
                    os.makedirs(os.path.dirname(res_path))
                with open(res_path, 'a', encoding='utf-8') as result_file:
                    with open(file_path, 'r', encoding='utf-8') as fh:
                        for line in fh:
                            json_data = json.loads(line)
                            if json_data['id'] not in duplicates[file_path]:
                                json.dump(json_data, result_file, ensure_ascii=False)
                                total_rows += 1
                                total_words += len(json_data['text'].split())
                                result_file.write('\n')
                                if total_rows % 1000000 == 999999:
                                    idx += 1
            log_file.write(f"Number of words: {total_words}\n")
            log_file.write(f"Filtered rows: {total_rows}\n")
            log_file.write(f"Deduplication Time: {time.time() - start_time:.3f}\n")
