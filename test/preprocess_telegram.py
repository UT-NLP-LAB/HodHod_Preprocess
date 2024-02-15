import json
import time
from tqdm import tqdm

from preprocess.deduplication import Deduplication
from preprocess.preprocess_document import Preprocessor

start_time = time.time()
preprocessor = Preprocessor()
deduplication = Deduplication()
if __name__ == '__main__':
    # preprocessor.preprocess_files('../data/telegram')
    deduplication.preprocess_files('../result/normalized/telegram')
    print("--- %s seconds ---" % (time.time() - start_time))
