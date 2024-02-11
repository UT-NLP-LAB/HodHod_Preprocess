import json
import time
from tqdm import tqdm

from preprocess.preprocess_document import Preprocessor

start_time = time.time()
preprocessor = Preprocessor()

if __name__ == '__main__':
    preprocessor.preprocess_files('../data/telegram')
    print("--- %s seconds ---" % (time.time() - start_time))
