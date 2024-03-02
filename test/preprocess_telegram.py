import time

from preprocess.deduplication import Deduplication
from preprocess.preprocess_document import Preprocessor

start_time = time.time()
preprocessor = Preprocessor()
deduplication = Deduplication()
if __name__ == '__main__':
    preprocessor.preprocess_files('crawl', filtering=True)
    deduplication.preprocess_files('crawl')
    print("--- %s seconds ---" % (time.time() - start_time))
