import json
import time
from tqdm import tqdm

from preprocess.preprocess_document import Preprocessor

start_time = time.time()
preprocessor = Preprocessor()

# Open the JSONL file for reading
with open('../data/output.txt', 'w', encoding='utf-8') as file1:
    with open('../data/wikipedia.jsonl', 'r', encoding='utf-8') as file:
        # Iterate through each line in the file
        for line in tqdm(file):
            # Decode the JSON object from the line
            data = json.loads(line)
            file1.write(preprocessor.preprocess_document(data["text"]))
print("--- %s seconds ---" % (time.time() - start_time))
