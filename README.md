# HodHod_Preprocess

![GitHub repo size](https://img.shields.io/github/repo-size/UT-NLP-LAB/HodHod_Preprocess)
![GitHub contributors](https://img.shields.io/github/contributors/UT-NLP-LAB/HodHod_Preprocess)
![GitHub stars](https://img.shields.io/github/stars/UT-NLP-LAB/HodHod_Preprocess?style=social)
![GitHub forks](https://img.shields.io/github/forks/UT-NLP-LAB/HodHod_Preprocess?style=social)

🥇 Welcome to HodHod_Preprocess! This repository offers scripts designed for end-to-end pre-processing of the HodHod
dataset. HodHod is the first large-scale, cleaned Persian dataset available in text format. You can find the
dataset [here](https://huggingface.co/UT-NLP-LAP).

## Key Features:

* **Normalization and Filtering:** Cleans and filters text data for improved quality.
* **Deduplication:** Removes redundant documents to create a unique dataset.
* **Ease of Use:** Provides clear instructions and scripts for pre-processing.

## Requirements

* Python 3.11+
* All packages listed in `requirements.txt`

## Installation

1. Create a virtual environment:
   ```bash
   virtualenv <env_name>
   source <env_name>/bin/activate
   ```
   
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Preprocessing Steps

The pre-processing involves the following steps:

1. Normalization and Filtering:

    This step cleans and filters the text data to enhance its quality.
    **Script:**
       ```python
       from preprocess.preprocess_document import Preprocessor
       
       preprocessor = Preprocessor()
       preprocessor.preprocess_files('crawl', filtering=True)
       ```
    * Replace ```'crawl'``` with the subdirectory containing your data.
    * Set ```filtering=True``` to remove low-quality documents.
    * The normalized and filtered documents will be stored in  ```./result/normalized``` directory.

2. Deduplication of redundant documents
    This step removes redundant documents to create a unique dataset.
    **Script:**
        ```python
         from preprocess.deduplication import Deduplication
         
         deduplication = Deduplication()
         deduplication.preprocess_files('crawl')
        ```   
    * Replace ```'crawl'``` with the subdirectory of your data folder.
    * The deduplicated data is will be saved in the ```./result/deduplicated``` directory.
    * logs for each step will be available in ```./result/logs```.

### Directory Structure

    .
    ├── data
    │   ├── book  # Example data folder
    │   │   ├── History  # Example data subfolder
    │   │   │   └── A Modest Proposal.json  # Example data file
    │   │   └── ...
    │   ├── social_media  # Example data folder
    │   └── ...
    ├── preprocess  # Code directory
    └── ...

**Data Format** : Each data file should be a text file, JSON file, or JSONL file containing a "text" field.

### Additional notes

The deduplication contains four steps:

1. MinHash Generation
2. Duplicate Pairs Generation (Stored in ```./result/lsh```)
3. Duplicate Graph Construction & Search for Connected Components
4. Delete the redundant documents
   More information about deduplication can be
   found [here.](https://github.com/Cerebras/modelzoo/tree/main/modelzoo/transformers/data_processing/slimpajama)

## License

**GNU Lesser General Public License v2.1**

Primarily used for software libraries, the GNU LGPL requires that derived works be licensed under the same license, but
works that only link to it do not fall under this restriction. There are two commonly used versions of the GNU LGPL.

See [LICENSE](https://github.com/UT-NLP-LAB/HodHod_Preprocess/blob/main/LICENSE)

## About ️

* Developed by the [Tehran university NLP lab](https://ece.ut.ac.ir/lab/nlp)

Contributors:

* [Dr. Heshaam Faili](https://www.linkedin.com/in/heshaam-faili-7b9b2468/?originalSubdomain=ir)
* [Hamed Khademi Khaledi](https://www.linkedin.com/in/hamed-khademi/)
* [Hamed Baghbani](https://www.linkedin.com/in/hamedbaghbani/)
* [Mostafa Amiri](https://www.linkedin.com/in/mostafaamirii/)