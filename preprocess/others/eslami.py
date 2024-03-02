import os
from bs4 import BeautifulSoup
import json

def extract_text_from_html(html_file):
    with open(html_file, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')
        # Extract text from HTML
        text = soup.get_text(separator='\n', strip=True)
    return text


def create_jsonl(directory_path):
    html_files = [file for file in os.listdir(directory_path) if file.endswith('.htm')]
    for html_file in html_files:
        base_name = os.path.basename(html_file)
        text = extract_text_from_html(os.path.join(directory_path, html_file))
        jsonl_file = os.path.join(directory_path, base_name.replace('.htm', '.jsonl'))
        with open(jsonl_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    directory_path = '../../data/books/eslami'
    create_jsonl(directory_path)
