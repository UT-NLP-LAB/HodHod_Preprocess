import json


# Function to extract text and create a new dictionary
def extract_text(data):
    text = ""
    if 'abstracts' in data and 'fa' in data['abstracts']:
        text += data['abstracts']['fa'] + " "
    if 'bodies' in data and 'fa' in data['bodies']:
        text += data['bodies']['fa']
    return text.strip()


if __name__ == '__main__':
    # Load the JSON file
    with open('../../data/wikies/qomnet/full', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Process each dictionary item and extract text if 'uri' exists
    extracted_data = []
    for item in data:
        if 'uri' in item:
            text = extract_text(item)
            if text != "":
                extracted_data.append({'uri': item['uri'], 'text': text})

    # Write the extracted data to a JSON Lines file
    with open('../../data/wikies/qomnet/qomnet.jsonl', 'w', encoding='utf-8') as f:
        for item in extracted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
