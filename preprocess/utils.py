import json
import os
import pathlib


def get_all_files(data_dir: str):
    all_files = []
    for subdir, dirs, files in os.walk(data_dir):
        for file in files:
            file = os.path.join(subdir, file)
            all_files.append(str(pathlib.PurePosixPath(pathlib.Path(str(file)))))
    return all_files


def get_out_file(file_paths: list[str], dir_path: str):
    for file_path in file_paths:
        file_type = os.path.splitext(file_path)[-1]
        with open(file_path, 'r', encoding='utf-8') as fh:
            if file_type == '.jsonl':
                for line in fh:
                    json_data = json.loads(line)
                    source = os.path.split(json_data['source'])[0]
                    res_folder = f'{dir_path}/{source}'
                if not os.path.exists(res_folder):
                    os.makedirs(res_folder)
                return res_folder
