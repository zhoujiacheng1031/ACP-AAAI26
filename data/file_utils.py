# -*- coding: utf-8 -*-
"""
@purpose: file utils
"""

import json
import jsonlines
import os

def save_json_lines(data, file_path):
    """
    Save a list of JSON objects to a file in JSON Lines format.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    try:
        with jsonlines.open(file_path, 'w') as file:
            file.write_all(data)
    except Exception as e:
        raise Exception(f"Error saving JSON lines to {file_path}: {e}")

def save_json(data, file_path):
    """
    Save data to a file in JSON format.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
    except Exception as e:
        raise Exception(f"Error saving JSON to {file_path}: {e}")

def load_json(file_path):
    """
    Load data from a JSON file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except Exception as e:
        raise Exception(f"Error loading JSON from {file_path}: {e}")
