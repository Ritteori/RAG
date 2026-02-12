import os
from pathlib import Path
import re

files_pathes = []
for root, dirs, files in os.walk(r'data'):
    if len(files) == 0:
        continue
    for file in files:
        files_pathes.append(Path(root)/file)

questions = []
questions_pattern = re.compile(r"\b\d+\.\s*[^?]*\?")
for path in files_pathes:
    with open(path, 'r', encoding="utf-8") as f:
        for line in f:
            match = questions_pattern.search(line)
            if match and len(match.group()) > 10:
                questions.append(match.group().split('.',1)[1].strip())

if os.path.exists('questions.txt'):
    os.remove('questions.txt')
with open('questions.txt','x',encoding='utf-8') as f:
    for question in questions:
        f.write(f'{question}\n')