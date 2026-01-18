import os
import re

base_dir = os.path.join(os.getcwd(), "data")

data_texts = {}
trash_symbols = "↑←➢⋯⇨├└⚡️❖⚖️⚠️`&éЙ✅❌❗️�"
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith(".txt"):
            path = os.path.join(root, file)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read().lower()
                text = re.sub(f"[{re.escape(trash_symbols)}]", "", text)
                data_texts[path] = text

def build_chunks(overlap=200, chunk_size=750):
    chunked_texts = {}

    for key, value in data_texts.items():

        start = 0
        index = 1

        while start < len(value):

            path = key.split('RAG/')
            name = path[1]
            topic = name.split('/', 2)[1]

            raw_chunk = value[start:start + chunk_size]
            chunk_text = raw_chunk.replace('\n', ' ')
            char_end = start + len(raw_chunk)

            if len(chunk_text) < 50:
                break

            chunk = {
                "text": chunk_text,
                "path": key,
                "category": topic,
                "chunk_index": index,
                "source_file": key.split('RAG')[1],
                "char_start": start,
                "char_end": char_end
            }

            chunked_texts[f"{name}::{index}"] = chunk

            start += chunk_size - overlap
            index += 1

    return chunked_texts

if __name__ == "__main__":
    chunked_texts = build_chunks()