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

def build_chunks(data_texts=data_texts, overlap=200, chunk_size=750):
    """
    Split text documents into overlapping chunks with metadata.

    Iterates over all loaded .txt files and splits each document into
    character-based chunks with overlap for context preservation.

    Args:
        overlap (int): Number of overlapping characters between chunks.
        chunk_size (int): Maximum size of each chunk in characters.

    Returns:
        dict: Mapping chunk_id -> chunk metadata (text, source file,
              category, chunk index, char_start, char_end).
    """

    chunked_texts = {}
    reg = r'(?=\b\d+\.\s+)'

    for key, value in data_texts.items():
        questions = re.split(reg,value)
        result = [item for item in questions if item]

        path = key.split('RAG/')
        name = path[1]
        topic = name.split('/', 2)[1]

        idx = 0
        for question_text in result:
            
            question_length = len(question_text)

            if question_length < 50:
                continue
            elif question_length > 2000:
                start = 0

                while start < question_length:

                    raw_chunk = question_text[start:start + chunk_size]
                    chunk_text = raw_chunk.replace('\n', ' ')

                    if len(raw_chunk) < 50:
                        break
                    
                    chunk = {
                        "text": chunk_text,
                        "path": key,
                        "category": topic,
                        "chunk_index": idx,
                        "source_file": key.split('RAG')[1]
                    }

                    chunked_texts[f"{name}::{idx}"] = chunk
                    start += chunk_size - overlap
                    idx += 1
            else:
                chunk = {
                    "text": question_text,
                    "path": key,
                    "category": topic,
                    "chunk_index": idx,
                    "source_file": key.split('RAG')[1]
                }

                chunked_texts[f"{name}::{idx}"] = chunk
                idx +=1 

    return chunked_texts

if __name__ == "__main__":
    chunked_texts = build_chunks()