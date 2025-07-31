import json
import os
import pickle
from typing import List, Dict
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class RAGDataProcessor:

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None
        self.index = None

    def load_jsonl_data(self, file_path: str, text_field: str = "text") -> List[Dict]:
        documents = []

        print("JSONL dosyası yükleniyor...")
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(tqdm(file, desc="Satırlar işleniyor")):
                try:
                    if not line.strip():
                        continue
                    data = json.loads(line.strip())
                    text = None
                    possible_fields = [
                        text_field, 'text', 'content', 'message', 'document',
                        'body', 'description', 'prompt', 'output', 'response',
                        'question', 'answer', 'input', 'instruction'
                    ]
                    for field in possible_fields:
                        if field in data and data[field]:
                            if isinstance(data[field], str):
                                text = data[field].strip()
                                break
                            elif isinstance(data[field], dict):
                                for subfield in ['text', 'content', 'value']:
                                    if subfield in data[field]:
                                        text = str(data[field][subfield]).strip()
                                        break
                                if text:
                                    break
                    if not text:
                        text_parts = []
                        for key, value in data.items():
                            if isinstance(value, str) and len(value.strip()) > 10:
                                text_parts.append(value.strip())
                        if text_parts:
                            text = " ".join(text_parts[:3])
                    if text and len(text.strip()) > 10:
                        documents.append({
                            'id': f"jsonl_{line_num}",
                            'text': text.strip(),
                            'metadata': data
                        })
                except Exception as e:
                    print(f"Satır {line_num + 1}'de hata: {e}")
                    continue

        print(f"Toplam {len(documents)} JSONL dökümanı yüklendi.")
        self.documents.extend(documents)
        return documents

    def chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 100) -> List[str]:
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            if end == len(words):
                break
            start += chunk_size - overlap
        return chunks

    def load_txt_folder(self, folder_path: str, encoding: str = 'utf-8',
                        chunk_size: int = 512, overlap: int = 100) -> List[Dict]:
        txt_documents = []
        print("TXT dosyaları chunk’lanarak yükleniyor...")

        for i, filename in enumerate(tqdm(os.listdir(folder_path), desc="Dosyalar işleniyor")):
            if filename.endswith(".txt"):
                try:
                    file_path = os.path.join(folder_path, filename)
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read().strip()
                        if len(content) > 10:
                            chunks = self.chunk_text(content, chunk_size, overlap)
                            for j, chunk in enumerate(chunks):
                                txt_documents.append({
                                    'id': f"txt_{i}_{j}",
                                    'text': chunk,
                                    'metadata': {
                                        'filename': filename,
                                        'chunk_index': j,
                                        'total_chunks': len(chunks)
                                    }
                                })
                except Exception as e:
                    print(f"{filename} dosyasında hata: {e}")
                    continue

        print(f"Toplam {len(txt_documents)} chunk’lanmış TXT dökümanı yüklendi.")
        self.documents.extend(txt_documents)
        return txt_documents

    def create_embeddings(self, batch_size: int = 32) -> np.ndarray:
        if not self.documents:
            raise ValueError("Önce dökümanları yükleyin!")

        texts = [doc['text'] for doc in self.documents]

        print("Embedding’ler oluşturuluyor...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        self.embeddings = embeddings
        return embeddings

    def build_faiss_index(self, index_type: str = "flat") -> faiss.Index:
        if self.embeddings is None:
            raise ValueError("Önce embedding’leri oluşturun!")

        d = self.embeddings.shape[1]

        if index_type == "flat":
            index = faiss.IndexFlatIP(d)
        elif index_type == "ivf":
            nlist = min(100, len(self.documents) // 10)
            quantizer = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFFlat(quantizer, d, nlist)
            index.train(self.embeddings)
        else:
            raise ValueError("Desteklenen index tipleri: 'flat', 'ivf'")

        print("FAISS indexi oluşturuluyor...")
        index.add(self.embeddings)

        self.index = index
        return index

    def save_index(self, index_path: str, documents_path: str):
        if self.index is None:
            raise ValueError("Önce indexi oluşturun!")

        faiss.write_index(self.index, index_path)

        with open(documents_path, 'wb') as f:
            pickle.dump(self.documents, f)

        print(f"Index kaydedildi: {index_path}")
        print(f"Dökümanlar kaydedildi: {documents_path}")

    def load_index(self, index_path: str, documents_path: str):
        self.index = faiss.read_index(index_path)
        with open(documents_path, 'rb') as f:
            self.documents = pickle.load(f)
        print(f"Index yüklendi: {index_path}")
        print(f"Dökümanlar yüklendi: {documents_path}")


if __name__ == "__main__":
    processor = RAGDataProcessor()

    # Verileri yükle
    processor.load_jsonl_data("geodi_fine_tuning_dataset.jsonl")
    processor.load_txt_folder("RAG_database", chunk_size=512, overlap=100)

    # Embedding ve index oluştur
    processor.create_embeddings()
    processor.build_faiss_index()
    processor.save_index("faiss_index.bin", "documents.pkl")
