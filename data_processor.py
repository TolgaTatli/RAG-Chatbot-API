import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict, Tuple
import pickle
import os
from tqdm import tqdm

class RAGDataProcessor:

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Args:
            model_name: Sentence transformer model adı
        """
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
                    if not line.strip():  # Boş satırları atla
                        continue

                    data = json.loads(line.strip())

                    # Farklı olası text alanlarını dene
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
                                # Nested dictionary durumu
                                for subfield in ['text', 'content', 'value']:
                                    if subfield in data[field]:
                                        text = str(data[field][subfield]).strip()
                                        break
                                if text:
                                    break

                    # Eğer hiç text bulunamadıysa, tüm string değerlerini birleştir
                    if not text:
                        text_parts = []
                        for key, value in data.items():
                            if isinstance(value, str) and len(value.strip()) > 10:
                                text_parts.append(value.strip())
                        if text_parts:
                            text = " ".join(text_parts[:3])  # İlk 3 anlamlı string'i al

                    if text and len(text.strip()) > 10:  # En az 10 karakter olsun
                        documents.append({
                            'id': line_num,
                            'text': text.strip(),
                            'metadata': data
                        })

                    # İlerleme raporu
                    if line_num % 1000 == 0 and line_num > 0:
                        print(f"İşlenen satır: {line_num}, Bulunan döküman: {len(documents)}")

                except json.JSONDecodeError as e:
                    print(f"Satır {line_num + 1}'de JSON hatası: {e}")
                    continue
                except Exception as e:
                    print(f"Satır {line_num + 1}'de hata: {e}")
                    continue

        print(f"Toplam {len(documents)} döküman yüklendi.")

        # Eğer hiç döküman bulunamadıysa, dosyanın yapısını analiz et
        if len(documents) == 0:
            print("\nHiç döküman bulunamadı. Dosya yapısını analiz ediliyor...")
            self._analyze_file_structure(file_path)

        self.documents = documents
        return documents

    def _analyze_file_structure(self, file_path: str, sample_lines: int = 5):
        """Dosya yapısını analiz et ve kullanıcıya bilgi ver"""
        print(f"\nİlk {sample_lines} satırın yapısı:")
        with open(file_path, 'r', encoding='utf-8') as file:
            for i, line in enumerate(file):
                if i >= sample_lines:
                    break
                try:
                    data = json.loads(line.strip())
                    print(f"\nSatır {i+1} anahtarları: {list(data.keys())}")

                    # Her anahtarın değer tipini göster
                    for key, value in data.items():
                        value_type = type(value).__name__
                        if isinstance(value, str):
                            preview = value[:50] + "..." if len(value) > 50 else value
                            print(f"  {key}: {value_type} = '{preview}'")
                        else:
                            print(f"  {key}: {value_type}")
                except:
                    print(f"Satır {i+1}: JSON parse edilemedi")

    def create_embeddings(self, batch_size: int = 32) -> np.ndarray:
        """
        Dökümanlar için embedding'ler oluştur

        Args:
            batch_size: Batch boyutu

        Returns:
            Embedding matrisi
        """
        if not self.documents:
            raise ValueError("Önce dökümanları yükleyin!")

        texts = [doc['text'] for doc in self.documents]

        print("Embedding'ler oluşturuluyor...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        self.embeddings = embeddings
        return embeddings

    def build_faiss_index(self, index_type: str = "flat") -> faiss.Index:
        """
        FAISS indexi oluştur

        Args:
            index_type: Index tipi ("flat" veya "ivf")

        Returns:
            FAISS index
        """
        if self.embeddings is None:
            raise ValueError("Önce embedding'leri oluşturun!")

        d = self.embeddings.shape[1]  # Embedding boyutu

        if index_type == "flat":
            index = faiss.IndexFlatIP(d)  # Inner Product (cosine similarity için)
        elif index_type == "ivf":
            nlist = min(100, len(self.documents) // 10)  # Cluster sayısı
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
        """Index ve dökümanları kaydet"""
        if self.index is None:
            raise ValueError("Önce indexi oluşturun!")

        # FAISS indexi kaydet
        faiss.write_index(self.index, index_path)

        # Dökümanları kaydet
        with open(documents_path, 'wb') as f:
            pickle.dump(self.documents, f)

        print(f"Index kaydedildi: {index_path}")
        print(f"Dökümanlar kaydedildi: {documents_path}")

    def load_index(self, index_path: str, documents_path: str):
        """Kaydedilmiş index ve dökümanları yükle"""
        # FAISS indexi yükle
        self.index = faiss.read_index(index_path)

        # Dökümanları yükle
        with open(documents_path, 'rb') as f:
            self.documents = pickle.load(f)

        print(f"Index yüklendi: {index_path}")
        print(f"Dökümanlar yüklendi: {documents_path}")

if __name__ == "__main__":
    # Örnek kullanım
    processor = RAGDataProcessor()

    # JSONL dosyasını yükle
    documents = processor.load_jsonl_data("geodi_fine_tuning_dataset.jsonl")

    # Embedding'leri oluştur
    embeddings = processor.create_embeddings()

    # FAISS indexi oluştur
    index = processor.build_faiss_index()

    # Kaydet
    processor.save_index("faiss_index.bin", "documents.pkl")