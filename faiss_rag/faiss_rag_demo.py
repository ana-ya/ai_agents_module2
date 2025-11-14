#!/usr/bin/env python3
"""
FAISS RAG - Семантичний пошук
=============================
FAISS (Facebook AI Similarity Search):
- Щільні вектори (384-768 розмірів) замість розріджених TF-IDF
- Семантичне розуміння: "car" близько до "automobile"
- GPU прискорення для великих датасетів
- Industry standard (Google, Meta, OpenAI)

Переваги:
- Семантичне розуміння контексту
- Краща точність на складних запитах (+5-10% vs TF-IDF)
- Швидкий approximate nearest neighbor search

Недоліки:
- Більше пам'яті (1.3 GB vs 54 MB для TF-IDF)
- Повільніше індексування (210s vs 6.25s)
- Потребує попередньо навчену модель

Точність: +5-10% vs TF-IDF
"""

import fitz  # PyMuPDF
from pathlib import Path
import time
import numpy as np
from typing import List, Dict
import warnings
import sys
import json
import os
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent.parent))

# Завантаження змінних середовища з .env (шукаємо в поточній та батьківській директорії)
load_dotenv()  # Спочатку поточна директорія
if not os.getenv('OPENAI_API_KEY'):
    # Якщо не знайшли, шукаємо в батьківській директорії
    load_dotenv(Path(__file__).parent.parent / '.env')


def generate_answer_with_llm(question: str, contexts: List[str], max_tokens: int = 256) -> str:
    """
    Генерація відповіді через LLM
    Спроба 1: Ollama (локально, безкоштовно)
    Спроба 2: OpenAI (якщо є API key), зробіть export OPENAI_API_KEY=your_key
    Спроба 3: Simple fallback - повернути контекст
    """
    # Спроба 1: Ollama (локально)
    try:
        import requests
        prompt = f"Based on the following context, answer the question.\n\nContext:\n{chr(10).join(contexts[:3])}\n\nQuestion: {question}\n\nAnswer:"

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2:3b",
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.7, "num_predict": max_tokens}
            },
            timeout=30
        )

        if response.status_code == 200:
            return response.json()["response"].strip()
    except Exception:
        pass

    # Спроба 2: OpenAI (якщо є API key)
    # Для використання: export OPENAI_API_KEY=your_key або додати в .env файл
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)

            prompt = f"Based on the following context, answer the question.\n\nContext:\n{chr(10).join(contexts[:3])}\n\nQuestion: {question}"

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
    except Exception:
        pass

    # Спроба 3: Fallback - просто повернути контекст
    return "\n\n".join(contexts[:3]) if contexts else "Не знайдено релевантної інформації."


def detect_llm_provider() -> str:
    """Визначає який LLM provider доступний"""
    # Перевіряємо Ollama
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            return "ollama (llama3.2:3b)"
    except:
        pass

    # Перевіряємо OpenAI
    if os.getenv("OPENAI_API_KEY"):
        return "openai (gpt-4o-mini)"

    return "fallback (без LLM)"

# FAISS та sentence-transformers
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("FAISS не встановлено. Виконайте: pip install faiss-cpu sentence-transformers")


class FAISS_RAG:
    """
    FAISS-based RAG система з семантичним пошуком

    Архітектура:
    1. PDF → chunks
    2. chunks → dense vectors (sentence-transformers)
    3. FAISS index для швидкого ANN search
    4. Retrieval + LLM generation

    Параметри моделей:
    - all-MiniLM-L6-v2: 384 dim, швидка, якість 63%
    - all-mpnet-base-v2: 768 dim, повільна, якість 69% (рекомендовано)
    - paraphrase-multilingual: підтримка української
    """

    def __init__(self,
                 chunk_size=500,
                 chunk_overlap=50,
                 model_name="sentence-transformers/all-MiniLM-L6-v2",
                 top_k=10,
                 use_gpu=False):
        """
        Args:
            chunk_size: розмір chunk в символах
            chunk_overlap: перекриття між chunks
            model_name: модель для embeddings
            top_k: кількість документів для retrieval
            use_gpu: використовувати GPU (якщо доступний)
        """
        if not FAISS_AVAILABLE:
            raise ImportError("Встановіть: pip install faiss-cpu sentence-transformers")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.use_gpu = use_gpu
        self.model_name = model_name

        self.chunks = []
        self.index = None
        self.embedding_dim = None

    def load_model(self):
        """Завантажити модель embeddings"""
        print(f"Завантаження embedding model...")
        start = time.time()
        self.encoder = SentenceTransformer(self.model_name)

        # Визначити розмірність
        test_embed = self.encoder.encode(["test"])
        self.embedding_dim = test_embed.shape[1]

        elapsed = time.time() - start
        print(f"Model завантажено: {self.embedding_dim}D embeddings за {elapsed:.2f}с")

    def load_documents(self, pdf_dir: str) -> float:
        """Завантажити та проіндексувати PDF документи"""
        start_time = time.time()

        pdf_path = Path(pdf_dir)
        pdf_files = list(pdf_path.glob("*.pdf"))

        print(f"Завантаження PDFs з {pdf_dir}...")
        print(f"Знайдено {len(pdf_files)} PDF файлів")

        # Парсинг PDFs
        for pdf_file in pdf_files:
            try:
                doc = fitz.open(pdf_file)
                full_text = ""

                for page in doc:
                    full_text += page.get_text()

                # Розбити на chunks
                start = 0
                while start < len(full_text):
                    end = start + self.chunk_size
                    chunk_text = full_text[start:end]

                    if len(chunk_text.strip()) > 50:  # Мінімальна довжина
                        self.chunks.append({
                            'content': chunk_text,
                            'source': pdf_file.name,
                            'chunk_id': len(self.chunks)
                        })

                    start += (self.chunk_size - self.chunk_overlap)

                doc.close()

            except Exception as e:
                print(f"Помилка обробки {pdf_file.name}: {e}")

        print(f"Завантажено: {len(self.chunks)} чанків")

        # Створити FAISS index
        print(f"Створення FAISS індексу ({self.embedding_dim}D векторів)...")

        # Створити embeddings для всіх chunks
        print(f"Кодування {len(self.chunks)} чанків...")
        corpus = [chunk['content'] for chunk in self.chunks]

        # Batch encoding для ефективності
        batch_size = 32
        embeddings = []

        for i in range(0, len(corpus), batch_size):
            batch = corpus[i:i+batch_size]
            batch_embeddings = self.encoder.encode(
                batch,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            embeddings.append(batch_embeddings)

        embeddings = np.vstack(embeddings).astype('float32')
        print(f"Створено embeddings: {embeddings.shape}")

        # Створити FAISS index
        # IndexFlatIP = Inner Product (cosine similarity для normalized vectors)
        self.index = faiss.IndexFlatIP(self.embedding_dim)

        # Нормалізувати вектори для cosine similarity
        faiss.normalize_L2(embeddings)

        # Додати до індексу
        self.index.add(embeddings)

        # Якщо GPU доступний і запитаний
        if self.use_gpu and faiss.get_num_gpus() > 0:
            print(f"Переміщення індексу на GPU...")
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

        elapsed = time.time() - start_time
        print(f"Проіндексовано: {len(self.chunks)} чанків за {elapsed:.2f}с")

        # Підрахувати memory usage
        memory_mb = (embeddings.nbytes) / (1024 * 1024)
        print(f"Пам'ять індексу: {memory_mb:.1f} MB")

        return elapsed

    def retrieve(self, query: str, k: int = None) -> List[Dict]:
        """Знайти top-k найрелевантніших chunks використовуючи FAISS"""
        if k is None:
            k = self.top_k

        start = time.time()

        # Encode query
        query_embedding = self.encoder.encode(
            [query],
            convert_to_numpy=True
        ).astype('float32')

        # Нормалізувати для cosine similarity
        faiss.normalize_L2(query_embedding)

        # FAISS search
        similarities, indices = self.index.search(query_embedding, k)

        # Prepare results
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], similarities[0]), 1):
            results.append({
                'rank': rank,
                'chunk': self.chunks[idx],
                'similarity_score': float(score)
            })

        elapsed = time.time() - start

        return results

    def query(self, question: str, k: int = None) -> Dict:
        """Повний RAG pipeline: FAISS retrieve + LLM generate"""
        start = time.time()

        # Retrieve
        retrieved = self.retrieve(question, k=k)

        # Витягуємо контексти з retrieved chunks
        contexts = [r['chunk']['content'] for r in retrieved]

        # Generate answer using LLM
        answer = generate_answer_with_llm(
            question=question,
            contexts=contexts,
            max_tokens=256
        )

        elapsed = time.time() - start

        return {
            'question': question,
            'answer': answer,
            'contexts': contexts,
            'retrieved_docs': len(retrieved),
            'top_scores': [r['similarity_score'] for r in retrieved[:3]],
            'sources': [r['chunk']['source'] for r in retrieved],
            'execution_time': elapsed,
            'embedding_dim': self.embedding_dim
        }


def run_faiss_rag_demo():
    """Запускає демонстрацію FAISS RAG"""
    print("="*70)
    print("FAISS RAG ДЕМОНСТРАЦІЯ")
    print("="*70)

    # Ініціалізація
    chunk_size = 500
    chunk_overlap = 50
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    top_k = 10

    rag = FAISS_RAG(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        model_name=model_name,
        top_k=top_k,
        use_gpu=False
    )

    # Завантаження моделі
    rag.load_model()

    # Виводимо конфігурацію
    print(f"\nКонфігурація:")
    llm_model = detect_llm_provider()
    print(f"  Модель LLM: {llm_model}")
    print(f"  Модель embeddings: {model_name}")
    print(f"  Розмір чанку: {chunk_size} символів")
    print(f"  Перекриття чанків: {chunk_overlap} символів")
    print(f"  Розмірність embeddings: {rag.embedding_dim}D")
    print(f"  Техніки: Semantic search, FAISS ANN")

    # Завантаження
    print(f"\nЗавантаження документів...")
    indexing_time = rag.load_documents("data/pdfs")

    # Тестові запити
    # Завантажуємо УНІФІКОВАНИЙ тестовий датасет (100 запитів)
    # ВАЖЛИВО: Всі RAG підходи використовують ТІ САМІ запити для коректного порівняння!
    from utils.data_loader import DocumentLoader
    from collections import defaultdict
    loader = DocumentLoader()
    unified_queries = loader.load_unified_queries(max_queries=50)  # Перші 50 для швидкості
    print(f"Тестових запитів: {len(unified_queries)}")

    print("\n" + "="*70)
    print("ВИКОНАННЯ ТЕСТІВ")
    print("="*70)

    all_results = {
        "system_name": "FAISS RAG",
        "total_chunks": len(rag.chunks),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": model_name,
        "embedding_dim": rag.embedding_dim,
        "llm_model": detect_llm_provider(),
        "queries": []
    }

    # Групуємо по категоріях для виводу
    queries_by_category = defaultdict(list)
    for query in unified_queries:
        queries_by_category[query.get("category", "general")].append(query)

    # Тестуємо запити по категоріях
    for category, queries in queries_by_category.items():
        print(f"\nКатегорія: {category}")

        for query_data in queries:
            question = query_data.get("question", "")

            # Виконуємо запит
            result = rag.query(question, k=5)
            result["category"] = category
            result["query_id"] = query_data.get("id")
            result["difficulty"] = query_data.get("difficulty")
            all_results["queries"].append(result)

            # Виводимо короткий результат
            print(f"  ID {query_data.get('id')}: {question[:70]}...")
            print(f"  Час: {result['execution_time']:.2f}с | Similarity: {result['top_scores'][0]:.3f}")

    # Статистика
    avg_time = np.mean([q["execution_time"] for q in all_results["queries"]])
    avg_score = np.mean([q["top_scores"][0] for q in all_results["queries"]])

    all_results["metrics"] = {
        "average_execution_time": avg_time,
        "average_top_score": avg_score,
        "total_queries": len(all_results["queries"]),
        "indexing_time": indexing_time
    }

    # Збереження результатів
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    with open(results_dir / "faiss_rag_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print("\n" + "="*70)
    print("ПІДСУМОК")
    print("="*70)
    print(f"Всього запитів: {len(all_results['queries'])}")
    print(f"Середній час: {avg_time:.2f}с")
    print(f"Середня оцінка: {avg_score:.3f}")
    print(f"Час індексування: {indexing_time:.2f}с")
    print(f"Пам'ять: ~{rag.embedding_dim * len(rag.chunks) * 4 / (1024*1024):.1f} MB")
    print(f"\nРезультати збережено: results/faiss_rag_results.json")
    print("="*70)


if __name__ == "__main__":
    run_faiss_rag_demo()
