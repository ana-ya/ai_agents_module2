#!/usr/bin/env python3
"""
BM25 RAG - Keyword-based пошук
==============================
BM25 (Best Matching 25) - покращена версія TF-IDF з:
- Term saturation: багато повторень не завжди означає більше релевантності
- Document length normalization: враховує довжину документа
- Industry standard (Elasticsearch, Solr)

Параметри:
- k1 = 1.5: контролює term saturation
- b = 0.75: контролює document length normalization

Точність: +2-3% vs TF-IDF
"""

import fitz  # PyMuPDF
from pathlib import Path
import time
import numpy as np
from typing import List, Dict
import math
from collections import Counter
import sys
import json
import os
from dotenv import load_dotenv

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
    # Для використання: export OPENAI_API_KEY=your_key
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


class BM25Scorer:
    """
    BM25 (Okapi BM25) scorer

    Формула:
    score(D, Q) = Σ IDF(qi) · (f(qi, D) · (k1 + 1)) / (f(qi, D) + k1 · (1 - b + b · |D| / avgdl))

    Де:
    - f(qi, D) - частота терму qi в документі D
    - |D| - довжина документа D
    - avgdl - середня довжина документа
    - k1, b - параметри налаштування
    """

    def __init__(self, k1=1.5, b=0.75):
        """
        Args:
            k1: контролює term saturation (типово 1.2-2.0)
            b: контролює вплив довжини документа (типово 0.75)
        """
        self.k1 = k1
        self.b = b
        self.corpus = []
        self.corpus_size = 0
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.inverted_index = {}  # term -> list of doc_ids

    def fit(self, corpus: List[str]):
        """Навчити BM25 на корпусі документів"""
        self.corpus = corpus
        self.corpus_size = len(corpus)

        # Підрахувати довжини документів
        self.doc_len = [len(doc.split()) for doc in corpus]
        self.avgdl = sum(self.doc_len) / self.corpus_size if self.corpus_size > 0 else 0

        # Побудувати inverted index та document frequencies
        df = {}
        self.inverted_index = {}

        for doc_idx, document in enumerate(corpus):
            words = set(document.lower().split())
            for word in words:
                df[word] = df.get(word, 0) + 1

                # Додати до inverted index
                if word not in self.inverted_index:
                    self.inverted_index[word] = []
                self.inverted_index[word].append(doc_idx)

        # Обчислити IDF для кожного слова
        self.idf = {}
        for word, freq in df.items():
            # IDF = log((N - df + 0.5) / (df + 0.5) + 1)
            self.idf[word] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1)

    def score(self, query: str, doc_idx: int) -> float:
        """Обчислити BM25 score для query відносно документа"""
        score = 0.0
        doc = self.corpus[doc_idx]
        doc_words = doc.lower().split()
        doc_len = self.doc_len[doc_idx]

        # Підрахувати частоти термів у документі
        term_freqs = Counter(doc_words)

        # Для кожного терму в запиті
        for term in query.lower().split():
            if term not in self.idf:
                continue

            # Частота терму в документі
            tf = term_freqs.get(term, 0)

            # BM25 formula
            idf = self.idf[term]
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))

            score += idf * (numerator / denominator)

        return score

    def get_top_k(self, query: str, k: int = 10) -> List[tuple]:
        """Знайти top-k документів для запиту (оптимізовано з inverted index)"""
        # Використати inverted index для швидкого пошуку candidates
        query_terms = query.lower().split()
        candidate_docs = set()

        for term in query_terms:
            if term in self.inverted_index:
                candidate_docs.update(self.inverted_index[term])

        # Якщо нема candidates, повернути порожній результат
        if not candidate_docs:
            return []

        # Обчислити scores тільки для candidates
        scores = [(doc_idx, self.score(query, doc_idx)) for doc_idx in candidate_docs]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


class BM25_RAG:
    """
    BM25-based RAG система

    Переваги vs TF-IDF:
    - Краща точність (+2-3%)
    - Term saturation
    - Document length normalization

    Коли використовувати:
    - Документи різної довжини
    - Keyword-heavy queries
    - Потрібна простота + якість
    """

    def __init__(self,
                 chunk_size=500,
                 chunk_overlap=50,
                 k1=1.5,
                 b=0.75,
                 top_k=10):
        """
        Args:
            chunk_size: розмір chunk в символах
            chunk_overlap: перекриття між chunks
            k1: BM25 term saturation parameter
            b: BM25 document length parameter
            top_k: кількість документів для retrieval
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k1 = k1
        self.b = b
        self.top_k = top_k

        self.chunks = []
        self.bm25 = None

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

        # Створити BM25 index
        print(f"Створення BM25 індексу (k1={self.k1}, b={self.b})...")
        corpus = [chunk['content'] for chunk in self.chunks]

        self.bm25 = BM25Scorer(k1=self.k1, b=self.b)
        self.bm25.fit(corpus)

        elapsed = time.time() - start_time
        print(f"Проіндексовано: {len(self.chunks)} чанків за {elapsed:.2f}с")

        return elapsed

    def retrieve(self, query: str, k: int = None) -> List[Dict]:
        """Знайти top-k найрелевантніших chunks використовуючи BM25"""
        if k is None:
            k = self.top_k

        start = time.time()

        # BM25 scoring
        top_docs = self.bm25.get_top_k(query, k=k)

        # Prepare results
        results = []
        for rank, (doc_idx, score) in enumerate(top_docs, 1):
            results.append({
                'rank': rank,
                'chunk': self.chunks[doc_idx],
                'bm25_score': score
            })

        elapsed = time.time() - start

        return results

    def query(self, question: str, k: int = None) -> Dict:
        """Повний RAG pipeline: BM25 retrieve + LLM generate"""
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
            'top_scores': [r['bm25_score'] for r in retrieved[:3]],
            'sources': [r['chunk']['source'] for r in retrieved],
            'execution_time': elapsed,
            'k1': self.k1,
            'b': self.b
        }


def run_bm25_rag_demo():
    """Запускає демонстрацію BM25 RAG"""
    print("="*70)
    print("BM25 RAG ДЕМОНСТРАЦІЯ")
    print("="*70)

    # Ініціалізація
    chunk_size = 500
    chunk_overlap = 50
    k1 = 1.5
    b = 0.75
    top_k = 10

    rag = BM25_RAG(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        k1=k1,
        b=b,
        top_k=top_k
    )

    # Виводимо конфігурацію
    print(f"\nКонфігурація:")
    llm_model = detect_llm_provider()
    print(f"  Модель LLM: {llm_model}")
    print(f"  Розмір чанку: {chunk_size} символів")
    print(f"  Перекриття чанків: {chunk_overlap} символів")
    print(f"  BM25 параметри: k1={k1}, b={b}")
    print(f"  Техніки: BM25 scoring, Inverted index")

    # Завантаження
    print(f"\nЗавантаження документів...")
    indexing_time = rag.load_documents("data/pdfs")

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
        "system_name": "BM25 RAG",
        "total_chunks": len(rag.chunks),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "k1": k1,
        "b": b,
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
            print(f"  Час: {result['execution_time']:.2f}с | BM25: {result['top_scores'][0]:.2f}")

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

    with open(results_dir / "bm25_rag_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print("\n" + "="*70)
    print("ПІДСУМОК")
    print("="*70)
    print(f"Всього запитів: {len(all_results['queries'])}")
    print(f"Середній час: {avg_time:.2f}с")
    print(f"Середня оцінка: {avg_score:.2f}")
    print(f"Час індексування: {indexing_time:.2f}с")
    print(f"\nРезультати збережено: results/bm25_rag_results.json")
    print("="*70)


if __name__ == "__main__":
    run_bm25_rag_demo()
