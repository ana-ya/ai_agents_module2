"""
RAGAS Evaluation для всіх RAG підходів
=======================================

Цей скрипт оцінює ВСІХ підходів RAG (Naive, Advanced, Hybrid, Corrective тощо)
за допомогою RAGAS metrics окремо від їх виконання.

Використання:
1. Спочатку запустіть всі demo скрипти (вони створять JSON результати)
2. Потім запустіть цей скрипт для RAGAS evaluation

Результат: Порівняльна таблиця з RAGAS метриками для кожного підходу
"""

import json
import os
from pathlib import Path
from typing import Dict, List
import sys
from dotenv import load_dotenv

# Завантаження змінних середовища з .env (шукаємо в поточній та батьківській директорії)
load_dotenv()  # Спочатку поточна директорія
if not os.getenv('OPENAI_API_KEY'):
    # Якщо не знайшли, шукаємо в батьківській директорії
    load_dotenv(Path(__file__).parent.parent / '.env')

# RAGAS imports
try:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy
    HAS_RAGAS = True
except ImportError:
    HAS_RAGAS = False
    print("❌ RAGAS не встановлено")
    print("   Встановіть: pip install ragas datasets langchain-openai")
    sys.exit(1)


def load_results_file(file_path: str) -> Dict:
    """Завантажити результати з JSON файлу"""
    path = Path(file_path)
    if not path.exists():
        return None

    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def prepare_ragas_dataset(results: Dict) -> Dataset:
    """
    Підготувати дані у форматі RAGAS Dataset

    Args:
        results: Результати з JSON файлу

    Returns:
        Dataset для RAGAS
    """
    # Підтримка різних форматів JSON
    queries = results.get("queries", [])

    # Hybrid RAG має структуру: config_results[].queries
    if not queries and "config_results" in results:
        queries = []
        for config in results["config_results"]:
            queries.extend(config.get("queries", []))

    data = {
        "question": [],
        "answer": [],
        "contexts": []
    }

    for query in queries:
        # Беремо question (підтримка різних форматів)
        question = query.get("question") or query.get("query", "")
        if not question:
            continue

        # Беремо answer
        answer = query.get("answer", "")

        # Беремо contexts (якщо є)
        contexts = query.get("contexts", [])
        if not contexts:
            # Якщо contexts немає, беремо з sources або пропускаємо
            sources = query.get("sources", [])
            if sources:
                # Використовуємо sources як contexts (не ідеально але краще ніж нічого)
                contexts = [f"Source: {s}" for s in sources]
            else:
                # Пропускаємо якщо немає контексту
                continue

        data["question"].append(question)
        data["answer"].append(answer)
        data["contexts"].append(contexts)

    return Dataset.from_dict(data)


def evaluate_rag_approach(
    approach_name: str,
    results: Dict,
    llm,
    embeddings
) -> Dict:
    """
    Оцінити один підхід RAG за допомогою RAGAS

    Args:
        approach_name: Назва підходу (Naive RAG, Advanced RAG тощо)
        results: Результати з JSON
        llm: LLM для RAGAS
        embeddings: Embeddings для RAGAS

    Returns:
        Dict з RAGAS метриками
    """
    print(f"\n{'='*70}")
    print(f"📊 Оцінка: {approach_name}")
    print(f"{'='*70}")

    try:
        # Підготовка dataset
        dataset = prepare_ragas_dataset(results)

        if len(dataset) == 0:
            print(f"⚠️  Немає даних для оцінки")
            return {"error": "No data"}

        print(f"✅ Підготовлено {len(dataset)} запитів")

        # Метрики для оцінки
        metrics = [
            faithfulness,      # Чи базується відповідь на контексті?
            answer_relevancy,  # Чи релевантна відповідь запиту?
        ]

        print("🧪 Запуск RAGAS evaluation...")
        print("   Це може зайняти 1-2 хвилини...")

        # Запуск evaluation з Ollama
        evaluation_result = evaluate(
            dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings
        )

        # Конвертація результатів
        # RAGAS може повернути списки (по одному значенню на запит), тому беремо середнє
        # EvaluationResult об'єкт підтримує індексацію, але не .get()
        faithfulness_val = evaluation_result["faithfulness"]
        answer_relevancy_val = evaluation_result["answer_relevancy"]

        # Якщо список - беремо середнє, якщо скаляр - просто конвертуємо
        if isinstance(faithfulness_val, list):
            import numpy as np
            faithfulness_val = np.mean(faithfulness_val)
        if isinstance(answer_relevancy_val, list):
            import numpy as np
            answer_relevancy_val = np.mean(answer_relevancy_val)

        ragas_scores = {
            "approach": approach_name,
            "faithfulness": float(faithfulness_val),
            "answer_relevancy": float(answer_relevancy_val),
            "queries_evaluated": len(dataset)
        }

        # Середній score
        ragas_scores["average_score"] = (
            ragas_scores["faithfulness"] + ragas_scores["answer_relevancy"]
        ) / 2

        print(f"✅ Faithfulness:    {ragas_scores['faithfulness']:.3f}")
        print(f"✅ Answer Relevancy: {ragas_scores['answer_relevancy']:.3f}")
        print(f"✅ Average Score:    {ragas_scores['average_score']:.3f}")

        return ragas_scores

    except Exception as e:
        print(f"❌ Помилка оцінки: {e}")
        return {"error": str(e)}


def print_comparison_table(all_scores: List[Dict]):
    """Вивести порівняльну таблицю метрик"""
    print("\n" + "="*90)
    print("📊 ПОРІВНЯЛЬНА ТАБЛИЦЯ RAGAS МЕТРИК")
    print("="*90)
    print()

    # Заголовок таблиці
    print(f"{'Підхід':<25} {'Faithfulness':>15} {'Relevancy':>15} {'Average':>15} {'Queries':>10}")
    print("-" * 90)

    # Сортуємо за середнім score
    sorted_scores = sorted(
        [s for s in all_scores if "error" not in s],
        key=lambda x: x.get("average_score", 0),
        reverse=True
    )

    for scores in sorted_scores:
        approach = scores["approach"]
        faith = scores["faithfulness"]
        relev = scores["answer_relevancy"]
        avg = scores["average_score"]
        queries = scores["queries_evaluated"]

        # Визначаємо емоджі статусу
        if avg >= 0.85:
            status = "✅"
        elif avg >= 0.70:
            status = "⚠️ "
        else:
            status = "❌"

        print(f"{status} {approach:<23} {faith:>15.3f} {relev:>15.3f} {avg:>15.3f} {queries:>10}")

    print("="*90)
    print()
    print("Легенда:")
    print("  ✅ Відмінно (≥0.85) - готово для production")
    print("  ⚠️  Прийнятно (0.70-0.85) - потрібні покращення")
    print("  ❌ Низько (<0.70) - критичні проблеми")
    print()


def main():
    """Головна функція - оцінює всі RAG підходи"""
    print("="*90)
    print(" RAGAS EVALUATION ДЛЯ ВСІХ RAG ПІДХОДІВ")
    print("="*90)
    print()

    if not HAS_RAGAS:
        return

    # Налаштування OpenAI для RAGAS
    print(" Налаштування OpenAI для RAGAS...")
    print("   Модель: gpt-4o-mini (швидко, недорого)")
    print()

    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        openai_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    except Exception as e:
        print(f"❌ Помилка підключення до OpenAI: {e}")
        print("   Переконайтесь що OPENAI_API_KEY встановлено")
        sys.exit(1)

    # Список підходів для оцінки
    approaches = [
        ("Naive RAG", "results/naive_rag_results.json"),
        ("Naive RAG (Chroma DB)", "results/naive_rag_chroma_db_results.json"),
        ("Advanced RAG", "results/advanced_rag_results.json"),
        ("BM25 RAG", "results/bm25_rag_results.json"),
        ("FAISS RAG", "results/faiss_rag_results.json"),
        ("Hybrid RAG (All)", "results/hybrid_rag_all_results.json"),
        ("Corrective RAG", "results/corrective_rag_results.json"),
        # ("Multimodal RAG", "results/multimodal_rag_results.json"),  # Поки немає
    ]

    all_scores = []

    # Оцінюємо кожен підхід
    for approach_name, results_file in approaches:
        print(f"\n{'='*90}")
        print(f"🔍 Завантаження: {results_file}")

        results = load_results_file(results_file)

        if results is None:
            print(f"  Файл не знайдено: {results_file}")
            print(f"   Спочатку запустіть відповідний demo скрипт")
            continue

        # Оцінюємо цей підхід
        scores = evaluate_rag_approach(
            approach_name,
            results,
            openai_llm,
            openai_embeddings
        )

        if "error" not in scores:
            all_scores.append(scores)

    # Виводимо порівняльну таблицю
    if all_scores:
        print_comparison_table(all_scores)

        # Зберігаємо результати
        output = {
            "ragas_comparison": all_scores,
            "summary": {
                "total_approaches": len(all_scores),
                "best_approach": max(all_scores, key=lambda x: x["average_score"])["approach"],
                "best_score": max(all_scores, key=lambda x: x["average_score"])["average_score"]
            }
        }

        output_file = "results/ragas_comparison.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f" Результати збережено: {output_file}")

        # Висновки
        best = output["summary"]["best_approach"]
        best_score = output["summary"]["best_score"]

        print()
        print("="*90)
        print(" ВИСНОВКИ")
        print("="*90)
        print(f"Найкращий підхід: {best} (score: {best_score:.3f})")
        print()

        if best_score >= 0.85:
            print(" Відмінний результат! Система готова для production.")
        elif best_score >= 0.70:
            print("  Прийнятний результат, але є простір для покращення.")
        else:
            print(" Низька якість. Рекомендується використати більш складний підхід.")

        print()
        print("="*90)
    else:
        print("\n Не вдалося оцінити жоден підхід")
        print("   Переконайтесь що demo скрипти були запущені:")
        print("   - python rag_demos/naive_rag/naive_rag_demo.py")
        print("   - python rag_demos/advanced_rag/advanced_rag_demo.py")
        print("   - тощо...")


if __name__ == "__main__":
    main()
