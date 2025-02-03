from model import qa_chain, embeddings
from retriever import get_multi_query_retriever
from sklearn.metrics import precision_score, recall_score, f1_score
import json

# Load test data (queries and relevant documents)
def load_test_data(test_data_path):
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    return test_data

# Evaluate the retriever
def evaluate_retriever(retriever, test_data):
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for query, relevant_docs in test_data.items():
        retrieved_docs = retriever.retrieve(query)
        retrieved_doc_ids = [doc.metadata["source"] for doc in retrieved_docs]
        relevant_doc_ids = relevant_docs  # Ground truth document IDs

        # Compute metrics
        precision = precision_score(relevant_doc_ids, retrieved_doc_ids, average='micro')
        recall = recall_score(relevant_doc_ids, retrieved_doc_ids, average='micro')
        f1 = f1_score(relevant_doc_ids, retrieved_doc_ids, average='micro')

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    return {
        "precision": sum(precision_scores) / len(precision_scores),
        "recall": sum(recall_scores) / len(recall_scores),
        "f1": sum(f1_scores) / len(f1_scores),
    }

# Evaluate the generator (QA chain)
def evaluate_generator(qa_chain, test_data):
    correct_answers = 0
    total_questions = len(test_data)

    for query, relevant_answer in test_data.items():
        result = qa_chain.invoke(query)
        generated_answer = result.get('result', "")

        # Compare generated answer to ground truth (simple exact match for now)
        if generated_answer.strip() == relevant_answer.strip():
            correct_answers += 1

    accuracy = correct_answers / total_questions
    return {"accuracy": accuracy}

# Main evaluation function
def evaluate(test_data_path):
    # Load test data
    test_data = load_test_data(test_data_path)

    # Evaluate retriever
    retriever = get_multi_query_retriever(qa_chain.llm)
    retriever_metrics = evaluate_retriever(retriever, test_data["retriever"])

    # Evaluate generator
    generator_metrics = evaluate_generator(qa_chain, test_data["generator"])

    # Print results
    print("Retriever Metrics:")
    print(f"Precision: {retriever_metrics['precision']:.2f}")
    print(f"Recall: {retriever_metrics['recall']:.2f}")
    print(f"F1-Score: {retriever_metrics['f1']:.2f}")

    print("\nGenerator Metrics:")
    print(f"Accuracy: {generator_metrics['accuracy']:.2f}")

if __name__ == "__main__":
    # Path to test data
    test_data_path = "test_data.json"
    evaluate(test_data_path)