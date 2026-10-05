from src.utils.utils import article_key_function
import numpy as np

# Case of just finding conflicts
def rex2(query, articles, conflicts, predicts_from_reranker, k0):
    
    # new_results = articles[:k0]
    true_articles = [a for a in articles[:k0] if a in predicts_from_reranker.get(query, []) and a != query]

    tested_articles = [article_key_function(a) for a in articles[:k0]]
    import copy
    conflicts = copy.deepcopy(conflicts)
    conflicts[query] = []
    prestige_articles = []
    # print(true_articles)
    for t in true_articles:
        
        prestige_articles.extend(conflicts.get(t, []))

    prestige_articles = list(set([article_key_function(p) for p in prestige_articles]))

    result_articles = [p for p in prestige_articles if p not in tested_articles]

    result_articles.sort()

    return result_articles


def rocchio_binary_retriever(model, laws_df, chroma_collection, article_to_check, top_k=500, batch_size = 8, tokenizer = None):
    """
    Bi-encoder retriever function to find conflicts in legal texts.

    Args:
    model_path (str): Path to the trained model directory.
    laws_csv_path (str): Path to the laws.csv file.
    chroma_db_name (str): Name of the Chroma DB where encodings will be stored.
    article_to_check (str): The article to check for conflicts.
    classification_method (str): Method to use for classification ('cosine' or 'classification_model').
    top_k_val (int): Number of top-k articles to retrieve.

    Returns:
    List of top-k articles that contradict the input article.
    """

    MAX_TOKEN_LENGTH = 4096
    ROCCHIO_TOP_K = 10
    import torch


     # Encode the input article
    inputs = tokenizer.encode_plus(
        article_to_check,
        add_special_tokens=True,
        max_length=MAX_TOKEN_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"].to(model.encoder.device)
    attention_mask = inputs["attention_mask"].to(model.encoder.device)

    with torch.inference_mode():
        encoded_article = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = encoded_article.last_hidden_state[0, 0, :].cpu().numpy()


    results = chroma_collection.query(
        query_embeddings=[pooled_output.tolist()],
        n_results=ROCCHIO_TOP_K,
        include=["embeddings", "documents"]
    )
    # Extract top_k embeddings and compute Rocchio vector
    top_k_embeddings = np.array(results["embeddings"][0])  # shape (top_k, dim)
    # Include original query vector
    all_vectors = np.vstack([pooled_output, top_k_embeddings])
    rocchio_vector = np.mean(all_vectors, axis=0)
    # Ensure vector is float32 to match model dtype
    rocchio_vector = rocchio_vector.astype(np.float32)

    # Retrieve documents based on Rocchio vector
    rocchio_results = chroma_collection.query(
        query_embeddings=[rocchio_vector.tolist()],
        n_results=top_k,
        include=["documents"]
    )
    rocchio_top_k_articles = rocchio_results["documents"][0]

    return rocchio_vector, rocchio_top_k_articles
