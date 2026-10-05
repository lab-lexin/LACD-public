import torch
import torch.nn.functional as F
from src.methods.LawGNN.article_network.article_network import ArticleNetwork
import numpy as np

from src.utils.encoder.utils import MAX_TOKEN_LENGTH
from src.utils.utils import article_key_function

import time

def cross_retriever(query, top_k_articles, cross_encoder_model, tokenizer, article_network:ArticleNetwork, batch_size=64, index_method = "none", query_vector = None, max_length=None, fp16=False):
    """
    Function to use a cross-encoder to distinguish conflicts in top-k retrieved articles using batch processing.

    Major args:
    query (str): The original article to compare against top-k retrieved articles.
    top_k_articles (list): List of top-k retrieved articles.
    model_path (str): Path to the cross-encoder model.
    top_k (int): The number of top articles to return based on contradiction score.
    batch_size (int): The number of articles to process in one batch.

    Returns:
    List of (article, score) where score represents the classification score of contradiction.
    """
    # Load tokenizer and cross-encoder model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cross_encoder_model.to(device)

    # FIX: Single reserved query slot (N0 + 1) to eliminate O(N) reallocation
    # and copying across hundreds of evaluation queries.
    if not hasattr(cross_encoder_model, "_base_node_count"):
        cross_encoder_model._base_node_count = cross_encoder_model.vector_tensor.shape[0]
        dummy_slot = torch.zeros(
            (1, cross_encoder_model.vector_tensor.shape[1]),
            device=cross_encoder_model.vector_tensor.device,
            dtype=cross_encoder_model.vector_tensor.dtype,
        )
        cross_encoder_model.vector_tensor = torch.cat(
            [cross_encoder_model.vector_tensor, dummy_slot], dim=0
        )

    base_n = cross_encoder_model._base_node_count

    query_key = article_key_function(query)
    node_idx = article_network.article_key_to_idx.get(query_key)
    is_new_query = node_idx is None or node_idx >= base_n
    if is_new_query:
        article_idx = base_n
        if index_method != "none":
            # hybrid: query_vector from binary_retriever is available; fallback to zeros if None
            if query_vector is None:
                query_vector_tensor = torch.zeros(
                    (cross_encoder_model.vector_tensor.shape[1],),
                    device=cross_encoder_model.vector_tensor.device,
                    dtype=cross_encoder_model.vector_tensor.dtype,
                )
            else:
                if isinstance(query_vector, np.ndarray):
                    query_vector_tensor = torch.from_numpy(query_vector).to(
                        device=cross_encoder_model.vector_tensor.device,
                        dtype=cross_encoder_model.vector_tensor.dtype,
                    )
                else:
                    query_vector_tensor = query_vector.to(
                        device=cross_encoder_model.vector_tensor.device,
                        dtype=cross_encoder_model.vector_tensor.dtype,
                    )
            with torch.no_grad():
                cross_encoder_model.vector_tensor[base_n].copy_(query_vector_tensor.flatten())
    else:
        article_idx = node_idx
    # fp16 autocast scope (CUDA only; no-op otherwise, never mutates the model)
    use_cuda = torch.cuda.is_available()
    autocast_ctx = torch.autocast(device_type="cuda" if use_cuda else "cpu", dtype=torch.float16, enabled=bool(fp16) and use_cuda)

    # Per-query GNN cache (eval only): conv output depends solely on tensors +
    # weights, identical for every batch of this query (slot already updated
    # above). No cross-query state is kept. Training callers never pass it.
    cached_nodes = None
    if index_method != "none" and not cross_encoder_model.training and hasattr(cross_encoder_model, "encode_graph_nodes"):
        with torch.inference_mode(), autocast_ctx:
            cached_nodes = cross_encoder_model.encode_graph_nodes()

    retrieval_start_time = time.time()


    # List to store results
    conflicts = []

    # Hoisted (once per call, not per batch): map every candidate to its graph
    # index up front, then slice per batch. Identical values; transfers drop
    # from per-batch to one. KeyError surfaces before the first batch instead
    # of mid-loop, with the same net effect (the query fails either way).
    all_article2_idx = torch.tensor(
        [min(article_network.article_key_to_idx.get(article_key_function(a), base_n), base_n) for a in top_k_articles],
        dtype=torch.long,
        device=device,
    )
    article1_idx_full = torch.full((len(top_k_articles),), article_idx, dtype=torch.long, device=device)

    # Process top_k_articles in batches
    for i in range(0, len(top_k_articles), batch_size):
        batch_articles = top_k_articles[i:i+batch_size]



        max_length = min([max_length or MAX_TOKEN_LENGTH, tokenizer.model_max_length])

        # Tokenize the batch of (article, batch_articles) pairs
        inputs = tokenizer.batch_encode_plus(
            [(query, retrieved_article) for retrieved_article in batch_articles],
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        # Move input tensors to the appropriate device (pinned staging makes
        # the non_blocking launch genuinely async; the forward syncs as needed)
        input_ids = inputs['input_ids'].pin_memory().to(device, non_blocking=True) # type: ignore
        attention_mask = inputs['attention_mask'].pin_memory().to(device, non_blocking=True) # type: ignore



        article2_idx_tensor = all_article2_idx[i:i+len(batch_articles)]
        article1_idx_tensor = article1_idx_full[i:i+len(batch_articles)]


        # Get model prediction for the batch
        with torch.inference_mode(), autocast_ctx:
            if index_method != "none":
                _fwd_kwargs = dict(article1_idx=article1_idx_tensor, article2_idx=article2_idx_tensor,
                                   input_ids=input_ids, attention_mask=attention_mask)
                # Vanilla has no conv layers (hence no encode_graph_nodes):
                # only models with a cache get the extra kwarg.
                if cached_nodes is not None:
                    _fwd_kwargs["precomputed_nodes"] = cached_nodes
                outputs = cross_encoder_model(**_fwd_kwargs)
            else:
                outputs = cross_encoder_model(input_ids=input_ids, attention_mask=attention_mask)

            logits = outputs.logits

            if logits.shape[1] == 1:
                # 한 개의 확률을 출력하는 경우 (예: [batch_size, 1])
                probs = torch.sigmoid(logits).cpu().numpy()  # Sigmoid로 변환하여 확률 얻기
            else:
                # 두 개의 확률을 출력하는 경우 (예: [batch_size, 2])
                probs = F.softmax(logits, dim=-1).cpu().numpy()  # Softmax로 변환하여 확률 얻기

        # Process the results for each article in the batch
        for idx, retrieved_article in enumerate(batch_articles):
            if logits.shape[1] == 1:
                # Sigmoid로 변환된 단일 값 (True일 확률)
                contradiction_score = probs[idx][0]
            else:
                # Softmax로 변환된 두 번째 값 (True일 확률)
                contradiction_score = probs[idx][1]  # 두 번째 값이 True일 확률
            
            conflicts.append((retrieved_article, contradiction_score))

    # Sort conflicts based on the contradiction score in descending order
    conflicts = sorted(conflicts, key=lambda x: x[1], reverse=True)
    
    retrieval_end_time = time.time()

    elapsed_time = retrieval_end_time - retrieval_start_time
    # print(f"crossencoder 실행 시간: {elapsed_time:.6f}초")


    return conflicts


def noLM_cross_retriever(article, top_k_articles, model_path, article_network:ArticleNetwork, top_k=10, batch_size=64, method = "baseline", index_method = "none", crossencoder_index_db="kbb-baseline", article_vector = None):
    """
    Function to use a cross-encoder to distinguish contradictions in top-k retrieved articles using batch processing.

    Major args:
    article (str): The original article to compare against top-k retrieved articles.
    top_k_articles (list): List of top-k retrieved articles.
    model_path (str): Path to the cross-encoder model.
    top_k (int): The number of top articles to return based on contradiction score.
    batch_size (int): The number of articles to process in one batch.

    Returns:
    List of (article, score) where score represents the classification score of contradiction.
    """
    # Load cross-encoder model
    cross_encoder_model = torch.load(model_path + "/model.pth", weights_only=False)
    cross_encoder_model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cross_encoder_model.to(device)

    # FIX: Single reserved query slot (N0 + 1) to eliminate O(N) reallocation
    # and copying across hundreds of evaluation queries.
    if not hasattr(cross_encoder_model, "_base_node_count"):
        cross_encoder_model._base_node_count = cross_encoder_model.vector_tensor.shape[0]
        dummy_slot = torch.zeros(
            (1, cross_encoder_model.vector_tensor.shape[1]),
            device=cross_encoder_model.vector_tensor.device,
            dtype=cross_encoder_model.vector_tensor.dtype,
        )
        cross_encoder_model.vector_tensor = torch.cat(
            [cross_encoder_model.vector_tensor, dummy_slot], dim=0
        )

    base_n = cross_encoder_model._base_node_count

    query_key = article_key_function(article)
    is_new_query = query_key not in article_network.article_key_to_idx
    if is_new_query:
        article_idx = base_n
        if index_method != "none":
            if article_vector is None:
                article_vector_tensor = torch.zeros(
                    (cross_encoder_model.vector_tensor.shape[1],),
                    device=cross_encoder_model.vector_tensor.device,
                    dtype=cross_encoder_model.vector_tensor.dtype,
                )
            else:
                if isinstance(article_vector, np.ndarray):
                    article_vector_tensor = torch.from_numpy(article_vector).to(
                        device=cross_encoder_model.vector_tensor.device,
                        dtype=cross_encoder_model.vector_tensor.dtype,
                    )
                else:
                    article_vector_tensor = article_vector.to(
                        device=cross_encoder_model.vector_tensor.device,
                        dtype=cross_encoder_model.vector_tensor.dtype,
                    )
            with torch.no_grad():
                cross_encoder_model.vector_tensor[base_n].copy_(article_vector_tensor.flatten())
    else:
        article_idx = article_network.article_key_to_idx[query_key]

    retrieval_start_time = time.time()

    if method == "caseaug":
        from src.methods.case_augmentation.prompt import generate_case
        case = generate_case(None, None, article)
        article = article+"\ncase:\n"+case

    # List to store results
    contradictions = []

    # Process top_k_articles in batches
    for i in range(0, len(top_k_articles), batch_size):
        batch_articles = top_k_articles[i:i+batch_size]

        if method == "caseaug":
            from src.methods.case_augmentation.prompt import generate_case
            original_articles = batch_articles
            batch_articles = [b+"\ncase:\n"+generate_case(None, None, b) for b in batch_articles]

        # Tokenize the batch of (article, batch_articles) pairs
        # NOTE: mapping stays per-batch (not hoisted): method == "caseaug"
        # rewrites batch texts mid-loop with cache side effects, so upfront
        # mapping would change behavior. Single device-side construction only.
        article2_idx_tensor = torch.tensor(
            [article_network.article_key_to_idx[article_key_function(retrieved_article)]
             for retrieved_article in batch_articles],
            dtype=torch.long, device=device)
        article1_idx_tensor = torch.full((len(batch_articles),), article_idx, dtype=torch.long, device=device)

        # Get model prediction for the batch
        with torch.no_grad():
            if index_method != "none":
                outputs = cross_encoder_model(article1_idx=article1_idx_tensor,article2_idx=article2_idx_tensor)
            logits = outputs.logits

            if logits.shape[1] == 1:
                # 한 개의 확률을 출력하는 경우 (예: [batch_size, 1])
                probs = torch.sigmoid(logits).cpu().numpy()  # Sigmoid로 변환하여 확률 얻기
            else:
                # 두 개의 확률을 출력하는 경우 (예: [batch_size, 2])
                probs = F.softmax(logits, dim=-1).cpu().numpy()  # Softmax로 변환하여 확률 얻기

        # Process the results for each article in the batch
        for idx, retrieved_article in enumerate(batch_articles):
            if logits.shape[1] == 1:
                # Sigmoid로 변환된 단일 값 (True일 확률)
                contradiction_score = probs[idx][0]
            else:
                # Softmax로 변환된 두 번째 값 (True일 확률)
                contradiction_score = probs[idx][1]  # 두 번째 값이 True일 확률
            if method == "caseaug":
                contradictions.append((original_articles[idx], contradiction_score))
            else:
                contradictions.append((retrieved_article, contradiction_score))

    # Sort contradictions based on the contradiction score in descending order
    contradictions = sorted(contradictions, key=lambda x: x[1], reverse=True)
    
    contradictions = [c for c in contradictions if c[1] >= 0.5]
    
    retrieval_end_time = time.time()

    elapsed_time = retrieval_end_time - retrieval_start_time
    # print(f"crossencoder 실행 시간: {elapsed_time:.6f}초")

    # Return the top_k articles with the highest contradiction scores
    return [c[0] for c in contradictions[:top_k]]
