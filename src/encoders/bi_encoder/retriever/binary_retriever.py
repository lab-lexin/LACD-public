import torch
from transformers import AutoTokenizer
import polars as pl
import chromadb

from torch.nn.functional import cosine_similarity
import numpy as np
from tqdm import tqdm

# from src.utils.encoder.utils import MAX_TOKEN_LENGTH
import time

MAX_TOKEN_LENGTH = 4096
# MAX_TOKEN_LENGTH = 512

# Populate-verified collections: (id(collection), n_expected) proven complete
# earlier in this process. force_rebuild wipe drops the entry; a different DB
# object or corpus size never hits it. Skips per-query to_list() + count().
_POPULATED_CACHE = {}

# our original code, bi-encoder.
def find_top_conflicts(article, model, tokenizer, chroma_collection, top_k=10, index_method = "none", max_length=None, fp16=False):
    """
    Function to find top conflicts using cosine similarity method with optional case augmentation.
    
    Args:
    article (str): The input article to check.
    model (torch.nn.Module): Pre-trained model.
    tokenizer (transformers.PreTrainedTokenizer): Tokenizer for the model.
    chroma_collection: Chroma DB collection containing the encoded laws.
    top_k (int): Number of top-k articles to retrieve.
    method (str): Method to use, "baseline" or "caseaug" for case augmentation.

    Returns:
    List of top-k articles that contradict the input article.
    """
    if top_k > 50000:
        top_k = 500

    # Encode the input article
    # max_length caps sequence length (None = legacy module cap), always bounded by tokenizer.model_max_length
    ml = min(max_length or MAX_TOKEN_LENGTH, tokenizer.model_max_length)
    inputs = tokenizer.encode_plus(
        article,
        add_special_tokens=True,
        max_length=ml,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"].to(model.encoder.device)
    attention_mask = inputs["attention_mask"].to(model.encoder.device)

    use_cuda = torch.cuda.is_available()
    with torch.inference_mode(), torch.autocast(device_type="cuda" if use_cuda else "cpu", dtype=torch.float16, enabled=bool(fp16) and use_cuda):
        encoded_article = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = encoded_article.last_hidden_state[0, 0, :].cpu().numpy()

    retrieval_start_time = time.time()

    results = chroma_collection.query(
        query_embeddings=[pooled_output.tolist()],
        n_results=top_k,
        # documents only: returned embeddings were deserialized then discarded.
        include=["documents"]
    )
    top_k_articles = results["documents"][0]


    retrieval_end_time = time.time()

    elapsed_time = retrieval_end_time - retrieval_start_time

    return pooled_output, top_k_articles


def _get_all_ids(chroma_collection, page_size=20000):
    """Fetch every id via limit/offset pagination: a single get() may truncate,
    which would make resume re-encode present docs (duplicate-add warnings)."""
    ids = []
    offset = 0
    while True:
        page = chroma_collection.get(include=[], limit=page_size, offset=offset)["ids"]
        if not page:
            break
        ids.extend(page)
        if len(page) < page_size:
            break
        offset += page_size
    return ids


def binary_retriever(
    model,
    laws_df,
    chroma_collection,
    article_to_check,
    top_k=500,
    batch_size=8,
    tokenizer=None,
    allowed_keys=None,
    article_network=None,
    max_length=None,
    fp16=False,
    force_rebuild=False,
    **kwargs
):
    """
    Bi-encoder retriever function to find conflicts in legal texts.

    Args:
        model: Pre-trained model module or str path to model directory.
        laws_df: Polars/Pandas DataFrame or str path to laws.csv.
        chroma_collection: Chroma DB collection object or str collection name.
        article_to_check: The article to check for conflicts.
        top_k: Number of top-k articles to retrieve.
        batch_size: Batch size for encoding if Chroma DB is empty.
        tokenizer: Tokenizer for the model.
        allowed_keys: Optional set of allowed article keys for filtering.
        article_network: Optional ArticleNetwork instance.
        max_length: Optional cap for tokenizer sequence length (None = legacy
            module cap MAX_TOKEN_LENGTH, always bounded by tokenizer.model_max_length).
        fp16: Enable fp16 autocast for encoding (CUDA only, no-op otherwise).
        force_rebuild: Wipe all existing docs and rebuild from scratch
            (for suspect content, e.g. corpus changed; partial DBs are
            otherwise resumed, not rebuilt).

    Returns:
        tuple (pooled_output, top_k_articles)
    """
    # 1. Resolve model & tokenizer
    if isinstance(model, str):
        model_path = model
        loaded_model = torch.load(model_path + "/model.pth", weights_only=False)
        loaded_model.eval()
        if tokenizer is None:
            global _TOKENIZER_CACHE
            try:
                _TOKENIZER_CACHE
            except NameError:
                _TOKENIZER_CACHE = {}  # type: ignore
            if model_path in _TOKENIZER_CACHE:
                tokenizer = _TOKENIZER_CACHE[model_path]
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, clean_up_tokenization_spaces=True)
                _TOKENIZER_CACHE[model_path] = tokenizer
        model = loaded_model

    # 2. Resolve laws_df using Polars
    global _LAWS_DF_CACHE, _FILTERED_DF_CACHE, _SUBSET_FILTER_LOGGED
    try:
        _LAWS_DF_CACHE
    except NameError:
        _LAWS_DF_CACHE = {}  # type: ignore
        _FILTERED_DF_CACHE = {}  # type: ignore
        _SUBSET_FILTER_LOGGED = False  # type: ignore

    if isinstance(laws_df, str):
        laws_csv = laws_df
        if laws_csv not in _LAWS_DF_CACHE:
            _LAWS_DF_CACHE[laws_csv] = pl.read_csv(laws_csv, infer_schema_length=10000)
        laws_df = _LAWS_DF_CACHE[laws_csv]

    # 3. Filter laws_df by allowed_keys / article_network if provided
    if article_network is not None and hasattr(article_network, 'all_article_keys'):
        allowed_keys = set(article_network.all_article_keys)
    if allowed_keys is not None:
        cache_key = id(article_network) if article_network is not None else hash(frozenset(allowed_keys))
        if cache_key in _FILTERED_DF_CACHE:
            laws_df = _FILTERED_DF_CACHE[cache_key]
        else:
            before = laws_df.height
            filtered_df = laws_df.filter(
                pl.col("article_title").str.replace_all("·", "ㆍ").is_in(list(allowed_keys))
            )
            _FILTERED_DF_CACHE[cache_key] = filtered_df
            laws_df = filtered_df
            if not _SUBSET_FILTER_LOGGED:
                _SUBSET_FILTER_LOGGED = True
                # Stay silent when nothing was filtered (full-corpus runs).
                if laws_df.height != before:
                    tqdm.write(f"[GRAPH] Chroma filter {before}->{laws_df.height} rows by allowed_keys ({len(allowed_keys)} keep)")

    # 4. Resolve chroma_collection
    if isinstance(chroma_collection, str):
        from src.utils.encoder.biencoder_utils import load_chromaDB_byname
        chroma_collection = load_chromaDB_byname(chroma_collection)

    # 5. Populate Chroma DB (streaming + crash-resume).
    # Each encoded chunk is added immediately so a crash no longer discards
    # prior GPU work. Ids are positional (str(idx)), so a restart encodes only
    # ids missing from the collection instead of silently proceeding with a
    # partial DB (the old count()==0 guard did the latter).
    if force_rebuild:
        _POPULATED_CACHE.pop((id(chroma_collection), laws_df.height), None)
        try:
            # include=[]: ids only, skip deserializing full documents.
            _wipe_ids = _get_all_ids(chroma_collection)
        except Exception:
            _wipe_ids = []
        if _wipe_ids:
            # Chroma caps one delete at 41,666 embeddings: wipe in batches.
            _wipe_ids = list(_wipe_ids)
            for _s in range(0, len(_wipe_ids), 10000):
                chroma_collection.delete(ids=_wipe_ids[_s:_s + 10000])
            tqdm.write(f"[REBUILD] wiped {len(_wipe_ids)} existing docs; rebuilding from scratch")
    n_expected = laws_df.height
    _pop_key = (id(chroma_collection), n_expected)
    if (not force_rebuild) and _POPULATED_CACHE.get(_pop_key, False):
        # Proven complete earlier in this process: skip to_list + count + build.
        missing, n_present, all_articles = [], n_expected, []
        _just_built = False
    else:
        all_articles = laws_df["contents"].to_list()
        existing_ids = set()
        try:
            n_present = chroma_collection.count()
        except Exception:
            n_present = 0
        if n_present > 0:
            # Short-circuit: same count as the deterministic positional selection
            # means complete (same seed+corpus => same id set); suspect content is
            # --force_rebuild's job, not resume's. Otherwise fetch ids only.
            if n_present == n_expected:
                missing = []
            else:
                try:
                    existing_ids = set(_get_all_ids(chroma_collection))
                    if len(existing_ids) != n_present:
                        tqdm.write(f"[WARN] id listing incomplete ({len(existing_ids)}/{n_present}); some present docs may re-encode")
                except Exception:
                    existing_ids = set()
                missing = [i for i in range(n_expected) if str(i) not in existing_ids]
        else:
            missing = list(range(n_expected))
        _just_built = False
        if n_expected > 0 and missing:
            _just_built = True
    if n_expected > 0 and not missing and n_present > n_expected:
        tqdm.write(f"[CHROMA] collection holds {n_present} docs for {n_expected} corpus rows; reusing as-is")
    if n_expected > 0 and missing:
        if existing_ids:
            tqdm.write(f"[RESUME] {n_expected - len(missing)}/{n_expected} docs present; encoding {len(missing)} missing")

        max_length = min(max_length or MAX_TOKEN_LENGTH, tokenizer.model_max_length) if tokenizer is not None else (max_length or MAX_TOKEN_LENGTH)
        use_cuda = torch.cuda.is_available()
        # fp16 autocast scope (CUDA only; no-op otherwise, never mutates the model)
        autocast_ctx = torch.autocast(device_type="cuda" if use_cuda else "cpu", dtype=torch.float16, enabled=bool(fp16) and use_cuda)

        # Adaptive batching with permanent step-down: on CUDA OOM, halve the
        # batch size globally (never scale back up) and re-split the failed
        # batch at the new size, so small-VRAM GPUs converge after ~1 OOM
        # instead of paying a synchronizing exception per batch.
        # OOM surfaces as torch.cuda.OutOfMemoryError (a RuntimeError subclass)
        # or plain RuntimeError on some builds, hence the message filter.
        enc_batch_size = max(1, batch_size)
        # D2H transfers and chroma adds are batched per chunk (not per encode
        # batch) to cut device synchronizations and SQLite commits; GPU/CPU
        # residency stays bounded (~48MB VRAM + ~500MB transient RAM) because
        # buffers flush every add. Chunk 16384 => ~4min rework on resume, ~5 adds full.
        chunk_size = 16384
        gpu_buf = []
        pending_docs = []
        pending_ids = []
        # Row counter (not len(gpu_buf): each entry is a variable-size batch,
        # so buffer length counts batches while the chunk budget is documents).
        pending_rows = 0
        # Stage 0 instrumentation: per-stage encode timings (decides whether
        # background tokenization threads are worth building later).
        t_tok = t_h2d = t_fwd = t_d2h = t_add = 0.0
        n_d2h = n_add = 0
        # Device-true timings (A): CUDA Events separate real device work from
        # host-side queue drain that perf_counter misattributes. Reused across
        # batches; CPU path keeps host timers only.
        _use_ev = use_cuda
        if _use_ev:
            _ev = {k: torch.cuda.Event(enable_timing=True) for k in ("h0", "h1", "f0", "f1", "d0", "d1")}
            t_h2d_ev = t_fwd_ev = t_d2h_ev = 0.0
        else:
            _ev = None
        # Length bucketing (DESCENDING): homogeneous lengths keep
        # padding="longest" batches tight (measured ~3.4x fewer forward tokens
        # on laws.csv), while peak-first ordering is allocator-friendly: the
        # largest blocks are cached up front and reused by shrinking batches,
        # so reserved memory stays flat instead of ballooning monotonically.
        # Triples (doc, embedding, id) keep their original positional ids, so
        # DB content and resume logic are order-independent.
        order = sorted(missing, key=lambda i: len(all_articles[i]), reverse=True)
        ptr = 0
        pbar = tqdm(total=len(order))
        while ptr < len(order):
            batch_idx = order[ptr:ptr + enc_batch_size]
            batch_articles = [all_articles[i] for i in batch_idx]
            _t0 = time.perf_counter()
            batch_inputs = tokenizer(
                batch_articles,
                add_special_tokens=True,
                max_length=max_length,
                padding="longest",
                pad_to_multiple_of=64,
                truncation=True,
                return_tensors="pt"
            )
            _t1 = time.perf_counter()
            if _use_ev:
                _ev["h0"].record()
            # B1: pinned staging removes the driver-side staging copy; with
            # pinned memory the non_blocking launch is genuinely async (the
            # consuming forward still syncs as needed, so semantics unchanged).
            input_ids = batch_inputs["input_ids"].pin_memory().to(model.encoder.device, non_blocking=True)
            attention_mask = batch_inputs["attention_mask"].pin_memory().to(model.encoder.device, non_blocking=True)
            if _use_ev:
                _ev["h1"].record()
                torch.cuda.synchronize()
                t_h2d_ev += _ev["h0"].elapsed_time(_ev["h1"]) / 1000.0
            _t2 = time.perf_counter()

            try:
                if _use_ev:
                    _ev["f0"].record()
                with torch.inference_mode(), autocast_ctx:
                    encoder_to_use = model.passage_encoder if hasattr(model, "passage_encoder") else model.encoder
                    encoded_articles = encoder_to_use(input_ids=input_ids, attention_mask=attention_mask)
                    # Compact copy of the CLS rows only: [:, 0, :] is a strided
                    # view pinning the full (B, L, H) output alive, so
                    # .contiguous() materializes just the rows we keep.
                    pooled_gpu = encoded_articles.last_hidden_state[:, 0, :].contiguous()
                if _use_ev:
                    _ev["f1"].record()
                    torch.cuda.synchronize()
                    t_fwd_ev += _ev["f0"].elapsed_time(_ev["f1"]) / 1000.0
                _t3 = time.perf_counter()
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                if enc_batch_size <= 1:
                    raise
                if use_cuda:
                    torch.cuda.empty_cache()
                enc_batch_size = max(1, enc_batch_size // 2)
                tqdm.write(f"[OOM] permanent step-down encode batch -> {enc_batch_size}, retrying (never scaling back up)")
                continue

            t_tok += _t1 - _t0
            t_h2d += _t2 - _t1
            t_fwd += _t3 - _t2
            gpu_buf.append(pooled_gpu)
            pending_docs.extend(batch_articles)
            pending_ids.extend(str(i) for i in batch_idx)
            pending_rows += len(batch_idx)
            ptr += len(batch_idx)
            # Flush (D2H + chroma add) per chunk or at the tail, then free all
            # stage buffers so RAM stays flat regardless of corpus size.
            if pending_rows >= chunk_size or ptr >= len(order):
                _t4 = time.perf_counter()
                # cat (not stack): buffered batches vary in size (tail partial
                # batch, OOM step-down), so equal-size stacking is wrong here.
                if _use_ev:
                    _ev["d0"].record()
                block = torch.cat(gpu_buf).cpu().numpy()
                if _use_ev:
                    _ev["d1"].record()
                    torch.cuda.synchronize()
                    t_d2h_ev += _ev["d0"].elapsed_time(_ev["d1"]) / 1000.0
                t_d2h += time.perf_counter() - _t4
                n_d2h += 1
                del gpu_buf[:]
                _t5 = time.perf_counter()
                chroma_collection.add(
                    documents=list(pending_docs),
                    embeddings=[e.tolist() for e in block],
                    ids=list(pending_ids),
                )
                # Reclaim allocator-held blocks each chunk: with shrinking batch
                # sizes ahead, cached giants would otherwise sit reserved.
                if use_cuda:
                    torch.cuda.empty_cache()
                t_add += time.perf_counter() - _t5
                n_add += 1
                del pending_docs[:]
                del pending_ids[:]
                pending_rows = 0
            pbar.update(len(batch_idx))
        pbar.close()
        t_sum = max(t_tok + t_h2d + t_fwd + t_d2h + t_add, 1e-9)
        tqdm.write(f"[ENCODE] tok={t_tok:.1f}s ({t_tok/t_sum:.0%}) h2d={t_h2d:.1f}s ({t_h2d/t_sum:.0%}) "
                   f"fwd={t_fwd:.1f}s ({t_fwd/t_sum:.0%}) d2h={t_d2h:.1f}s ({t_d2h/t_sum:.0%}, {n_d2h} transfers) "
                   f"add={t_add:.1f}s ({t_add/t_sum:.0%}, {n_add} chunks)")
        if _use_ev:
            tqdm.write(f"[ENCODE-device] h2d={t_h2d_ev:.1f}s fwd={t_fwd_ev:.1f}s d2h={t_d2h_ev:.1f}s "
                       f"(CUDA Events; host-vs-device gap = queue drain misattribution)")

    # Record proven-complete (reaching here means every missing id was added;
    # add() raises on failure, so partial states never get marked).
    if (not force_rebuild) and n_expected > 0 and (not missing or _just_built):
        _POPULATED_CACHE[_pop_key] = True

    # 6. Retrieve top conflicts
    article_vector, top_conflicts = find_top_conflicts(
        article_to_check, model, tokenizer, chroma_collection, top_k=top_k, max_length=max_length, fp16=fp16
    )

    return article_vector, top_conflicts


