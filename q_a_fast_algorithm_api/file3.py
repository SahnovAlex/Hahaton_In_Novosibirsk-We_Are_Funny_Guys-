import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Tuple

# ---------- String similarity with fallbacks ----------
try:
    import Levenshtein as _lev_mod
    def str_ratio(a, b): return _lev_mod.ratio(a, b)
except Exception:
    try:
        from rapidfuzz import fuzz as _rf_fuzz
        def str_ratio(a, b): return _rf_fuzz.ratio(a, b) / 100.0
    except Exception:
        import difflib as _difflib
        def str_ratio(a, b): return _difflib.SequenceMatcher(None, a, b).ratio()

# ==============================
# Config
# ==============================
DIRECT_ANSWER_MIN = 0.68
CLARIFY_MIN = 0.42
DUPLICATE_SIM_THR = 0.85
TOP_K_CANDIDATES = 50
DOC_FREQ_MIN = 15
SHORT_QUERY_MAX_TOKENS = 3

# ==============================
# Load KB
# ==============================
FILE_CANDIDATES = [
    'Q&A&L_expanded_v4_refined_no_understand_nodup.xlsx',
    'Q&A&L_expanded_v4_refined_no_understand.xlsx',
    'Q&A&L_expanded_v4_refined_cleaned.xlsx',
    'Q&A&L_expanded_v4_refined.xlsx',
]
for _p in FILE_CANDIDATES:
    try:
        df = pd.read_excel(_p)
        print(f"Loaded KB from: {_p} ({len(df)} rows)")
        break
    except Exception:
        df = None
if df is None:
    raise FileNotFoundError("Не найден Excel с базой знаний.")

questions = df['question'].astype(str).tolist()
answers = df['answer'].astype(str).tolist()
sources = df['source'].astype(str).tolist()

# ==============================
# Text processing
# ==============================
RU_STOPWORDS = {
    'что','как','для','на','по','из','от','за','при','и','или','но','не','в','с','у','о','а','же','ли','бы','то',
    'какие','какой','какая','когда','где','сколько','чем','чему','кто','чье','чья','чьи','есть','быть','этого','этот','эта',
    'зачем','почему','можно','нужно','пожалуйста','подскажите','подскажи','скажите','сообщите','это','этим','этом','того'
}

def preprocess(text: str) -> str:
    s = str(text).lower()
    s = s.replace('ё', 'е')
    s = re.sub(r'[^0-9a-zа-я\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def alpha_token_count(s: str) -> int:
    return sum(1 for t in s.split() if re.search(r'[а-яa-z]', t))

processed_questions = [preprocess(q) for q in questions]

# ------------------------------
# Build domain terms (for OOD + spell correction)
# ------------------------------
from collections import Counter
doc_freq = Counter()
for q in processed_questions:
    toks = set(w for w in q.split() if w not in RU_STOPWORDS and len(w) > 2)
    doc_freq.update(toks)
DOMAIN_TERMS = {t for t, c in doc_freq.items() if c >= DOC_FREQ_MIN}

# ==============================
# Опечатки — корректор
# ==============================
def correct_spelling(text, term_dict, min_ratio=0.8):
    tokens = text.split()
    corrected = []
    for w in tokens:
        if w in term_dict or len(w) < 3:
            corrected.append(w)
            continue
        best_term, best_score = None, 0
        for term in term_dict:
            score = str_ratio(w, term)
            if score > best_score:
                best_score, best_term = score, term
        corrected.append(best_term if best_score >= min_ratio else w)
    return " ".join(corrected)

# ==============================
# Vectorizer
# ==============================
vectorizer = TfidfVectorizer(ngram_range=(1,2), token_pattern=r'(?u)\b\w+\b', max_features=20000)
tfidf_matrix = vectorizer.fit_transform(processed_questions)

# ==============================
# Helpers
# ==============================
def domain_overlap_stats(text_proc: str):
    toks = [w for w in text_proc.split() if w not in RU_STOPWORDS and len(w) > 2]
    overlap = [w for w in toks if w in DOMAIN_TERMS]
    return len(toks), len(overlap)

def clean_candidates(indices, processed_questions, max_variants=3, duplicate_thr=DUPLICATE_SIM_THR):
    selected = []
    for idx in indices:
        txt = processed_questions[idx]
        if alpha_token_count(txt) < 2:
            continue
        is_dup = any(str_ratio(txt, processed_questions[j]) >= duplicate_thr for j in selected)
        if not is_dup:
            selected.append(idx)
        if len(selected) >= max_variants:
            break
    return selected

# ==============================
# Retrieval
# ==============================
def find_best_matches(user_question: str, top_k=TOP_K_CANDIDATES):
    proc = preprocess(user_question)
    # 🔹 исправляем опечатки
    proc_corr = correct_spelling(proc, DOMAIN_TERMS, min_ratio=0.8)
    user_vec = vectorizer.transform([proc_corr])
    cos = cosine_similarity(user_vec, tfidf_matrix).flatten()
    k = min(top_k, len(cos))
    order = np.argsort(cos)[-k:][::-1]

    scored = []
    for idx in order:
        lev = str_ratio(proc_corr, processed_questions[idx])
        score = 0.75 * cos[idx] + 0.25 * lev
        scored.append((idx, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored

# ==============================
# CLI loop
# ==============================
print("Система поддержки пользователей по 223-ФЗ. Для выхода введите 'exit'.")

current_context = None
history = []

def handle_user_input(user_input: str, current_context: dict, history: list) -> Tuple[Dict, dict, list]:
    history.append(("user", user_input))
    
    if user_input.strip().lower() == 'exit':
        return {
            "success": False,
            "message": "Диалог завершен",
            "extra": ""
            }, current_context, history

    # Обработка выбора из текущего контекста
    if current_context and user_input.isdigit():
        num = int(user_input)
        if 1 <= num <= len(current_context['options']):
            idx = current_context['options'][num-1]
            result = {
                "success": True,
                "message": answers[idx],
                "extra": sources[idx]
            }
            current_context = None
            history.append(("system", f"answer_idx={idx}"))
            return result, current_context, history

    # Поиск лучших совпадений
    scored = find_best_matches(user_input)
    if not scored:
        current_context = None
        return {
            "success": False,
            "message": "Извините, я не нашёл подходящего ответа в базе знаний",
            "extra": ""
            }, current_context, history

    best_idx, best_score = scored[0]
    proc_corr = correct_spelling(preprocess(user_input), DOMAIN_TERMS)
    _, overlap_kw = domain_overlap_stats(proc_corr)
    short_query = len([w for w in proc_corr.split() if w not in RU_STOPWORDS]) <= SHORT_QUERY_MAX_TOKENS
    min_required = 1 if short_query else 2
    has_domain_overlap = overlap_kw >= min_required

    if has_domain_overlap and best_score >= DIRECT_ANSWER_MIN:
        current_context = None
        return {
            "success": True,
            "message": answers[best_idx],
            "extra": sources[best_idx]
            }, current_context, history

    if has_domain_overlap:
        indices_sorted = [i for i, s in scored if s >= CLARIFY_MIN]
        cleaned = clean_candidates(indices_sorted, processed_questions, max_variants=3)
        if len(cleaned) >= 2:
            idx = cleaned[0]
            current_context = None
            return {
                "success": True,
                "message": answers[idx],
                "extra": sources[idx]
            }, current_context, history

    current_context = None
    return {
        "success": False,
        "message": "Похоже, вопрос вне тематики базы знаний или сформулирован слишком общо.\nПожалуйста, переформулируйте вопрос или уточните термины",
        "extra": ""
    }, current_context, history

print("\nСпасибо за использование системы!")
