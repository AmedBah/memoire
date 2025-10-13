"""
Module de métriques d'évaluation pour les architectures conversationnelles EasyTransfert.
"""
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter
import re


def calculate_perplexity(logits, targets) -> float:
    """
    Calcule la perplexité à partir des logits et des cibles.
    
    Args:
        logits: Tensor de forme (batch_size, seq_len, vocab_size)
        targets: Tensor de forme (batch_size, seq_len)
    
    Returns:
        float: Valeur de perplexité (plus bas = meilleur)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch n'est pas installé. Installez-le avec: pip install torch")
    
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    return torch.exp(loss).item()


def tokenize_text(text: str) -> List[str]:
    """
    Tokenise un texte en mots/tokens.
    
    Args:
        text: Texte à tokeniser
    
    Returns:
        Liste de tokens
    """
    # Normaliser et tokeniser
    text = text.lower()
    tokens = re.findall(r'\w+', text)
    return tokens


def compute_ngrams(tokens: List[str], n: int) -> Counter:
    """
    Calcule les n-grammes d'une liste de tokens.
    
    Args:
        tokens: Liste de tokens
        n: Taille des n-grammes
    
    Returns:
        Counter des n-grammes
    """
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        ngrams.append(ngram)
    return Counter(ngrams)


def calculate_bleu_score(reference: str, hypothesis: str, max_n: int = 4) -> Dict[str, float]:
    """
    Calcule le score BLEU entre une référence et une hypothèse.
    
    Args:
        reference: Texte de référence
        hypothesis: Texte généré (hypothèse)
        max_n: N-gramme maximum (défaut: 4 pour BLEU-4)
    
    Returns:
        Dictionnaire avec BLEU-1, BLEU-2, BLEU-3, BLEU-4 et score global
    """
    ref_tokens = tokenize_text(reference)
    hyp_tokens = tokenize_text(hypothesis)
    
    if len(hyp_tokens) == 0:
        return {f'bleu-{i}': 0.0 for i in range(1, max_n + 1)}
    
    # Calcul de la pénalité de brièveté (Brevity Penalty)
    ref_len = len(ref_tokens)
    hyp_len = len(hyp_tokens)
    
    if hyp_len > ref_len:
        bp = 1.0
    else:
        bp = np.exp(1 - ref_len / hyp_len) if hyp_len > 0 else 0.0
    
    # Calcul des précisions pour chaque n-gramme
    precisions = []
    scores = {}
    
    for n in range(1, max_n + 1):
        ref_ngrams = compute_ngrams(ref_tokens, n)
        hyp_ngrams = compute_ngrams(hyp_tokens, n)
        
        # Compter les correspondances
        matches = 0
        total = sum(hyp_ngrams.values())
        
        if total == 0:
            precision = 0.0
        else:
            for ngram, count in hyp_ngrams.items():
                matches += min(count, ref_ngrams.get(ngram, 0))
            precision = matches / total
        
        precisions.append(precision)
        scores[f'bleu-{n}'] = precision
    
    # Calcul du score BLEU global (moyenne géométrique)
    if all(p > 0 for p in precisions):
        geometric_mean = np.exp(np.mean([np.log(p) for p in precisions]))
        bleu_score = bp * geometric_mean
    else:
        bleu_score = 0.0
    
    scores['bleu'] = bleu_score
    
    return scores


def calculate_rouge_scores(reference: str, hypothesis: str) -> Dict[str, float]:
    """
    Calcule les scores ROUGE (ROUGE-1, ROUGE-2, ROUGE-L).
    
    Args:
        reference: Texte de référence
        hypothesis: Texte généré
    
    Returns:
        Dictionnaire avec les scores ROUGE
    """
    ref_tokens = tokenize_text(reference)
    hyp_tokens = tokenize_text(hypothesis)
    
    if len(hyp_tokens) == 0 or len(ref_tokens) == 0:
        return {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0}
    
    # ROUGE-1 (unigrammes)
    ref_unigrams = set(ref_tokens)
    hyp_unigrams = set(hyp_tokens)
    overlap_1 = len(ref_unigrams & hyp_unigrams)
    
    rouge_1_recall = overlap_1 / len(ref_unigrams) if len(ref_unigrams) > 0 else 0.0
    rouge_1_precision = overlap_1 / len(hyp_unigrams) if len(hyp_unigrams) > 0 else 0.0
    
    if rouge_1_precision + rouge_1_recall > 0:
        rouge_1_f1 = 2 * rouge_1_precision * rouge_1_recall / (rouge_1_precision + rouge_1_recall)
    else:
        rouge_1_f1 = 0.0
    
    # ROUGE-2 (bigrammes)
    ref_bigrams = compute_ngrams(ref_tokens, 2)
    hyp_bigrams = compute_ngrams(hyp_tokens, 2)
    
    overlap_2 = sum((ref_bigrams & hyp_bigrams).values())
    total_ref_2 = sum(ref_bigrams.values())
    total_hyp_2 = sum(hyp_bigrams.values())
    
    rouge_2_recall = overlap_2 / total_ref_2 if total_ref_2 > 0 else 0.0
    rouge_2_precision = overlap_2 / total_hyp_2 if total_hyp_2 > 0 else 0.0
    
    if rouge_2_precision + rouge_2_recall > 0:
        rouge_2_f1 = 2 * rouge_2_precision * rouge_2_recall / (rouge_2_precision + rouge_2_recall)
    else:
        rouge_2_f1 = 0.0
    
    # ROUGE-L (plus longue sous-séquence commune)
    def lcs_length(X, Y):
        """Calcule la longueur de la plus longue sous-séquence commune."""
        m, n = len(X), len(Y)
        L = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if X[i-1] == Y[j-1]:
                    L[i][j] = L[i-1][j-1] + 1
                else:
                    L[i][j] = max(L[i-1][j], L[i][j-1])
        
        return L[m][n]
    
    lcs_len = lcs_length(ref_tokens, hyp_tokens)
    rouge_l_recall = lcs_len / len(ref_tokens) if len(ref_tokens) > 0 else 0.0
    rouge_l_precision = lcs_len / len(hyp_tokens) if len(hyp_tokens) > 0 else 0.0
    
    if rouge_l_precision + rouge_l_recall > 0:
        rouge_l_f1 = 2 * rouge_l_precision * rouge_l_recall / (rouge_l_precision + rouge_l_recall)
    else:
        rouge_l_f1 = 0.0
    
    return {
        'rouge-1': rouge_1_f1,
        'rouge-2': rouge_2_f1,
        'rouge-l': rouge_l_f1
    }


def calculate_f1_score(precision: float, recall: float) -> float:
    """
    Calcule le score F1 (moyenne harmonique de précision et rappel).
    
    Args:
        precision: Score de précision
        recall: Score de rappel
    
    Returns:
        Score F1
    """
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def calculate_semantic_similarity(text1: str, text2: str, model=None) -> float:
    """
    Calcule la similarité sémantique entre deux textes.
    Si aucun modèle n'est fourni, utilise une similarité basique (Jaccard).
    
    Args:
        text1: Premier texte
        text2: Deuxième texte
        model: Modèle de sentence embeddings (optionnel)
    
    Returns:
        Score de similarité (0-1)
    """
    if model is not None:
        # Utiliser le modèle pour calculer les embeddings
        try:
            from sentence_transformers import util
            embedding1 = model.encode(text1, convert_to_tensor=True)
            embedding2 = model.encode(text2, convert_to_tensor=True)
            similarity = util.cos_sim(embedding1, embedding2).item()
            return similarity
        except Exception as e:
            print(f"Erreur lors du calcul de similarité avec modèle: {e}")
            # Fallback vers Jaccard
    
    # Similarité de Jaccard (fallback)
    tokens1 = set(tokenize_text(text1))
    tokens2 = set(tokenize_text(text2))
    
    if len(tokens1) == 0 and len(tokens2) == 0:
        return 1.0
    if len(tokens1) == 0 or len(tokens2) == 0:
        return 0.0
    
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    
    return intersection / union if union > 0 else 0.0


def evaluate_response(
    reference: str,
    hypothesis: str,
    semantic_model=None
) -> Dict[str, float]:
    """
    Évalue une réponse générée par rapport à une référence avec toutes les métriques.
    
    Args:
        reference: Texte de référence
        hypothesis: Texte généré
        semantic_model: Modèle pour similarité sémantique (optionnel)
    
    Returns:
        Dictionnaire contenant toutes les métriques
    """
    metrics = {}
    
    # BLEU scores
    bleu_scores = calculate_bleu_score(reference, hypothesis)
    metrics.update(bleu_scores)
    
    # ROUGE scores
    rouge_scores = calculate_rouge_scores(reference, hypothesis)
    metrics.update(rouge_scores)
    
    # Similarité sémantique
    semantic_sim = calculate_semantic_similarity(reference, hypothesis, semantic_model)
    metrics['semantic_similarity'] = semantic_sim
    
    return metrics


def calculate_response_relevance(response: str, query: str, semantic_model=None) -> float:
    """
    Calcule la pertinence d'une réponse par rapport à une requête.
    
    Args:
        response: Réponse générée
        query: Requête originale
        semantic_model: Modèle pour similarité sémantique (optionnel)
    
    Returns:
        Score de pertinence (0-1)
    """
    return calculate_semantic_similarity(response, query, semantic_model)


if __name__ == "__main__":
    # Tests basiques
    ref = "Bonjour, pour transférer de l'argent, veuillez suivre ces étapes."
    hyp = "Bonjour, pour envoyer de l'argent, suivez ces instructions."
    
    print("Test des métriques d'évaluation:")
    print(f"\nRéférence: {ref}")
    print(f"Hypothèse: {hyp}")
    
    metrics = evaluate_response(ref, hyp)
    
    print("\nMétriques calculées:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")
