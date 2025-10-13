# Scripts d'Évaluation des Architectures

Ce dossier contient les scripts pour évaluer et comparer les différentes architectures conversationnelles du projet EasyTransfert.

## 📁 Fichiers

### 1. `split_data.py`
Script pour diviser les données de conversation en ensembles train/validation/test.

**Usage:**
```bash
python split_data.py \
    --input data/conversations/conversation_1000_finetune.jsonl \
    --output data/conversations/splits \
    --train-ratio 0.80 \
    --val-ratio 0.15 \
    --test-ratio 0.05 \
    --seed 42
```

**Résultat:**
- `conversations_train.jsonl` (80% - 2424 conversations)
- `conversations_validation.jsonl` (15% - 454 conversations)
- `conversations_test.jsonl` (5% - 153 conversations)

### 2. `evaluation_metrics.py`
Module contenant toutes les métriques d'évaluation.

**Métriques implémentées:**
- **Perplexity**: Mesure l'incertitude du modèle (nécessite PyTorch)
- **BLEU Score** (BLEU-1 à BLEU-4): Similarité entre texte généré et référence
- **ROUGE Scores** (ROUGE-1, ROUGE-2, ROUGE-L): Rappel et chevauchement de tokens
- **F1 Score**: Moyenne harmonique de précision et rappel
- **Similarité sémantique**: Cosine similarity ou Jaccard similarity
- **Pertinence de la réponse**: Adéquation entre réponse et requête

**Usage dans du code:**
```python
from evaluation_metrics import evaluate_response

reference = "Bonjour, pour transférer de l'argent..."
hypothesis = "Salut, pour faire un transfert..."

metrics = evaluate_response(reference, hypothesis)
print(metrics)
# Output: {'bleu-1': 0.67, 'rouge-1': 0.63, ...}
```

### 3. `compare_architectures.py`
Script pour évaluer et comparer plusieurs architectures.

**Classes principales:**

#### `ArchitectureEvaluator`
Évalue une architecture sur un dataset de test.

```python
from compare_architectures import ArchitectureEvaluator, load_test_data

# Charger les données
test_data = load_test_data('data/conversations/splits/conversations_test.jsonl')

# Créer un évaluateur
evaluator = ArchitectureEvaluator('Architecture 1')

# Définir votre fonction de chat
def my_chat_function(query):
    # Votre logique ici
    return "Réponse du chatbot"

# Évaluer
evaluator.evaluate_dataset(test_data, my_chat_function, max_samples=10)
evaluator.print_summary()
evaluator.save_results('results/arch1_results.json')
```

#### `compare_architectures()`
Compare les résultats de plusieurs architectures.

```bash
python compare_architectures.py compare \
    results/architecture_1_results.json \
    results/architecture_2_results.json \
    results/architecture_3_results.json
```

### 4. `example_evaluation.py`
Exemple complet d'utilisation du système d'évaluation avec des architectures simulées.

**Usage:**
```bash
python example_evaluation.py
```

Ce script:
1. Charge les données de test
2. Évalue 3 architectures simulées sur 10 conversations
3. Affiche les résultats comparatifs
4. Sauvegarde les résultats dans `results/`

## 📊 Métriques d'Évaluation

### Métriques de Performance

| Métrique | Description | Unité | Meilleur |
|----------|-------------|-------|----------|
| **Latence** | Temps de réponse moyen | ms | Plus bas |
| **Throughput** | Nombre de requêtes par seconde | req/s | Plus haut |
| **Écart-type latence** | Variabilité du temps de réponse | ms | Plus bas |

### Métriques de Qualité NLP

| Métrique | Description | Plage | Meilleur |
|----------|-------------|-------|----------|
| **BLEU-1 à BLEU-4** | Précision des n-grammes (1 à 4) | 0-1 | Plus haut |
| **ROUGE-1** | Chevauchement des unigrammes | 0-1 | Plus haut |
| **ROUGE-2** | Chevauchement des bigrammes | 0-1 | Plus haut |
| **ROUGE-L** | Plus longue sous-séquence commune | 0-1 | Plus haut |
| **Similarité sémantique** | Similarité contextuelle | 0-1 | Plus haut |
| **Pertinence** | Adéquation réponse/requête | 0-1 | Plus haut |

### Interprétation des Scores

#### BLEU Score
- **> 0.70**: Excellent
- **0.50 - 0.70**: Bon
- **0.30 - 0.50**: Acceptable
- **< 0.30**: Faible

#### ROUGE Score
- **> 0.70**: Très bon
- **0.50 - 0.70**: Bon
- **0.30 - 0.50**: Moyen
- **< 0.30**: Faible

#### Similarité Sémantique
- **> 0.85**: Excellent
- **0.70 - 0.85**: Bon
- **0.50 - 0.70**: Acceptable
- **< 0.50**: Insuffisant

## 🚀 Utilisation Complète

### Étape 1: Préparer les Données

```bash
# Diviser les données en train/val/test
python split_data.py
```

### Étape 2: Évaluer Chaque Architecture

Créez un script pour chaque architecture:

```python
# evaluate_arch1.py
from compare_architectures import ArchitectureEvaluator, load_test_data

# Charger votre modèle
# model = load_your_model()

def chat_function(query):
    # Logique de votre architecture
    response = model.generate(query)
    return response

# Évaluer
test_data = load_test_data('data/conversations/splits/conversations_test.jsonl')
evaluator = ArchitectureEvaluator('Architecture 1')
evaluator.evaluate_dataset(test_data, chat_function, num_runs=3)
evaluator.print_summary()
evaluator.save_results('results/arch1_results.json')
```

### Étape 3: Comparer les Résultats

```bash
python compare_architectures.py compare \
    results/arch1_results.json \
    results/arch2_results.json \
    results/arch3_results.json
```

## 📝 Format des Résultats

Les résultats sont sauvegardés au format JSON:

```json
{
  "architecture": "Architecture 1",
  "num_samples": 153,
  "statistics": {
    "latency_ms_mean": 2847.5,
    "latency_ms_std": 85.2,
    "bleu-1_mean": 0.68,
    "bleu-4_mean": 0.56,
    "rouge-1_mean": 0.72,
    "rouge-l_mean": 0.69,
    "semantic_similarity_mean": 0.78,
    "relevance_mean": 0.82
  },
  "detailed_results": [
    {
      "query": "...",
      "reference": "...",
      "generated": "...",
      "latency_ms": 2845.3,
      "bleu-1": 0.72,
      ...
    }
  ]
}
```

## 🔧 Dépendances

**Minimales (pour métriques de base):**
```bash
pip install numpy
```

**Complètes (avec PyTorch et embeddings):**
```bash
pip install torch numpy sentence-transformers
```

## 📚 Références

- **BLEU**: Papineni et al. (2002) - "BLEU: a Method for Automatic Evaluation of Machine Translation"
- **ROUGE**: Lin (2004) - "ROUGE: A Package for Automatic Evaluation of Summaries"
- **BERTScore**: Zhang et al. (2020) - "BERTScore: Evaluating Text Generation with BERT"

## 🆘 Support

Pour des questions ou des problèmes, consultez:
- La documentation principale: `../ARCHITECTURE_README.md`
- Les notebooks d'exemples: `../notebooks/evaluation/`
- Le guide d'implémentation: `../IMPLEMENTATION_SUMMARY.md`
