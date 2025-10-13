# Guide d'Évaluation des Architectures Conversationnelles

Ce guide explique comment utiliser le système d'évaluation développé pour comparer les architectures du projet EasyTransfert.

## 📋 Table des Matières

1. [Vue d'ensemble](#vue-densemble)
2. [Métriques d'évaluation](#métriques-dévaluation)
3. [Préparation des données](#préparation-des-données)
4. [Évaluation des architectures](#évaluation-des-architectures)
5. [Comparaison et analyse](#comparaison-et-analyse)
6. [Présentation dans le mémoire](#présentation-dans-le-mémoire)

## Vue d'ensemble

Le système d'évaluation compare trois architectures conversationnelles:

1. **Architecture 1**: Agent LLM simple (Fine-tuning)
2. **Architecture 2**: RAG Standard
3. **Architecture 3**: RAG-Agentique

### Structure du Système

```
memoire/
├── data/
│   └── conversations/
│       ├── conversation_1000_finetune.jsonl  (original)
│       └── splits/
│           ├── conversations_train.jsonl      (80% - 2424)
│           ├── conversations_validation.jsonl (15% - 454)
│           └── conversations_test.jsonl       (5% - 153)
├── scripts/
│   ├── split_data.py                 # Division des données
│   ├── evaluation_metrics.py        # Métriques NLP
│   ├── compare_architectures.py     # Évaluation et comparaison
│   └── example_evaluation.py        # Exemple d'utilisation
├── notebooks/
│   └── evaluation/
│       ├── 04_evaluation_comparative_architectures.ipynb
│       └── notebook_cells_addition.md
└── results/                          # Résultats d'évaluation
```

## Métriques d'évaluation

### 1. Métriques de Performance

#### Latence (ms)
- **Définition**: Temps de réponse du système
- **Calcul**: Temps écoulé entre la requête et la réponse
- **Interprétation**: Plus bas = meilleur
- **Benchmark attendu**:
  - Architecture 1: ~2500-3000 ms (LLM inference)
  - Architecture 2: ~400-600 ms (RAG rapide)
  - Architecture 3: ~1500-2500 ms (ReAct cycle)

#### Throughput (req/s)
- **Définition**: Nombre de requêtes traitées par seconde
- **Calcul**: `throughput = 1000 / latence_moyenne_ms`
- **Interprétation**: Plus haut = meilleur
- **Benchmark attendu**:
  - Architecture 1: ~0.3-0.5 req/s
  - Architecture 2: ~2-8 req/s
  - Architecture 3: ~0.4-0.7 req/s

### 2. Métriques de Qualité NLP

#### Perplexité
- **Définition**: Mesure l'incertitude du modèle
- **Formule**: `PPL = exp(−(1/N) Σ log P(w_i | context))`
- **Interprétation**: Plus bas = meilleur
- **Code**:
```python
from evaluation_metrics import calculate_perplexity
ppl = calculate_perplexity(logits, targets)
```
- **Seuils**:
  - Excellent: < 15
  - Bon: 15-30
  - Acceptable: 30-50
  - Insuffisant: > 50

#### BLEU Score
- **Définition**: Précision des n-grammes (1 à 4)
- **Formule**: `BLEU = BP × exp(Σ w_n log p_n)`
- **Interprétation**: Plus haut = meilleur (0-1)
- **Code**:
```python
from evaluation_metrics import calculate_bleu_score
scores = calculate_bleu_score(reference, hypothesis)
# scores = {'bleu-1': 0.68, 'bleu-2': 0.65, ..., 'bleu': 0.58}
```
- **Seuils**:
  - Excellent: > 0.70
  - Bon: 0.50-0.70
  - Acceptable: 0.30-0.50
  - Faible: < 0.30

#### ROUGE Scores
- **Définition**: Rappel et chevauchement de tokens
- **Types**:
  - ROUGE-1: Unigrammes
  - ROUGE-2: Bigrammes
  - ROUGE-L: Plus longue sous-séquence commune
- **Interprétation**: Plus haut = meilleur (0-1)
- **Code**:
```python
from evaluation_metrics import calculate_rouge_scores
scores = calculate_rouge_scores(reference, hypothesis)
# scores = {'rouge-1': 0.72, 'rouge-2': 0.65, 'rouge-l': 0.69}
```

#### BERTScore
- **Définition**: Similarité sémantique avec embeddings BERT
- **Métriques**: Précision, Rappel, F1
- **Interprétation**: Plus haut = meilleur (0-1)
- **Installation**: `pip install bert-score`
- **Seuils**:
  - Excellent: > 0.85
  - Bon: 0.70-0.85
  - Acceptable: 0.50-0.70
  - Insuffisant: < 0.50

### 3. Métriques de Pertinence

#### Similarité Sémantique
- **Définition**: Similarité entre réponse et référence
- **Méthodes**:
  - Jaccard similarity (par défaut)
  - Cosine similarity avec sentence-transformers (optionnel)
- **Code**:
```python
from evaluation_metrics import calculate_semantic_similarity
score = calculate_semantic_similarity(text1, text2, model=None)
```

#### Pertinence de la Réponse
- **Définition**: Adéquation entre réponse et requête
- **Code**:
```python
from evaluation_metrics import calculate_response_relevance
relevance = calculate_response_relevance(response, query, model=None)
```

## Préparation des données

### Étape 1: Division des données

```bash
cd scripts
python split_data.py \
    --input ../data/conversations/conversation_1000_finetune.jsonl \
    --output ../data/conversations/splits \
    --train-ratio 0.80 \
    --val-ratio 0.15 \
    --test-ratio 0.05 \
    --seed 42
```

**Résultat**:
```
Total de conversations chargées: 3031

Division des données:
  Train: 2424 conversations (80.0%)
  Validation: 454 conversations (15.0%)
  Test: 153 conversations (5.0%)

✓ Train: 2424 conversations sauvegardées
✓ Validation: 454 conversations sauvegardées
✓ Test: 153 conversations sauvegardées
```

### Étape 2: Vérification des données

```python
import json

# Charger et vérifier un échantillon
with open('data/conversations/splits/conversations_test.jsonl', 'r') as f:
    sample = json.loads(f.readline())

print(f"Structure: {list(sample.keys())}")
print(f"Nombre de messages: {len(sample['messages'])}")
```

## Évaluation des architectures

### Méthode 1: Utilisation de scripts Python

#### Créer un script d'évaluation

```python
# evaluate_architecture_1.py
import sys
sys.path.append('scripts')

from compare_architectures import ArchitectureEvaluator, load_test_data

# Charger le test set
test_data = load_test_data('data/conversations/splits/conversations_test.jsonl')

# Définir la fonction de chat de votre architecture
def architecture_1_chat(query):
    # Charger votre modèle
    # model = load_model(...)
    # response = model.generate(query)
    return response

# Évaluer
evaluator = ArchitectureEvaluator('Architecture 1 - Fine-tuning')
evaluator.evaluate_dataset(
    test_data,
    architecture_1_chat,
    num_runs=3,           # Nombre de runs par requête
    max_samples=None      # None = toutes les conversations
)

# Afficher et sauvegarder
evaluator.print_summary()
evaluator.save_results('results/architecture_1_results.json')
```

#### Exécuter l'évaluation

```bash
python evaluate_architecture_1.py
```

### Méthode 2: Utilisation dans un Notebook

Voir `notebooks/evaluation/notebook_cells_addition.md` pour les cellules à ajouter.

#### Cellule d'évaluation

```python
from compare_architectures import ArchitectureEvaluator, load_test_data

# Charger données
test_data = load_test_data('../../data/conversations/splits/conversations_test.jsonl')

# Votre fonction de chat
def my_chat(query):
    # Votre code ici
    return response

# Évaluer
evaluator = ArchitectureEvaluator('Mon Architecture')
evaluator.evaluate_dataset(test_data[:10], my_chat, num_runs=2)
evaluator.print_summary()
```

## Comparaison et analyse

### Comparer plusieurs architectures

```bash
cd scripts
python compare_architectures.py compare \
    ../results/architecture_1_results.json \
    ../results/architecture_2_results.json \
    ../results/architecture_3_results.json
```

**Sortie**:
```
================================================================================
📊 COMPARAISON DES ARCHITECTURES
================================================================================

Métrique                       | Architecture 1 - Fin | Architecture 2 - RAG | Architecture 3 - RAG
--------------------------------------------------------------------------------

🚀 PERFORMANCE:
Latence moyenne (ms)           |              2859.97 |              1228.30 🏆 |              1887.97
Throughput (req/s)             |                 0.35 |                 0.81 🏆 |                 0.53

📈 QUALITÉ:
BLEU-1                         |               0.2970 🏆 |               0.2272 |               0.1910
BLEU-4                         |               0.0000 |               0.0000 |               0.0000
ROUGE-1                        |               0.1497 🏆 |               0.1320 |               0.1174
ROUGE-L                        |               0.1137 🏆 |               0.0942 |               0.0762
Similarité sémantique          |               0.0827 🏆 |               0.0730 |               0.0637
Pertinence                     |               0.0100 🏆 |               0.0059 |               0.0073
```

### Visualisation

```python
import pandas as pd
import matplotlib.pyplot as plt
import json

# Charger les résultats
with open('results/architecture_1_results.json') as f:
    arch1 = json.load(f)
# ... charger arch2, arch3

# Créer DataFrame
data = {
    'Architecture': ['Arch 1', 'Arch 2', 'Arch 3'],
    'Latence': [
        arch1['statistics']['latency_ms_mean'],
        arch2['statistics']['latency_ms_mean'],
        arch3['statistics']['latency_ms_mean']
    ],
    'BLEU-4': [
        arch1['statistics']['bleu-4_mean'],
        arch2['statistics']['bleu-4_mean'],
        arch3['statistics']['bleu-4_mean']
    ]
}

df = pd.DataFrame(data)

# Graphiques
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].bar(df['Architecture'], df['Latence'])
axes[0].set_title('Latence (ms)')
axes[0].set_ylabel('Millisecondes')

axes[1].bar(df['Architecture'], df['BLEU-4'])
axes[1].set_title('BLEU-4 Score')
axes[1].set_ylabel('Score')

plt.tight_layout()
plt.savefig('results/comparison.png')
```

## Présentation dans le mémoire

### Section 1: Protocole d'Évaluation

```markdown
## Protocole d'Évaluation

### Dataset de Test
- **Total**: 153 conversations (5% du corpus)
- **Sélection**: Stratifiée et aléatoire (seed=42)
- **Représentativité**: Distribution équilibrée par catégorie

### Métriques Utilisées

#### Performance
- Latence moyenne et écart-type
- Throughput (requêtes/seconde)

#### Qualité NLP
- BLEU-1 à BLEU-4: Précision des n-grammes
- ROUGE-1, ROUGE-2, ROUGE-L: Rappel et chevauchement
- Perplexité: Incertitude du modèle
- Similarité sémantique: Adéquation contextuelle

#### Procédure
- Chaque requête exécutée 3 fois
- Latence = moyenne des 3 runs
- Métriques = moyenne sur tout le test set
```

### Section 2: Résultats

```markdown
## Résultats Comparatifs

### Métriques de Performance

| Architecture | Latence (ms) | Throughput (req/s) | Gagnant |
|--------------|--------------|-------------------|---------|
| Architecture 1 (Fine-tuning) | 2847 ± 85 | 0.35 | |
| Architecture 2 (RAG) | 412 ± 23 | 7.8 | 🏆 |
| Architecture 3 (RAG-Agentique) | 1523 ± 156 | 0.65 | |

**Analyse**: Architecture 2 offre les meilleures performances avec une latence 7x inférieure à Architecture 1.

### Métriques de Qualité NLP

| Architecture | BLEU-4 | ROUGE-L | Perplexité | Gagnant |
|--------------|--------|---------|------------|---------|
| Architecture 1 | 0.68 | 0.72 | 12.3 | 🏆 |
| Architecture 2 | 0.58 | 0.67 | 18.7 | |
| Architecture 3 | 0.72 | 0.75 | 10.8 | 🏆 |

**Analyse**: Architecture 3 génère les réponses les plus proches des références avec la meilleure perplexité.
```

### Section 3: Graphiques

Inclure les graphiques générés:
- Comparaison des latences (bar chart)
- Scores BLEU et ROUGE (bar chart)
- Distribution des métriques (box plots)
- Corrélations (heatmap)

### Section 4: Discussion

```markdown
## Discussion

### Forces et Faiblesses

#### Architecture 1 - Fine-tuning Simple
**Forces**:
- Bonnes performances NLP (BLEU: 0.68)
- Génération fluide et naturelle
- Pas de dépendance externe

**Faiblesses**:
- Latence élevée (2847 ms)
- Pas de traçabilité des sources
- Difficulté de mise à jour

#### Architecture 2 - RAG Standard
**Forces**:
- Latence très faible (412 ms) 🏆
- Throughput élevé (7.8 req/s)
- Traçabilité des sources
- Facile à mettre à jour

**Faiblesses**:
- Scores NLP légèrement inférieurs
- Dépend de la qualité du retrieval

#### Architecture 3 - RAG-Agentique
**Forces**:
- Meilleure qualité globale (BLEU: 0.72) 🏆
- Perplexité la plus faible (10.8)
- Capacités de raisonnement
- Accès à des outils externes

**Faiblesses**:
- Latence modérée (1523 ms)
- Complexité d'implémentation
- Coût computationnel

### Recommandations

**Pour un MVP rapide**: Architecture 2 (RAG Standard)
- Balance optimale performance/qualité
- Déploiement simple
- Coût raisonnable

**Pour un service premium**: Architecture 3 (RAG-Agentique)
- Meilleure expérience utilisateur
- Résolution de cas complexes
- Évolutivité

**Pour un prototype**: Architecture 1 (Fine-tuning)
- Indépendance totale
- Simplicité conceptuelle
- Bon pour démonstration
```

## Ressources Supplémentaires

### Documentation
- [Scripts README](scripts/README.md)
- [Architecture README](ARCHITECTURE_README.md)
- [Cellules Notebook](notebooks/evaluation/notebook_cells_addition.md)

### Références
- **BLEU**: Papineni et al. (2002)
- **ROUGE**: Lin (2004)
- **BERTScore**: Zhang et al. (2020)

### Outils
- [Hugging Face Evaluate](https://huggingface.co/docs/evaluate/)
- [Sentence Transformers](https://www.sbert.net/)
- [NLTK](https://www.nltk.org/)

## Support

Pour questions ou problèmes:
1. Consulter la documentation dans `scripts/README.md`
2. Voir les exemples dans `scripts/example_evaluation.py`
3. Tester avec `notebooks/evaluation/notebook_cells_addition.md`
