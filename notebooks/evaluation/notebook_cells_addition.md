# Cellules à Ajouter au Notebook d'Évaluation

Ces cellules intègrent le système d'évaluation complet développé dans `scripts/`.

## Cellule 1: Importation des Scripts d'Évaluation

```python
# Import des modules d'évaluation développés
import sys
sys.path.append('../../scripts')

from evaluation_metrics import (
    calculate_perplexity,
    calculate_bleu_score,
    calculate_rouge_scores,
    calculate_f1_score,
    calculate_semantic_similarity,
    evaluate_response,
    calculate_response_relevance
)

from compare_architectures import (
    ArchitectureEvaluator,
    load_test_data,
    compare_architectures
)

print("✓ Modules d'évaluation importés avec succès")
```

## Cellule 2: Chargement des Données de Test

```python
# Charger les données de test depuis le split
test_data_path = '../../data/conversations/splits/conversations_test.jsonl'

try:
    test_conversations = load_test_data(test_data_path)
    print(f"✓ {len(test_conversations)} conversations de test chargées")
    
    # Afficher un exemple
    print("\nExemple de conversation:")
    example = test_conversations[0]
    for msg in example['messages'][:4]:
        print(f"  {msg['role']}: {msg['content'][:80]}...")
    
except FileNotFoundError:
    print("⚠️  Les données de test n'existent pas encore.")
    print("Exécutez d'abord: python ../../scripts/split_data.py")
```

## Cellule 3: Fonction calculate_perplexity

```python
import torch
import torch.nn.functional as F

def calculate_perplexity(logits, targets):
    """
    Calcule la perplexité à partir des logits et des cibles.
    
    Args:
        logits: Tensor de forme (batch_size, seq_len, vocab_size)
        targets: Tensor de forme (batch_size, seq_len)
    
    Returns:
        float: Valeur de perplexité (plus bas = meilleur)
    
    Exemple d'utilisation:
        # Avec un modèle PyTorch
        outputs = model(input_ids, labels=labels)
        logits = outputs.logits
        perplexity = calculate_perplexity(logits, labels)
    """
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    return torch.exp(loss).item()

# Test avec données fictives
if torch.cuda.is_available():
    # Simuler des logits et targets
    vocab_size = 50000
    seq_len = 20
    batch_size = 4
    
    logits = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    perplexity = calculate_perplexity(logits, targets)
    print(f"Exemple de perplexité: {perplexity:.2f}")
    print("(Plus bas = meilleur, valeurs acceptables: < 30)")
else:
    print("⚠️  GPU non disponible pour le test de perplexité")
```

## Cellule 4: Exemple d'Évaluation BLEU, ROUGE, F1

```python
# Exemple d'évaluation avec les métriques NLP

reference = """
Bonjour, Service client EasyTransfert 🤗. 
Pour transférer de l'argent de MTN vers Orange, suivez ces étapes:
1. Connectez-vous à l'application EasyTransfert
2. Sélectionnez 'Nouveau transfert'
3. Choisissez MTN comme source et Orange comme destination
4. Entrez le numéro du destinataire
5. Indiquez le montant
6. Confirmez la transaction
Les frais sont de 2% avec un minimum de 100 FCFA.
"""

hypothesis = """
Salut! Pour faire un transfert MTN vers Orange:
1. Ouvrez l'app EasyTransfert
2. Cliquez sur 'Transfert'
3. Sélectionnez MTN et Orange
4. Entrez le numéro et le montant
5. Validez
Les frais: 2% (min 100 FCFA).
"""

# Calculer toutes les métriques
metrics = evaluate_response(reference, hypothesis)

print("📊 MÉTRIQUES D'ÉVALUATION")
print("="*60)
print("\n🎯 Scores BLEU (précision des n-grams):")
for n in range(1, 5):
    score = metrics.get(f'bleu-{n}', 0)
    interpretation = "Excellent" if score > 0.7 else "Bon" if score > 0.5 else "Acceptable" if score > 0.3 else "Faible"
    print(f"  BLEU-{n}: {score:.4f} ({interpretation})")

print(f"\n📝 Scores ROUGE (rappel et chevauchement):")
print(f"  ROUGE-1: {metrics['rouge-1']:.4f}")
print(f"  ROUGE-2: {metrics['rouge-2']:.4f}")
print(f"  ROUGE-L: {metrics['rouge-l']:.4f}")

print(f"\n🔗 Similarité sémantique:")
print(f"  Score: {metrics['semantic_similarity']:.4f}")

print("\n" + "="*60)
```

## Cellule 5: Évaluation avec BERTScore (Optionnel)

```python
# BERTScore nécessite la bibliothèque bert-score
try:
    from bert_score import score as bert_score
    
    def calculate_bertscore(references, candidates, lang='fr'):
        """
        Calcule BERTScore entre références et candidats.
        
        Args:
            references: Liste de textes de référence
            candidates: Liste de textes générés
            lang: Langue ('fr' pour français)
        
        Returns:
            Tuple (precision, recall, f1)
        """
        P, R, F1 = bert_score(candidates, references, lang=lang, verbose=False)
        return P.mean().item(), R.mean().item(), F1.mean().item()
    
    # Test
    refs = [reference]
    hyps = [hypothesis]
    
    precision, recall, f1 = calculate_bertscore(refs, hyps)
    
    print("🎓 BERTScore (utilise BERT pour embeddings contextuels):")
    print(f"  Précision: {precision:.4f}")
    print(f"  Rappel: {recall:.4f}")
    print(f"  F1: {f1:.4f}")
    print("\nInterprétation:")
    print("  > 0.85: Excellent")
    print("  0.70-0.85: Bon")
    print("  0.50-0.70: Acceptable")
    print("  < 0.50: Insuffisant")
    
except ImportError:
    print("⚠️  BERTScore n'est pas installé.")
    print("Pour l'installer: !pip install bert-score")
```

## Cellule 6: Évaluation Complète d'une Architecture

```python
# Exemple d'évaluation complète d'une architecture sur le test set

def example_chat_function(query):
    """Fonction de chat simulée - remplacer par votre modèle réel"""
    import time
    import random
    
    time.sleep(random.uniform(0.5, 1.5))  # Simuler latence
    
    return "Bonjour, je suis l'assistant EasyTransfert. Comment puis-je vous aider?"

# Créer un évaluateur
evaluator = ArchitectureEvaluator("Architecture 1 - Exemple")

# Évaluer sur un sous-ensemble (pour test rapide)
print("🔍 Évaluation en cours...")
evaluator.evaluate_dataset(
    test_conversations[:5],  # Seulement 5 pour l'exemple
    example_chat_function,
    num_runs=2
)

# Afficher le résumé
evaluator.print_summary()

# Sauvegarder les résultats
evaluator.save_results('../../results/example_arch_results.json')
```

## Cellule 7: Comparaison Visuelle des Architectures

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Créer des données de comparaison fictives (remplacer par vos résultats réels)
comparison_data = {
    'Architecture': ['Fine-tuning\nSimple', 'RAG\nStandard', 'RAG-\nAgentique'],
    'Latence (ms)': [2847, 412, 1523],
    'Throughput (req/s)': [0.35, 7.8, 0.65],
    'BLEU-4': [0.68, 0.58, 0.72],
    'ROUGE-L': [0.72, 0.67, 0.75],
    'Perplexité': [12.3, 18.7, 10.8],
    'Similarité Sémantique': [0.78, 0.71, 0.82]
}

df = pd.DataFrame(comparison_data)

# Configuration du style
sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Comparaison des Architectures - Métriques Clés', fontsize=16, fontweight='bold')

# 1. Latence
ax = axes[0, 0]
bars = ax.bar(df['Architecture'], df['Latence (ms)'], color=['#3498db', '#2ecc71', '#e74c3c'])
ax.set_ylabel('Latence (ms)', fontweight='bold')
ax.set_title('Latence Moyenne\n(Plus bas = meilleur)')
for i, v in enumerate(df['Latence (ms)']):
    ax.text(i, v + 100, f'{v}ms', ha='center', fontweight='bold')

# 2. Throughput
ax = axes[0, 1]
bars = ax.bar(df['Architecture'], df['Throughput (req/s)'], color=['#3498db', '#2ecc71', '#e74c3c'])
ax.set_ylabel('Throughput (req/s)', fontweight='bold')
ax.set_title('Débit\n(Plus haut = meilleur)')
for i, v in enumerate(df['Throughput (req/s)']):
    ax.text(i, v + 0.2, f'{v}', ha='center', fontweight='bold')

# 3. BLEU-4
ax = axes[0, 2]
bars = ax.bar(df['Architecture'], df['BLEU-4'], color=['#3498db', '#2ecc71', '#e74c3c'])
ax.set_ylabel('Score BLEU-4', fontweight='bold')
ax.set_title('BLEU-4\n(Plus haut = meilleur)')
ax.set_ylim([0, 1])
for i, v in enumerate(df['BLEU-4']):
    ax.text(i, v + 0.02, f'{v:.2f}', ha='center', fontweight='bold')

# 4. ROUGE-L
ax = axes[1, 0]
bars = ax.bar(df['Architecture'], df['ROUGE-L'], color=['#3498db', '#2ecc71', '#e74c3c'])
ax.set_ylabel('Score ROUGE-L', fontweight='bold')
ax.set_title('ROUGE-L\n(Plus haut = meilleur)')
ax.set_ylim([0, 1])
for i, v in enumerate(df['ROUGE-L']):
    ax.text(i, v + 0.02, f'{v:.2f}', ha='center', fontweight='bold')

# 5. Perplexité
ax = axes[1, 1]
bars = ax.bar(df['Architecture'], df['Perplexité'], color=['#3498db', '#2ecc71', '#e74c3c'])
ax.set_ylabel('Perplexité', fontweight='bold')
ax.set_title('Perplexité\n(Plus bas = meilleur)')
for i, v in enumerate(df['Perplexité']):
    ax.text(i, v + 0.5, f'{v:.1f}', ha='center', fontweight='bold')

# 6. Similarité Sémantique
ax = axes[1, 2]
bars = ax.bar(df['Architecture'], df['Similarité Sémantique'], color=['#3498db', '#2ecc71', '#e74c3c'])
ax.set_ylabel('Score', fontweight='bold')
ax.set_title('Similarité Sémantique\n(Plus haut = meilleur)')
ax.set_ylim([0, 1])
for i, v in enumerate(df['Similarité Sémantique']):
    ax.text(i, v + 0.02, f'{v:.2f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('../../results/architectures_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Graphique sauvegardé dans results/architectures_comparison.png")
```

## Cellule 8: Tableau Récapitulatif

```python
# Créer un tableau récapitulatif formaté

from IPython.display import display, HTML

def create_comparison_table(df):
    """Crée un tableau HTML formaté avec les résultats"""
    
    # Identifier les meilleurs scores pour chaque métrique
    best_latency = df['Latence (ms)'].min()
    best_throughput = df['Throughput (req/s)'].max()
    best_bleu = df['BLEU-4'].max()
    best_rouge = df['ROUGE-L'].max()
    best_perplexity = df['Perplexité'].min()
    best_semantic = df['Similarité Sémantique'].max()
    
    html = """
    <style>
        .comparison-table {
            border-collapse: collapse;
            width: 100%;
            font-family: Arial, sans-serif;
        }
        .comparison-table th {
            background-color: #3498db;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: bold;
        }
        .comparison-table td {
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }
        .comparison-table tr:hover {
            background-color: #f5f5f5;
        }
        .best-score {
            background-color: #d4edda;
            font-weight: bold;
        }
        .trophy {
            color: #ffd700;
        }
    </style>
    
    <table class="comparison-table">
        <thead>
            <tr>
                <th>Architecture</th>
                <th>Latence (ms) ↓</th>
                <th>Throughput (req/s) ↑</th>
                <th>BLEU-4 ↑</th>
                <th>ROUGE-L ↑</th>
                <th>Perplexité ↓</th>
                <th>Similarité ↑</th>
            </tr>
        </thead>
        <tbody>
    """
    
    for _, row in df.iterrows():
        arch = row['Architecture'].replace('\n', ' ')
        
        # Formater chaque cellule avec mise en évidence du meilleur
        lat_class = 'best-score' if row['Latence (ms)'] == best_latency else ''
        thr_class = 'best-score' if row['Throughput (req/s)'] == best_throughput else ''
        bleu_class = 'best-score' if row['BLEU-4'] == best_bleu else ''
        rouge_class = 'best-score' if row['ROUGE-L'] == best_rouge else ''
        perp_class = 'best-score' if row['Perplexité'] == best_perplexity else ''
        sem_class = 'best-score' if row['Similarité Sémantique'] == best_semantic else ''
        
        trophy = '<span class="trophy">🏆</span>'
        
        html += f"""
            <tr>
                <td><strong>{arch}</strong></td>
                <td class="{lat_class}">{row['Latence (ms)']:.0f} {trophy if lat_class else ''}</td>
                <td class="{thr_class}">{row['Throughput (req/s)']:.2f} {trophy if thr_class else ''}</td>
                <td class="{bleu_class}">{row['BLEU-4']:.3f} {trophy if bleu_class else ''}</td>
                <td class="{rouge_class}">{row['ROUGE-L']:.3f} {trophy if rouge_class else ''}</td>
                <td class="{perp_class}">{row['Perplexité']:.1f} {trophy if perp_class else ''}</td>
                <td class="{sem_class}">{row['Similarité Sémantique']:.3f} {trophy if sem_class else ''}</td>
            </tr>
        """
    
    html += """
        </tbody>
    </table>
    <p style="margin-top: 10px; font-size: 12px; color: #666;">
        ↑ = Plus haut est meilleur | ↓ = Plus bas est meilleur | 🏆 = Meilleur score
    </p>
    """
    
    return html

# Afficher le tableau
display(HTML(create_comparison_table(df)))
```

## Notes d'Utilisation

1. **Exécuter les cellules dans l'ordre** pour assurer que toutes les dépendances sont chargées
2. **Adapter les fonctions de chat** dans la cellule 6 pour vos architectures réelles
3. **Remplacer les données fictives** dans les cellules 7 et 8 par vos résultats réels
4. **Utiliser le test set complet** en production (pas seulement 5 conversations)
5. **Sauvegarder les résultats** pour comparaisons futures

## Métriques Recommandées par Architecture

### Architecture 1 (Fine-tuning Simple)
- Focus: BLEU, ROUGE, Perplexité
- Latence élevée attendue (LLM inference)

### Architecture 2 (RAG Standard)  
- Focus: Précision factuelle, traçabilité
- Latence moyenne (retrieval + generation)

### Architecture 3 (RAG-Agentique)
- Focus: Complétude, précision, résolution complexe
- Latence variable selon complexité
