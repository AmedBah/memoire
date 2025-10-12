# 📊 Évaluation Comparative des Architectures

Ce dossier contient les notebooks et ressources pour l'évaluation des trois architectures conversationnelles développées pour EasyTransfert.

## 📁 Contenu

### Notebooks

- **`04_evaluation_comparative_architectures.ipynb`** : Notebook principal d'évaluation
  - Description complète des métriques communes d'évaluation en français
  - Structure d'évaluation pour comparer les 3 architectures
  - Code d'implémentation pour calculer et visualiser les métriques
  - Recommandations pour la sélection de l'architecture optimale

## 🎯 Objectifs de l'Évaluation

Ce framework d'évaluation permet de :

1. **Quantifier les performances** de chaque architecture (1, 2, 3)
2. **Comparer objectivement** les différentes approches
3. **Identifier les forces et faiblesses** de chaque solution
4. **Guider la décision** sur l'architecture à déployer
5. **Établir des benchmarks** pour les améliorations futures

## 📋 Métriques d'Évaluation Couvertes

### 🔧 Métriques Techniques
- **Latence** : Temps de réponse (ms)
- **Throughput** : Requêtes par seconde
- **Utilisation mémoire** : RAM/VRAM
- **Coût computationnel** : Ressources CPU/GPU

### 🎯 Métriques de Qualité
- **Pertinence** : Score de similarité sémantique
- **Factualité** : Taux d'hallucinations
- **Complétude** : Couverture des informations
- **Traçabilité** : Capacité à citer sources
- **Cohérence** : Consistance des réponses
- **Adaptation linguistique** : Compréhension des expressions locales

### 💼 Métriques Métier
- **Taux de résolution** : Résolution au premier contact
- **Satisfaction client** : CSAT, NPS
- **Taux d'escalade** : Transferts vers agents humains
- **Temps de résolution** : Durée moyenne
- **Taux de containment** : Gestion autonome
- **ROI** : Retour sur investissement

## 🚀 Utilisation

### Prérequis

```bash
pip install numpy pandas matplotlib seaborn
pip install sentence-transformers bert-score
pip install scikit-learn
```

### Exécution

1. Ouvrir le notebook `04_evaluation_comparative_architectures.ipynb`
2. Lire la documentation des métriques (sections markdown)
3. Adapter les fonctions de chat pour chaque architecture
4. Exécuter les cellules d'évaluation
5. Analyser les résultats et visualisations

### Exemple d'utilisation

```python
from evaluation import ArchitectureEvaluator

# Créer l'évaluateur
evaluator = ArchitectureEvaluator("Architecture 1")

# Évaluer sur le dataset de test
metrics = evaluator.evaluate_dataset(test_queries, chat_function)

# Afficher les résultats
evaluator.print_summary(metrics)
```

## 📊 Benchmarks Attendus

| Métrique | Arch 1 | Arch 2 | Arch 3 | Objectif |
|----------|--------|--------|--------|----------|
| Latence moyenne | ~2-3s | ~3-5s | ~4-7s | < 5s |
| Throughput | ~0.3-0.5 req/s | ~0.2-0.3 req/s | ~0.15-0.25 req/s | > 0.2 req/s |
| Taux hallucination | 20-40% | 5-15% | 2-8% | < 10% |
| Taux résolution | 40-60% | 60-75% | 75-90% | > 70% |
| Traçabilité | 0% | 80-95% | 90-100% | > 80% |

## 🎓 Méthodologie

### Dataset de Test

Le notebook inclut un dataset de test standardisé avec :
- 100-200 requêtes couvrant tous les cas d'usage
- Distribution : 40% FAQ, 30% procédures, 20% complexe, 10% edge cases
- Catégories : informations générales, procédures, troubleshooting, vérification statut

### Protocole d'Évaluation

1. **Exécution** : Chaque requête testée sur les 3 architectures
2. **Répétition** : 3-5 runs pour stabilité des mesures
3. **Environnement** : Conditions identiques (hardware, modèles, données)
4. **Annotation** : Évaluation humaine pour métriques qualitatives
5. **Analyse** : Statistiques agrégées et visualisations comparatives

## 📈 Recommandations de Sélection

### Architecture 1 : Fine-tuning Simple
**✅ Choisir si :**
- Budget limité / POC rapide
- Requêtes simples et répétitives
- Besoin de latence ultra-faible

**❌ Ne pas choisir si :**
- Besoin de fiabilité critique
- Informations évolutives
- Traçabilité requise

### Architecture 2 : RAG Standard
**✅ Choisir si :**
- Balance performance/complexité
- Besoin de traçabilité
- Base de connaissances évolutive
- Réduction hallucinations

**❌ Ne pas choisir si :**
- Requêtes complexes multi-étapes
- Besoin d'accès données opérationnelles

### Architecture 3 : RAG-Agentique
**✅ Choisir si :**
- Service client complet
- Requêtes complexes
- Accès données opérationnelles
- Maximisation autonomie
- Budget infrastructure suffisant

**⚠️ Attention :**
- Nécessite infrastructure robuste
- Maintenance plus complexe

## 🔗 Liens Connexes

- **Documentation principale** : `ARCHITECTURE_README.md`
- **Architecture 1** : `notebooks/architecture_1/01_architecture_1_simple_agent_finetuning.ipynb`
- **Architecture 2** : `notebooks/architecture_2/02_architecture_2_rag_standard.ipynb`
- **Architecture 3** : `notebooks/architecture_3/03_architecture_3_rag_agentique.ipynb`
- **Exemples de données** : `notebooks/data_examples/data_sources_examples.ipynb`

## 📚 Ressources

### Métriques et Frameworks
- BLEU, ROUGE, BERTScore pour l'évaluation NLP
- RAGAS pour l'évaluation RAG
- Sentence-Transformers pour similarité sémantique

### Outils
- Weights & Biases : Tracking expérimentations
- MLflow : Gestion des expériences
- Hugging Face Evaluate : Bibliothèque de métriques

## 📞 Support

Pour toute question sur l'évaluation :
- **Documentation** : Voir `ARCHITECTURE_README.md`
- **Support EasyTransfert** : 2522018730 (WhatsApp 24h/24)
- **Issues GitHub** : [github.com/AmedBah/memoire/issues](https://github.com/AmedBah/memoire/issues)

---

*Framework d'évaluation développé pour le projet de mémoire EasyTransfert - KAYBIC AFRICA*
