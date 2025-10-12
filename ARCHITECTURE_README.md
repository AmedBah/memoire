# Architectures Expérimentales - Système Conversationnel EasyTransfert

Ce dépôt implémente trois architectures expérimentales pour un assistant conversationnel intelligent destiné au service client d'EasyTransfert, une application de transfert d'argent mobile en Côte d'Ivoire.

## 📋 Table des Matières

- [Vue d'ensemble](#vue-densemble)
- [🚀 Utilisation sur Google Colab Pro](#-utilisation-sur-google-colab-pro)
- [Architecture 1: Agent Simple (Baseline)](#architecture-1-agent-simple-baseline)
- [Architecture 2: RAG Standard](#architecture-2-rag-standard)
- [Architecture 3: RAG-Agentique](#architecture-3-rag-agentique)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Sources de Données](#sources-de-données)
- [Comparaison des Architectures](#comparaison-des-architectures)

## 🎯 Vue d'ensemble

Le projet vise à automatiser le service client d'EasyTransfert en explorant trois approches progressives :

1. **Architecture 1** : Fine-tuning d'un LLM (approche baseline)
2. **Architecture 2** : Retrieval-Augmented Generation (RAG)
3. **Architecture 3** : RAG avec capacités agentiques (ReAct)

### Contexte EasyTransfert

EasyTransfert permet des transferts d'argent inter-opérateurs entre :
- MTN Mobile Money
- Orange Money
- Moov Money
- Wave
- Trésor Money

## 🚀 Utilisation sur Google Colab Pro

**Tous les notebooks sont optimisés pour Google Colab Pro !**

### Accès Rapide

Chaque notebook dispose d'un badge "Open in Colab" pour un lancement en un clic :

| Architecture | Notebook | Badge Colab |
|-------------|----------|-------------|
| Architecture 1 | Fine-tuning LoRA | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AmedBah/memoire/blob/main/notebooks/architecture_1/01_architecture_1_simple_agent_finetuning.ipynb) |
| Architecture 2 | RAG Standard | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AmedBah/memoire/blob/main/notebooks/architecture_2/02_architecture_2_rag_standard.ipynb) |
| Architecture 3 | RAG-Agentique | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AmedBah/memoire/blob/main/notebooks/architecture_3/03_architecture_3_rag_agentique.ipynb) |

### Configuration Automatique

Les notebooks incluent automatiquement :
- ✅ **Détection d'environnement** : Colab vs Local
- ✅ **Vérification GPU** : Type et mémoire disponible
- ✅ **Montage Google Drive** : Pour données persistantes
- ✅ **Clonage du repository** : Accès automatique aux données
- ✅ **Chemins flexibles** : Fonctionnement sur Colab et en local

### Runtimes Recommandés

| Architecture | GPU Minimum | GPU Optimal | RAM | Durée |
|-------------|-------------|-------------|-----|-------|
| Architecture 1 | T4 (16 GB) | V100/A100 | High-RAM | 2-4h |
| Architecture 2 | T4 (16 GB) | T4/V100 | Standard | 30-60min |
| Architecture 3 | T4 (16 GB) | V100 | High-RAM | 1-2h |

### Guide Complet

Pour un guide détaillé sur l'utilisation de Colab Pro, consultez : **[COLAB_GUIDE.md](./COLAB_GUIDE.md)**

Le guide couvre :
- Configuration initiale et authentification
- Gestion des données (Drive vs clonage)
- Optimisation de la mémoire GPU
- Résolution de problèmes courants
- Bonnes pratiques et conseils

## 🏗️ Architecture 1: Agent Simple (Baseline)

### Description

Architecture de référence basée sur un modèle Llama 3.2 3B fine-tuné avec LoRA sur les conversations historiques d'EasyTransfert.

### Caractéristiques

- **Modèle** : Llama 3.2 3B Instruct
- **Technique** : LoRA (r=16, α=32, dropout=0.05)
- **Framework** : Unsloth pour entraînement optimisé
- **Données** : 3000+ conversations historiques
- **Mémoire** : ~50 MB (adaptateurs LoRA)

### Avantages

✅ Simplicité architecturale  
✅ Inférence rapide (~2-3s)  
✅ Faible empreinte mémoire  
✅ Déploiement facile  
✅ Style conversationnel cohérent  

### Limites

❌ Risque d'hallucinations  
❌ Nécessite réentraînement pour nouvelles données  
❌ Pas de traçabilité des sources  
❌ Connaissances limitées aux données d'entraînement  

### Notebook

```
notebooks/architecture_1/01_architecture_1_simple_agent_finetuning.ipynb
```

## 🔍 Architecture 2: RAG Standard

### Description

Système de Génération Augmentée par Récupération qui sépare la capacité de raisonnement (LLM) de la base de connaissances (ChromaDB).

### Caractéristiques

- **Base vectorielle** : ChromaDB
- **Embedding** : paraphrase-multilingual-mpnet-base-v2 (768 dims)
- **Chunking** : 512 tokens max, 50 tokens overlap
- **LLM** : Llama 3.2 3B (peut utiliser le modèle fine-tuné)
- **Sources** : FAQ, procédures, documentation opérateurs, conversations

### Flux RAG

#### Phase de Récupération (~150-200ms)
1. Vectorisation de la question utilisateur
2. Recherche de similarité cosinus dans ChromaDB
3. Sélection des top-k chunks (défaut: 3, score > 0.5)
4. Enrichissement du prompt avec contexte + métadonnées

#### Phase de Génération (~2-3s)
1. LLM reçoit le prompt augmenté
2. Génération conditionnée sur le contexte
3. Post-traitement avec citations des sources

### Avantages

✅ Véracité : réponses ancrées dans sources vérifiables  
✅ Traçabilité : citations des sources  
✅ Actualisation : ajout de documents sans réentraînement  
✅ Réduction drastique des hallucinations  
✅ Transparence avec scores de pertinence  

### Limites

❌ Latence plus élevée (~2-3.5s)  
❌ Complexité accrue (pipeline multi-composants)  
❌ Dépendances multiples  
❌ Pas de raisonnement multi-étapes  

### Notebook

```
notebooks/architecture_2/02_architecture_2_rag_standard.ipynb
```

## 🤖 Architecture 3: RAG-Agentique

### Description

Architecture la plus sophistiquée combinant RAG avec des capacités agentiques selon le paradigme ReAct (Reasoning + Acting).

### Caractéristiques

- **Framework** : LangChain + LangGraph
- **Paradigme** : ReAct (Thought-Action-Observation)
- **Base** : Tout de l'Architecture 2 + couche agentique
- **Capacités** : Raisonnement multi-étapes, planification, adaptation

### Toolbox (4 Outils Métier)

#### 1. RAG Retriever
- Recherche vectorielle dans ChromaDB
- Paramètres : query, top_k, filters
- Retour : Chunks avec métadonnées et scores

#### 2. Operator Info
- Consultation base PostgreSQL (simulée)
- Données : Formats identifiants, limites, frais, compatibilités
- Retour : Informations structurées par opérateur

#### 3. Entity Extractor
- Extraction via regex + règles métier
- Patterns : Identifiants EasyTransfert/opérateurs, téléphones, montants
- Retour : Dictionnaire d'entités par catégorie

#### 4. Conversation Memory
- Gestion historique conversationnel
- Actions : get, update, search
- Retour : Contexte et problèmes similaires passés

### Cycle ReAct

```
1. Thought  : Analyse et planification
2. Action   : Choix et invocation d'outil
3. Observation : Examen du résultat
4. (Répéter jusqu'à résolution)
5. Final Answer : Réponse utilisateur
```

### Avantages

✅ Tout de l'Architecture 2 +  
✅ Raisonnement multi-étapes et planification  
✅ Accès bases de données et APIs  
✅ Adaptation contextuelle et émotionnelle  
✅ Autonomie décisionnelle  
✅ Traçabilité complète (cycle ReAct visible)  

### Limites

❌ Latence plus élevée (~3-5s)  
❌ Complexité architecturale élevée  
❌ Consommation mémoire importante  
❌ Configuration plus délicate  

### Notebook

```
notebooks/architecture_3/03_architecture_3_rag_agentique.ipynb
```

## 📦 Installation

### Prérequis

- Python 3.10+
- CUDA 11.8+ (pour GPU, recommandé)
- 16 GB RAM minimum (32 GB recommandé)
- GPU avec 8 GB VRAM minimum (pour Llama 3.2 3B en 4-bit)

### Installation des Dépendances

```bash
# Cloner le dépôt
git clone https://github.com/AmedBah/memoire.git
cd memoire

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt
```

### Dépendances Principales

- **LLM & Fine-tuning** : transformers, unsloth, peft, bitsandbytes
- **RAG & Vectoriel** : chromadb, sentence-transformers, langchain
- **Agent** : langgraph, langchain-community
- **Utils** : pandas, numpy, wandb, jupyter

## 🚀 Utilisation

### Architecture 1 : Fine-tuning

```bash
jupyter notebook notebooks/architecture_1/01_architecture_1_simple_agent_finetuning.ipynb
```

**Étapes** :
1. Charger Llama 3.2 3B avec Unsloth
2. Configurer LoRA (r=16, α=32)
3. Préparer les données (conversation_1000_finetune.jsonl)
4. Entraîner avec SFTTrainer
5. Tester l'inférence

### Architecture 2 : RAG

```bash
jupyter notebook notebooks/architecture_2/02_architecture_2_rag_standard.ipynb
```

**Étapes** :
1. Initialiser ChromaDB
2. Charger modèle d'embedding (paraphrase-multilingual-mpnet-base-v2)
3. Créer chunks (512 tokens, overlap 50)
4. Vectoriser et indexer documents
5. Implémenter pipeline RAG (retrieval + generation)

### Architecture 3 : RAG-Agentique

```bash
jupyter notebook notebooks/architecture_3/03_architecture_3_rag_agentique.ipynb
```

**Étapes** :
1. Initialiser ChromaDB (comme Architecture 2)
2. Créer les 4 outils métier
3. Configurer LLM avec prompt ReAct
4. Créer l'agent avec LangChain
5. Tester le cycle ReAct

## 📊 Sources de Données

Notebook de démonstration :
```bash
jupyter notebook notebooks/data_examples/data_sources_examples.ipynb
```

### Types de Données

#### 1. Conversations Historiques
- **Fichier** : `conversation_1000_finetune.jsonl`
- **Format** : JSON Lines avec messages role-based
- **Usage** : Fine-tuning Architecture 1, exemples pour RAG

#### 2. FAQ EasyTransfert
- **Contenu** : Questions-réponses officielles
- **Catégories** : Général, opérateurs, utilisation, tarifs, problèmes
- **Usage** : Base de connaissances RAG

#### 3. Documentation Opérateurs
- **Contenu** : Formats identifiants, limites, frais, compatibilités
- **Opérateurs** : MTN, Orange, Moov, Wave, Trésor Money
- **Usage** : Outil Operator Info (Architecture 3)

#### 4. Procédures de Résolution
- **Contenu** : Guides étape par étape pour problèmes courants
- **Types** : Transaction échouée, mot de passe, erreur numéro
- **Usage** : Base de connaissances RAG

#### 5. Expressions Ivoiriennes
- **Contenu** : Abréviations et expressions locales
- **Usage** : Enrichissement linguistique, Entity Extractor

#### 6. Logs de Transactions
- **Format** : JSON avec métadonnées complètes
- **Usage** : Simulation vérification statut (Architecture 3)

## 📈 Comparaison des Architectures

| Critère | Architecture 1 | Architecture 2 | Architecture 3 |
|---------|---------------|---------------|---------------|
| **Latence** | ~2-3s | ~2-3.5s | ~3-5s |
| **Mémoire** | Faible (~50 MB) | Moyenne | Élevée |
| **Complexité** | Simple | Modérée | Élevée |
| **Hallucinations** | Élevé | Faible | Minimal |
| **Traçabilité** | Aucune | Citations | Complète |
| **Actualisation** | Réentraînement | Ajout docs | Ajout docs/outils |
| **Raisonnement** | Simple | Simple | Multi-étapes |
| **Accès données** | Non | ChromaDB | ChromaDB + APIs |
| **Adaptation** | Non | Non | Contextuelle |

### Recommandations d'Utilisation

#### Architecture 1
- ✅ POC rapide
- ✅ Ressources limitées
- ✅ Requêtes simples et répétitives
- ❌ Besoins de précision critique
- ❌ Informations évolutives

#### Architecture 2
- ✅ Balance performance/complexité
- ✅ Besoin de traçabilité
- ✅ Base de connaissances évolutive
- ✅ Réduction hallucinations
- ❌ Requêtes complexes multi-étapes

#### Architecture 3
- ✅ Service client complet
- ✅ Requêtes complexes
- ✅ Accès données opérationnelles
- ✅ Adaptation contextuelle
- ✅ Automatisation maximale
- ⚠️ Nécessite infrastructure robuste

## 🎓 Méthodologie d'Évaluation

### Métriques Techniques

- **Latence** : Temps de réponse (ms)
- **Throughput** : Requêtes/seconde
- **Mémoire** : Utilisation RAM/VRAM

### Métriques Qualité

- **Pertinence** : Score de pertinence des réponses
- **Factualité** : Taux d'hallucinations
- **Complétude** : Couverture des informations
- **Traçabilité** : Capacité à citer sources

### Métriques Métier

- **Résolution** : Taux de résolution au premier contact
- **Satisfaction** : Score de satisfaction client
- **Escalade** : Taux de transfert vers agent humain
- **Temps** : Temps moyen de résolution

## 📝 Format des Identifiants

### EasyTransfert
- **Format** : `EFB.XXXXXXXXX`
- **Exemple** : `EFB.ABC123456`

### Par Opérateur
- **MTN** : Chiffres uniquement (10 chiffres)
- **Orange** : `MP` + 10 chiffres
- **Moov** : `MRCH*` ou `CF*` + alphanumériques
- **Wave** : Variable, souvent `T` + chiffres
- **Trésor Money** : Format variable

## 🔧 Configuration

### Variables d'Environnement

Créer un fichier `.env` :

```env
# Modèle LLM
MODEL_NAME=meta-llama/Llama-3.2-3B-Instruct
MAX_SEQ_LENGTH=2048
LOAD_IN_4BIT=true

# ChromaDB
CHROMA_PATH=./chromadb_easytransfert
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-mpnet-base-v2

# RAG
CHUNK_SIZE=512
CHUNK_OVERLAP=50
TOP_K=3
MIN_SIMILARITY_SCORE=0.5

# Agent
MAX_ITERATIONS=5
AGENT_VERBOSE=true

# Monitoring
WANDB_PROJECT=easytransfert-architectures
WANDB_ENTITY=your-entity
```

## 📞 Support

Pour toute question ou problème :

- **Issues GitHub** : [github.com/AmedBah/memoire/issues](https://github.com/AmedBah/memoire/issues)
- **Documentation** : Voir `doc.txt.txt` pour détails complets
- **Contact EasyTransfert** : 2522018730 (WhatsApp 24h/24)

## 📄 Licence

Ce projet est développé dans le cadre d'un mémoire de recherche pour KAYBIC AFRICA / EasyTransfert.

## 🙏 Remerciements

- KAYBIC AFRICA et l'équipe EasyTransfert
- Unsloth pour les outils d'optimisation
- LangChain pour le framework agentique
- Hugging Face pour les modèles et outils

---

**Auteur** : Amed Bah  
**Organisation** : KAYBIC AFRICA  
**Projet** : Système Conversationnel Intelligent EasyTransfert  
**Date** : 2024-2025
