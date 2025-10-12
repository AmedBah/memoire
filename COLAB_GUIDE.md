# 🚀 Guide d'Utilisation sur Google Colab Pro

Ce guide vous explique comment exécuter les notebooks d'architectures expérimentales sur Google Colab Pro.

## 📋 Table des Matières

1. [Prérequis](#prérequis)
2. [Configuration Initiale](#configuration-initiale)
3. [Lancement des Notebooks](#lancement-des-notebooks)
4. [Choix du Runtime](#choix-du-runtime)
5. [Gestion des Données](#gestion-des-données)
6. [Optimisation pour Colab Pro](#optimisation-pour-colab-pro)
7. [Résolution de Problèmes](#résolution-de-problèmes)
8. [Conseils et Bonnes Pratiques](#conseils-et-bonnes-pratiques)

---

## 🎯 Prérequis

### Compte Google Colab Pro

**Pourquoi Colab Pro ?**
- GPU plus puissants (V100, A100 vs T4 gratuit)
- Plus de RAM (jusqu'à 32 GB vs 12 GB)
- Sessions plus longues (24h vs 12h)
- Priorité d'accès aux ressources
- Essentiel pour l'entraînement de modèles 3B

**Tarification:**
- Colab Pro: ~$9.99/mois
- Colab Pro+: ~$49.99/mois (recommandé pour Architecture 1)

### Authentification HuggingFace

Les modèles Llama 3.2 nécessitent une authentification:

1. Créer un compte sur [HuggingFace](https://huggingface.co/)
2. Accepter les conditions d'utilisation de [Llama 3.2](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
3. Créer un token d'accès: [Settings > Access Tokens](https://huggingface.co/settings/tokens)
4. Conserver le token pour l'authentification dans les notebooks

---

## ⚙️ Configuration Initiale

### 1. Ouvrir un Notebook

Chaque notebook dispose d'un badge "Open in Colab" :

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AmedBah/memoire/blob/main/notebooks/architecture_1/01_architecture_1_simple_agent_finetuning.ipynb)

**Ou bien:**
1. Aller sur [Google Colab](https://colab.research.google.com/)
2. File > Open notebook > GitHub
3. Entrer: `AmedBah/memoire`
4. Sélectionner le notebook désiré

### 2. Vérifier le Runtime

**Cellule de vérification automatique** (déjà incluse dans les notebooks):
```python
# Détection automatique de l'environnement
IS_COLAB = 'google.colab' in sys.modules
```

Cette cellule affiche:
- ✓ Type d'environnement (Colab ou local)
- ✓ GPU disponible et sa mémoire
- ⚠️ Avertissements si configuration inadéquate

---

## 🎮 Lancement des Notebooks

### Architecture 1: Agent Simple (Fine-tuning)

**Notebook:** `01_architecture_1_simple_agent_finetuning.ipynb`

**Runtime recommandé:**
- GPU: V100 (16 GB) ou A100 (40 GB)
- RAM: High-RAM (32 GB)
- Durée estimée: 2-4 heures

**Étapes:**
1. Cliquer sur le badge Colab ou ouvrir le notebook
2. Runtime > Change runtime type > GPU (V100 ou mieux)
3. Exécuter les cellules de configuration Colab
4. Authentifier HuggingFace quand demandé
5. Lancer l'entraînement

**Données utilisées:**
- `data/conversations/conversation_1000_finetune.jsonl`
- `data/expressions/expressions_ivoiriennes.json`

---

### Architecture 2: RAG Standard

**Notebook:** `02_architecture_2_rag_standard.ipynb`

**Runtime recommandé:**
- GPU: T4 (16 GB) suffisant
- RAM: Standard (12 GB) ou High-RAM
- Durée estimée: 30-60 minutes

**Étapes:**
1. Ouvrir le notebook sur Colab
2. Runtime > Change runtime type > GPU
3. Exécuter les cellules de configuration
4. Initialiser ChromaDB
5. Indexer les données
6. Tester les requêtes RAG

**Données utilisées:**
- `data/faqs/faq_easytransfert.json`
- `data/operators/operators_info.json`
- `data/procedures/procedures_resolution.json`
- `data/documents/doc.txt.txt`

---

### Architecture 3: RAG-Agentique

**Notebook:** `03_architecture_3_rag_agentique.ipynb`

**Runtime recommandé:**
- GPU: V100 (16 GB) recommandé
- RAM: High-RAM (25 GB)
- Durée estimée: 1-2 heures

**Étapes:**
1. Ouvrir le notebook sur Colab
2. Runtime > Change runtime type > GPU (V100)
3. Exécuter les cellules de configuration
4. Charger tous les outils (RAG Retriever, Operator Info, etc.)
5. Initialiser l'agent ReAct
6. Tester les scénarios

**Données utilisées:**
- Toutes les données des Architectures 1 et 2
- `data/documents/transaction_logs_sample.json`

---

## 🎛️ Choix du Runtime

### Comment changer le runtime ?

1. Menu: **Runtime > Change runtime type**
2. Choisir les paramètres:
   - **Hardware accelerator**: GPU
   - **GPU type** (Colab Pro uniquement):
     - Standard: T4 (16 GB VRAM)
     - Premium: V100 (16 GB) ou A100 (40 GB)
   - **Runtime shape**:
     - Standard: 12 GB RAM
     - High-RAM: 25-32 GB RAM

### Recommandations par Architecture

| Architecture | GPU Minimum | GPU Recommandé | RAM | Durée |
|-------------|-------------|----------------|-----|-------|
| Architecture 1 | T4 (16 GB) | V100/A100 | High-RAM | 2-4h |
| Architecture 2 | T4 (16 GB) | T4/V100 | Standard | 30-60min |
| Architecture 3 | T4 (16 GB) | V100 | High-RAM | 1-2h |

### Vérification GPU

Après avoir changé le runtime, exécutez:
```python
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Mémoire: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

---

## 💾 Gestion des Données

### Option 1: Clonage Automatique du Repository (Recommandé)

**Déjà configuré dans les notebooks!**

```python
# Automatique dans la cellule de configuration
!git clone https://github.com/AmedBah/memoire.git /content/memoire
DATA_DIR = '/content/memoire/data'
```

**Avantages:**
- ✅ Simple et automatique
- ✅ Pas besoin de Google Drive
- ✅ Données toujours à jour

**Inconvénients:**
- ⚠️ Re-téléchargement à chaque session (~6 MB)
- ⚠️ Perdu si session se termine

---

### Option 2: Google Drive (Pour Résultats Persistants)

**Pour sauvegarder modèles et résultats:**

```python
# Monter Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Copier les données dans Drive (une seule fois)
!cp -r /content/memoire/data /content/drive/MyDrive/memoire/

# Utiliser Drive comme répertoire de données
DATA_DIR = '/content/drive/MyDrive/memoire/data'
```

**Avantages:**
- ✅ Données persistantes entre sessions
- ✅ Sauvegarde des modèles entraînés
- ✅ Partage facile

**Inconvénients:**
- ⚠️ Plus lent (accès réseau)
- ⚠️ Nécessite espace Drive

---

### Structure des Données

```
data/
├── conversations/          # 6.4 MB - Données conversationnelles
├── documents/             # 65 KB - Documentation et logs
├── faqs/                  # ~3 KB - Questions-réponses
├── operators/             # ~5 KB - Infos opérateurs
├── procedures/            # ~4 KB - Procédures résolution
└── expressions/           # ~2 KB - Expressions ivoiriennes
```

**Total:** ~6.5 MB (téléchargement rapide)

---

## ⚡ Optimisation pour Colab Pro

### 1. Gérer les Timeouts

**Sessions Colab:**
- Gratuit: 12 heures max
- Pro: 24 heures max
- Pro+: 24 heures avec moins d'interruptions

**Pour les entraînements longs:**

```python
# Sauvegarder régulièrement les checkpoints
training_args = TrainingArguments(
    output_dir="/content/drive/MyDrive/memoire/checkpoints",
    save_strategy="steps",
    save_steps=500,  # Sauvegarder tous les 500 steps
    save_total_limit=3,  # Garder seulement les 3 derniers
)
```

---

### 2. Optimiser la Mémoire GPU

**Pour éviter les OOM (Out of Memory):**

```python
# Architecture 1: Réduire batch_size si nécessaire
per_device_train_batch_size = 2  # Au lieu de 4
gradient_accumulation_steps = 8  # Compenser

# Architecture 2: Réduire chunk_size
chunk_size = 256  # Au lieu de 512

# Architecture 3: Charger le modèle en 4-bit
load_in_4bit = True
```

---

### 3. Monitoring avec Weights & Biases

**Déjà intégré dans les notebooks:**

```python
import wandb

# Se connecter (première fois)
wandb.login()

# Tracking automatique
wandb.init(project="easytransfert-architectures")
```

**Avantages:**
- 📊 Visualisation en temps réel
- 📈 Courbes d'entraînement
- 💾 Sauvegarde automatique des métriques
- 🔗 Accès depuis n'importe où

---

### 4. Utiliser les GPU Efficacement

**Nettoyer la mémoire entre exécutions:**

```python
import torch
import gc

# Libérer la mémoire GPU
torch.cuda.empty_cache()
gc.collect()
```

**Vérifier l'utilisation:**

```python
# Pendant l'entraînement
print(f"Mémoire utilisée: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Mémoire réservée: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

---

## 🔧 Résolution de Problèmes

### Erreur: "Runtime disconnected"

**Causes:**
- Session inactive trop longtemps
- Dépassement du quota GPU
- Erreur dans le code

**Solutions:**
1. Runtime > Reconnect
2. Réexécuter les cellules depuis le début
3. Vérifier les logs pour erreurs

---

### Erreur: "Out of Memory (OOM)"

**Message:** `CUDA out of memory`

**Solutions:**

```python
# 1. Réduire batch_size
per_device_train_batch_size = 1
gradient_accumulation_steps = 16

# 2. Utiliser gradient checkpointing
use_gradient_checkpointing = True

# 3. Réduire max_seq_length
max_seq_length = 1024  # Au lieu de 2048

# 4. Activer quantification
load_in_4bit = True
```

---

### Erreur: "No GPU available"

**Cause:** Runtime CPU sélectionné

**Solution:**
1. Runtime > Change runtime type
2. Hardware accelerator > GPU
3. Save
4. Runtime > Restart runtime

---

### Erreur: "Cannot access HuggingFace model"

**Message:** `GatedRepoError` ou `401 Unauthorized`

**Solutions:**
1. Accepter les conditions: [Llama 3.2](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
2. Créer un token: [Settings > Tokens](https://huggingface.co/settings/tokens)
3. Authentifier:
   ```python
   from huggingface_hub import login
   login(token="your_token_here")
   ```

---

### Erreur: "Module not found"

**Cause:** Dépendance manquante

**Solution:**

```python
# Installer la dépendance
!pip install -q package_name

# Ou réinstaller toutes les dépendances
!pip install -q -r /content/memoire/requirements.txt
```

---

## 💡 Conseils et Bonnes Pratiques

### 1. Gestion des Sessions

✅ **À faire:**
- Sauvegarder régulièrement (Drive ou W&B)
- Utiliser des checkpoints
- Exécuter cellule par cellule pour déboguer
- Vérifier GPU avant entraînement long

❌ **À éviter:**
- Laisser session inactive (timeout)
- Entraîner sans sauvegardes
- Ignorer les warnings de mémoire
- Exécuter plusieurs notebooks en parallèle

---

### 2. Optimisation des Coûts

**Colab Pro ($9.99/mois):**
- Architecture 2 et 3 fonctionnent bien
- Architecture 1 possible mais plus lent

**Colab Pro+ ($49.99/mois):**
- Recommandé pour Architecture 1 (fine-tuning)
- A100 GPU pour entraînement rapide
- Sessions plus longues et stables

**Astuce:** Commencer avec Pro, passer à Pro+ si nécessaire

---

### 3. Organisation du Travail

**Structure recommandée dans Drive:**
```
MyDrive/
└── memoire/
    ├── data/                # Données (copié une fois)
    ├── checkpoints/         # Checkpoints d'entraînement
    ├── models/              # Modèles finaux
    ├── outputs/             # Résultats et logs
    └── notebooks/           # Notebooks modifiés (optionnel)
```

---

### 4. Workflow Recommandé

**Session Typique:**

1. **Préparation (5 min)**
   - Ouvrir notebook
   - Vérifier runtime (GPU, RAM)
   - Monter Drive si nécessaire
   - Cloner repository

2. **Exécution (30 min - 4h)**
   - Exécuter cellules de configuration
   - Lancer entraînement/expériences
   - Monitorer avec W&B

3. **Sauvegarde (5 min)**
   - Sauvegarder modèle dans Drive
   - Exporter métriques
   - Télécharger résultats importants

4. **Nettoyage**
   - Arrêter runtime si terminé
   - Libérer ressources pour autres utilisateurs

---

## 📊 Métriques de Performance Attendues

### Architecture 1 (Fine-tuning)

**Avec V100 (Colab Pro):**
- Temps par epoch: ~30-45 minutes
- Epochs recommandés: 3-5
- Durée totale: 2-3 heures
- Mémoire GPU: ~12-14 GB

**Avec A100 (Colab Pro+):**
- Temps par epoch: ~15-20 minutes
- Durée totale: 1-1.5 heures
- Mémoire GPU: ~12-14 GB

---

### Architecture 2 (RAG)

**Avec T4:**
- Indexation: ~5-10 minutes
- Requête RAG: ~2-3 secondes
- Mémoire GPU: ~6-8 GB

---

### Architecture 3 (RAG-Agentique)

**Avec V100:**
- Configuration: ~10-15 minutes
- Cycle ReAct: ~3-5 secondes
- Mémoire GPU: ~8-10 GB

---

## 🎓 Ressources Supplémentaires

### Documentation
- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)
- [Colab Pro Features](https://colab.research.google.com/signup)
- [HuggingFace Authentication](https://huggingface.co/docs/hub/security-tokens)
- [Weights & Biases Guide](https://docs.wandb.ai/guides)

### Communauté
- [GitHub Repository](https://github.com/AmedBah/memoire)
- [Issues](https://github.com/AmedBah/memoire/issues)
- Support EasyTransfert: 2522018730

---

## 📞 Support

Pour toute question ou problème:

1. Vérifier ce guide
2. Consulter les [Issues GitHub](https://github.com/AmedBah/memoire/issues)
3. Créer une nouvelle issue si nécessaire
4. Contacter: support@easytransfert.ci

---

**Bon entraînement sur Colab Pro! 🚀**

*Dernière mise à jour: 2024-10-12*
