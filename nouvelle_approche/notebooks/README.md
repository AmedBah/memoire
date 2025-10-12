# Notebooks - Architectures Conversationnelles EasyTransfert

## 📚 Vue d'ensemble

Ce dossier contient deux notebooks Jupyter exécutables sur Google Colab, implémentant et comparant deux approches d'intelligence artificielle conversationnelle pour l'automatisation du service client d'EasyTransfert.

## 🎯 Notebooks disponibles

### 1. Architecture 1 : Agent LLM (Llama 3.2 + LoRA)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AmedBah/memoire/blob/main/nouvelle_approche/notebooks/Architecture_1_Agent_LLM.ipynb)

**Fichier**: `Architecture_1_Agent_LLM.ipynb`

**Approche**: Agent conversationnel end-to-end basé sur un Large Language Model

**Caractéristiques**:
- Modèle: Llama 3.2 3B Instruct
- Technique: LoRA (Low-Rank Adaptation) - r=16, alpha=32
- Framework: Unsloth (optimisé pour fine-tuning rapide)
- Données: 3031 conversations réelles EasyTransfert

**Contenu du notebook**:
1. Configuration environnement et vérification GPU
2. Installation automatique des dépendances
3. Chargement et prétraitement des données
   - Nettoyage
   - Anonymisation RGPD
   - Normalisation code-switching français-nouchi
4. Configuration modèle Llama 3.2 + adaptateurs LoRA
5. Fine-tuning supervisé
6. Évaluation (métriques techniques + métier)
7. Tests interactifs

**Prérequis GPU**: T4 (16GB) minimum, V100/A100 recommandé

---

### 2. Architecture 2 : Deep Learning + NLP Modulaire

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AmedBah/memoire/blob/main/nouvelle_approche/notebooks/Architecture_2_DL_NLP.ipynb)

**Fichier**: `Architecture_2_DL_NLP.ipynb`

**Approche**: Système modulaire avec composants NLP spécialisés

**Architecture**:
1. **Classification d'intentions** (BiLSTM + Attention)
2. **Named Entity Recognition** (BiLSTM-CRF)
3. **Analyse de sentiment** (CamemBERT)
4. **Dialogue State Tracking**
5. **Génération de réponses** (Templates + Retrieval + Seq2Seq)

**Contenu du notebook**:
1. Configuration environnement (CPU/GPU)
2. Installation automatique des dépendances
3. Chargement et prétraitement des données
4. Implémentation des 5 modules spécialisés
5. Pipeline complet intégré
6. Évaluation par module et globale
7. Comparaison avec Architecture 1

**Avantage**: Fonctionne sur CPU (pas de GPU obligatoire)

---

## 🚀 Guide d'utilisation

### Exécution sur Google Colab (Recommandé)

1. **Cliquez sur le badge "Open in Colab"** dans le notebook souhaité
2. **Connectez-vous** à votre compte Google
3. **Activez le GPU** (pour Architecture 1):
   - Menu: `Runtime` > `Change runtime type`
   - Hardware accelerator: `GPU`
   - GPU type: `T4` (gratuit) ou `V100/A100` (Colab Pro)
4. **Exécutez les cellules** dans l'ordre (`Ctrl+Enter` ou `Runtime > Run all`)
5. Le notebook va automatiquement:
   - Installer toutes les dépendances nécessaires
   - Cloner le repository et charger les données
   - Exécuter le pipeline complet

### Temps d'exécution estimés

| Architecture | GPU | Temps total |
|--------------|-----|-------------|
| Architecture 1 (Agent LLM) | T4 (16GB) | 4-5 heures |
| Architecture 1 (Agent LLM) | V100 (16GB) | 2-3 heures |
| Architecture 1 (Agent LLM) | A100 (40GB) | 1-1.5 heures |
| Architecture 2 (DL + NLP) | CPU | 2-3 heures |
| Architecture 2 (DL + NLP) | GPU | 1-2 heures |

### Données requises

Les notebooks téléchargent automatiquement les données depuis le repository GitHub:
- `conversation_1000_finetune.jsonl` (3031 conversations, ~5.2 MB)

Aucune action manuelle requise pour les données.

---

## 📊 Résultats attendus

### Sections d'évaluation

Chaque notebook contient des sections d'évaluation avec des **placeholders** pour vos résultats:

**Métriques techniques**:
- BLEU-4
- ROUGE-L F1
- Perplexité
- Latence moyenne
- Throughput (requêtes/seconde)

**Métriques métier**:
- Taux de résolution
- Taux d'hallucination
- Net Promoter Score (NPS)
- Coût d'inférence

### Actions requises après exécution

1. **Remplacer les résultats placeholder** par vos mesures réelles
2. **Ajouter les captures d'écran** des courbes d'entraînement
3. **Compléter l'analyse qualitative** avec des exemples de conversations
4. **Documenter les observations** pendant l'entraînement

---

## 🔧 Dépannage

### Problème: "CUDA out of memory"

**Solution**:
- Réduire `per_device_train_batch_size` à 1
- Augmenter `gradient_accumulation_steps`
- Utiliser GPU avec plus de mémoire (Colab Pro)

### Problème: Installation échoue

**Solution**:
- Redémarrer le runtime: `Runtime > Restart runtime`
- Vérifier la connexion internet
- Réexécuter la cellule d'installation

### Problème: Données introuvables

**Solution**:
- Vérifier que le clone du repository a réussi
- Exécuter manuellement:
  ```bash
  !git clone https://github.com/AmedBah/memoire.git
  %cd memoire
  !ls conversation_1000_finetune.jsonl
  ```

---

## 📚 Références académiques

Les notebooks incluent des citations complètes pour:

### Architecture 1 (Agent LLM)
1. Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022.
2. Touvron et al. (2023). Llama 2: Open Foundation and Fine-Tuned Chat Models. arXiv:2307.09288.
3. Meta AI (2024). Llama 3.2: Lightweight Open Language Models.
4. Dettmers et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs.

### Architecture 2 (DL + NLP)
1. Liu & Lane (2016). Attention-Based Recurrent Neural Network Models for Joint Intent Detection. Interspeech.
2. Lample et al. (2016). Neural Architectures for Named Entity Recognition. NAACL.
3. Martin et al. (2019). CamemBERT: a Tasty French Language Model. ACL.
4. Hochreiter & Schmidhuber (1997). Long Short-Term Memory. Neural Computation.
5. Ma & Hovy (2016). End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF. ACL.
6. Vaswani et al. (2017). Attention Is All You Need. NeurIPS.

---

## 🎓 Pour la soutenance

### Suggestions de présentation

1. **Démonstration live** (5 min):
   - Ouvrir un notebook sur Colab
   - Montrer l'exécution d'une cellule de génération
   - Tester avec une requête réelle

2. **Résultats comparatifs** (3 min):
   - Présenter le tableau comparatif Architecture 1 vs 2
   - Mettre en évidence les trade-offs
   - Justifier la recommandation (Architecture 2 pour production)

3. **Diagrammes Mermaid** (2 min):
   - Les notebooks incluent des diagrammes Mermaid
   - Copier/coller dans votre présentation
   - Expliquer le flux de traitement

---

## ✨ Points forts pour le jury

- ✅ **Reproductibilité**: Notebooks exécutables en 1 clic sur Colab
- ✅ **Comparaison rigoureuse**: Deux approches complémentaires
- ✅ **Citations académiques**: Références solides pour chaque choix
- ✅ **Prétraitement adapté**: Gestion code-switching, anonymisation RGPD
- ✅ **Production-ready**: Recommandations concrètes pour déploiement
- ✅ **Documentation**: Diagrammes Mermaid, commentaires détaillés

---

## 📝 Notes

- Les notebooks sont **directement exécutables** sans modification
- Les **placeholders sont clairement marqués** avec 🔹
- Les **diagrammes Mermaid** sont intégrés dans les cellules markdown
- Le **prétraitement est identique** dans les deux notebooks (cohérence)
- Les **citations** suivent le format ACL/IEEE

---

## 🤝 Support

Pour toute question sur l'exécution des notebooks:
1. Vérifier la section **Dépannage** ci-dessus
2. Consulter les commentaires dans les cellules du notebook
3. Vérifier la documentation Colab: https://colab.research.google.com/

Bon travail ! 🎓
