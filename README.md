# 🤖 Système Conversationnel EasyTransfert - Architectures Expérimentales

Implémentation de trois architectures expérimentales pour un assistant conversationnel intelligent destiné au service client d'EasyTransfert, une application de transfert d'argent mobile en Côte d'Ivoire.

## 🚀 Lancement Rapide sur Google Colab Pro

**Tous les notebooks sont optimisés pour Google Colab Pro !** Cliquez sur un badge pour démarrer :

| Architecture | Description | Colab |
|-------------|-------------|-------|
| **Architecture 1** | Fine-tuning LoRA (Baseline) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AmedBah/memoire/blob/main/notebooks/architecture_1/01_architecture_1_simple_agent_finetuning.ipynb) |
| **Architecture 2** | RAG Standard | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AmedBah/memoire/blob/main/notebooks/architecture_2/02_architecture_2_rag_standard.ipynb) |
| **Architecture 3** | RAG-Agentique | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AmedBah/memoire/blob/main/notebooks/architecture_3/03_architecture_3_rag_agentique.ipynb) |

**Guide complet**: Consultez [COLAB_GUIDE.md](./COLAB_GUIDE.md) pour instructions détaillées.

## 📁 Structure du Projet

```
memoire/
├── data/                          # Sources de données organisées
│   ├── conversations/             # Données conversationnelles (3000+)
│   ├── documents/                 # Documentation et logs
│   ├── faqs/                      # Questions-réponses (8 entrées)
│   ├── operators/                 # Infos opérateurs (5 opérateurs)
│   ├── procedures/                # Procédures résolution (3 procédures)
│   └── expressions/               # Expressions ivoiriennes (20+)
├── notebooks/                     # Notebooks Jupyter (adaptés Colab)
│   ├── architecture_1/            # Fine-tuning LoRA
│   ├── architecture_2/            # RAG Standard
│   ├── architecture_3/            # RAG-Agentique
│   └── evaluation/                # Évaluation comparative
├── ARCHITECTURE_README.md         # Documentation des architectures
├── COLAB_GUIDE.md                 # Guide utilisation Colab Pro
├── IMPLEMENTATION_SUMMARY.md      # Résumé implémentation
└── requirements.txt               # Dépendances Python
```

## 📚 Documentation

- **[ARCHITECTURE_README.md](./ARCHITECTURE_README.md)** - Description détaillée des 3 architectures
- **[COLAB_GUIDE.md](./COLAB_GUIDE.md)** - Guide complet Google Colab Pro
- **[IMPLEMENTATION_SUMMARY.md](./IMPLEMENTATION_SUMMARY.md)** - Résumé de l'implémentation
- **[data/README.md](./data/README.md)** - Documentation des sources de données

## 🎯 Les Trois Architectures

### Architecture 1: Agent Simple (Baseline)
- Llama 3.2 3B fine-tuné avec LoRA
- Toutes les connaissances dans les paramètres du modèle
- Inférence rapide (~2-3s)
- **GPU recommandé**: V100 ou A100

### Architecture 2: RAG Standard
- Génération Augmentée par Récupération
- Base vectorielle ChromaDB
- Séparation connaissances/raisonnement
- **GPU recommandé**: T4 ou V100

### Architecture 3: RAG-Agentique
- RAG + Capacités agentiques (ReAct)
- 4 outils métier spécialisés
- Raisonnement multi-étapes
- **GPU recommandé**: V100

## 🛠️ Installation Locale

```bash
# Cloner le repository
git clone https://github.com/AmedBah/memoire.git
cd memoire

# Installer les dépendances
pip install -r requirements.txt

# Lancer un notebook
jupyter notebook notebooks/architecture_1/01_architecture_1_simple_agent_finetuning.ipynb
```

**Prérequis locaux:**
- Python 3.8+
- GPU avec CUDA (recommandé)
- 16-32 GB RAM
- ~6 GB espace disque

## 📊 Sources de Données

Le dossier `data/` contient toutes les sources de données structurées :

- **Conversations** : 3031 conversations historiques pour fine-tuning
- **FAQs** : 8 questions-réponses officielles par catégorie
- **Opérateurs** : Infos complètes sur 5 opérateurs (MTN, Orange, Moov, Wave, Trésor Money)
- **Procédures** : 3 guides de résolution détaillés
- **Expressions** : 20+ expressions ivoiriennes avec contexte
- **Documentation** : Documentation technique complète

Voir [data/README.md](./data/README.md) pour détails.

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :

1. Fork le projet
2. Créer une branche (`git checkout -b feature/amelioration`)
3. Commit vos changements (`git commit -m 'Ajout fonctionnalité'`)
4. Push vers la branche (`git push origin feature/amelioration`)
5. Ouvrir une Pull Request

## 📄 Licence

Ce projet est sous licence MIT.

## 📞 Contact

- **Email** : support@easytransfert.ci
- **Téléphone** : 2522018730 (WhatsApp 24h/24)

---

**Développé avec ❤️ pour EasyTransfert**