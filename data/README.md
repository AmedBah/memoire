# Data Directory - EasyTransfert AI System

Ce dossier contient toutes les sources de données structurées utilisées pour la vectorisation et l'entraînement du système conversationnel EasyTransfert.

## 📁 Structure du Dossier

```
data/
├── conversations/          # Données conversationnelles pour fine-tuning
├── documents/             # Documentation et logs de transactions
├── faqs/                  # Questions-réponses fréquentes
├── operators/             # Informations sur les opérateurs mobile money
├── procedures/            # Procédures de résolution de problèmes
└── expressions/           # Expressions et abréviations ivoiriennes
```

## 📋 Description des Sources de Données

### 1. Conversations (`conversations/`)

**Fichiers:**
- `conversation_1000_finetune.jsonl` - 3031 conversations historiques au format JSON Lines
- `Conversation_easybot.json` - Conversations additionnelles

**Format:**
```json
{
  "messages": [
    {"role": "user", "content": "Question de l'utilisateur"},
    {"role": "assistant", "content": "Réponse de l'assistant"}
  ]
}
```

**Usage:**
- Fine-tuning de modèles LLM (Architecture 1)
- Exemples pour RAG (Architectures 2 et 3)
- Analyse de patterns conversationnels

---

### 2. Documents (`documents/`)

**Fichiers:**
- `doc.txt.txt` - Documentation technique complète du système
- `transaction_logs_sample.json` - Exemples de logs de transactions

**Format transaction_logs_sample.json:**
```json
{
  "transaction_id": "MP1234567890",
  "timestamp": "2024-10-12T10:30:00",
  "operator_from": "MTN",
  "operator_to": "Orange",
  "amount": 25000,
  "status": "success"
}
```

**Usage:**
- Base de connaissances pour RAG
- Simulation de vérification de statut de transactions
- Documentation système

---

### 3. FAQs (`faqs/`)

**Fichiers:**
- `faq_easytransfert.json` - 8+ questions-réponses officielles

**Format:**
```json
{
  "id": 1,
  "categorie": "general",
  "question": "Qu'est-ce qu'EasyTransfert ?",
  "reponse": "...",
  "mots_cles": ["easytransfert", "application"]
}
```

**Catégories:**
- `general` - Questions générales sur EasyTransfert
- `operateurs` - Informations sur les opérateurs compatibles
- `utilisation` - Guide d'utilisation
- `tarifs` - Frais et coûts
- `limites` - Limites de transactions
- `problemes` - Résolution de problèmes
- `securite` - Sécurité et protection
- `support` - Service client

**Usage:**
- Base de connaissances ChromaDB (RAG)
- Réponses rapides aux questions courantes
- Vectorisation sémantique

---

### 4. Operators (`operators/`)

**Fichiers:**
- `operators_info.json` - Informations détaillées sur 5 opérateurs

**Opérateurs couverts:**
- MTN Mobile Money
- Orange Money
- Moov Money
- Wave Mobile Money
- Trésor Money

**Format:**
```json
{
  "MTN": {
    "nom_complet": "MTN Mobile Money",
    "format_identifiant": {
      "type": "alphanumérique",
      "pattern": "MP suivi de 10 chiffres",
      "exemple": "MP1234567890"
    },
    "limites_transaction": {
      "minimum": 100,
      "maximum": 1500000,
      "devise": "FCFA"
    },
    "frais": {...},
    "prefixes_telephone": ["05", "06", "07"],
    "compatible_avec": [...]
  }
}
```

**Usage:**
- Outil "Operator Info" (Architecture 3)
- Validation de formats d'identifiants
- Vérification de limites et compatibilités
- Base de connaissances RAG

---

### 5. Procedures (`procedures/`)

**Fichiers:**
- `procedures_resolution.json` - 3 procédures de résolution détaillées

**Procédures incluses:**
1. Transaction échouée mais débit effectué
2. Mot de passe oublié
3. Erreur de numéro de destinataire

**Format:**
```json
{
  "id": 1,
  "titre": "Transaction échouée mais débit effectué",
  "categorie": "probleme_technique",
  "niveau_gravite": "élevé",
  "etapes": [
    {
      "numero": 1,
      "action": "Vérifier le statut",
      "details": "..."
    }
  ],
  "informations_requises": [...],
  "delai_resolution": "24-48 heures"
}
```

**Usage:**
- Guides étape par étape pour résolution
- Base de connaissances RAG
- Formation des agents de support

---

### 6. Expressions (`expressions/`)

**Fichiers:**
- `expressions_ivoiriennes.json` - 20+ expressions et abréviations locales

**Format:**
```json
{
  "expressions": [
    {
      "expression": "c'est comment",
      "signification": "comment vas-tu",
      "contexte": "salutation"
    }
  ],
  "abreviations": {
    "stp": "s'il te plaît",
    "tkt": "ne t'inquiète pas"
  }
}
```

**Usage:**
- Enrichissement linguistique
- Entity Extractor (Architecture 3)
- Compréhension du contexte ivoirien
- Normalisation de texte

---

## 🔧 Utilisation dans les Architectures

### Architecture 1: Agent Simple (Fine-tuning)
- ✅ `conversations/conversation_1000_finetune.jsonl` - Entraînement LoRA
- ✅ `expressions/expressions_ivoiriennes.json` - Enrichissement linguistique

### Architecture 2: RAG Standard
- ✅ `faqs/faq_easytransfert.json` - Base ChromaDB
- ✅ `operators/operators_info.json` - Documentation opérateurs
- ✅ `procedures/procedures_resolution.json` - Guides de résolution
- ✅ `documents/doc.txt.txt` - Documentation technique
- ✅ Échantillons de `conversations/` - Exemples historiques

### Architecture 3: RAG-Agentique
- ✅ Toutes les sources de l'Architecture 2 pour RAG Retriever
- ✅ `operators/operators_info.json` - Outil Operator Info
- ✅ `documents/transaction_logs_sample.json` - Simulation vérification statut
- ✅ `expressions/expressions_ivoiriennes.json` - Entity Extractor

---

## 📊 Statistiques

| Type de données | Nombre de fichiers | Taille approximative |
|----------------|-------------------|---------------------|
| Conversations | 2 | ~6.4 MB |
| Documents | 2 | ~65 KB |
| FAQs | 1 | 8 entrées |
| Opérateurs | 1 | 5 opérateurs |
| Procédures | 1 | 3 procédures |
| Expressions | 1 | 20+ expressions |

---

## 🔄 Mise à Jour des Données

Pour ajouter ou mettre à jour des données:

1. **Conversations**: Ajouter au format JSON Lines avec structure `messages`
2. **FAQs**: Respecter le format avec `id`, `categorie`, `question`, `reponse`, `mots_cles`
3. **Opérateurs**: Maintenir la structure complète avec tous les champs
4. **Procédures**: Suivre le format avec `etapes` numérotées
5. **Expressions**: Ajouter à la liste `expressions` ou au dictionnaire `abreviations`

---

## 🚀 Vectorisation et Chunking

### Paramètres recommandés pour RAG:

**Embedding:**
- Modèle: `paraphrase-multilingual-mpnet-base-v2`
- Dimensions: 768
- Langues: Français, Anglais, + 50 autres

**Chunking:**
- Taille maximale: 512 tokens
- Overlap: 50 tokens
- Préservation des métadonnées: Oui

**Métadonnées à conserver:**
- `source`: Nom du fichier source
- `categorie`: Type de contenu
- `operateur`: Si applicable
- `date`: Date de création/modification

---

## 📝 Notes Importantes

1. **Données sensibles**: Tous les numéros et identifiants sont anonymisés
2. **Format FCFA**: Toutes les valeurs monétaires sont en Francs CFA
3. **Opérateurs**: Informations à jour au moment de la création
4. **Expressions**: Spécifiques au contexte ivoirien
5. **Maintenance**: Mettre à jour régulièrement avec nouvelles données

---

## 📞 Contact

Pour questions sur les données:
- Email: support@easytransfert.ci
- Téléphone: 2522018730

---

**Dernière mise à jour**: 2024-10-12
