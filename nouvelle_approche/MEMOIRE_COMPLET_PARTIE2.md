# MÉMOIRE COMPLET - PARTIE 2

## Suite du Chapitre I et Chapitres II-III

---

# CHAPITRE II : ÉTAT DE L'ART

Ce chapitre présente une synthèse de l'état de l'art scientifique et technologique dans les domaines de l'intelligence artificielle conversationnelle, du traitement automatique du langage naturel (NLP) et de leurs applications au service client. Nous explorons l'évolution des systèmes conversationnels, les avancées récentes en matière de Large Language Models et de Deep Learning, puis nous analysons les applications concrètes dans le domaine du support client automatisé.

## I. Intelligence Artificielle Conversationnelle

### 1. Évolution des systèmes conversationnels

**Première génération : Systèmes à base de règles (1960-1990)**

Les premiers systèmes conversationnels, illustrés par ELIZA (Weizenbaum, 1966) et PARRY (Colby, 1972), reposaient sur des règles manuelles et des patterns de reconnaissance. Ces systèmes utilisaient des techniques de pattern matching pour identifier des mots-clés dans l'entrée utilisateur et générer des réponses préprogrammées.

Caractéristiques :
- Vocabulaire limité (50-200 patterns)
- Pas de véritable compréhension du langage
- Dialogue rigide et prévisible
- Domaine d'application très restreint

Limitations : Incapacité à gérer les variations linguistiques, absence d'apprentissage, scalabilité limitée.

**Deuxième génération : Approches statistiques et Machine Learning (1990-2015)**

L'avènement des méthodes statistiques et du machine learning a permis de développer des systèmes plus robustes. Cette période a vu l'émergence de :

- **SVM et Naive Bayes** pour la classification d'intentions (Joachims, 1998)
- **Hidden Markov Models (HMM)** pour la modélisation du dialogue (Rabiner, 1989)
- **Conditional Random Fields (CRF)** pour l'extraction d'entités (Lafferty et al., 2001)
- **TF-IDF et Topic Models** pour la récupération d'information (Blei et al., 2003)

Avancées clés :
- Apprentissage automatique à partir de données
- Modélisation probabiliste du langage
- Capacité de généralisation sur des données non vues
- Méthodes d'évaluation quantitatives

Limitations : Nécessité de feature engineering manuel, compréhension contextuelle limitée, difficulté à capturer les dépendances long-terme.

**Troisième génération : Deep Learning et représentations distribuées (2015-2020)**

L'introduction des réseaux de neurones profonds a révolutionné le NLP :

- **Word2Vec et GloVe** (Mikolov et al., 2013; Pennington et al., 2014) : Embeddings de mots capturant les relations sémantiques
- **Seq2Seq avec attention** (Bahdanau et al., 2015) : Génération de séquences contextuelles
- **RNN, LSTM, GRU** (Hochreiter & Schmidhuber, 1997) : Modélisation des dépendances temporelles
- **CNN pour NLP** (Kim, 2014) : Extraction de features locales dans le texte

Applications conversationnelles :
- Chatbots de questions-réponses plus robustes
- Traduction automatique neuronale
- Systèmes de dialogue orientés tâche
- Génération de texte conditionné

Avancées : Élimination du feature engineering, apprentissage de représentations end-to-end, meilleure gestion du contexte.

Limitations persistantes : Nécessité de grandes quantités de données d'entraînement, coût computationnel élevé, difficulté de transfert cross-domain.

**Quatrième génération : Transformers et modèles pré-entraînés (2018-présent)**

L'architecture Transformer (Vaswani et al., 2017) et les modèles pré-entraînés ont marqué un tournant majeur :

- **BERT** (Devlin et al., 2019) : Pré-entraînement bidirectionnel
- **GPT** série (OpenAI, 2018-2024) : Modèles autorégressifs à grande échelle
- **T5, BART** (Raffel et al., 2020; Lewis et al., 2020) : Modèles encodeur-décodeur unifiés
- **Llama** série (Meta, 2023-2024) : LLM open-source performants

Paradigme du pré-entraînement + fine-tuning :
1. **Pré-entraînement** sur de vastes corpus (Common Crawl, Wikipedia, livres)
2. **Fine-tuning** sur des tâches spécifiques avec moins de données
3. **Prompting** et **in-context learning** sans fine-tuning

Capacités émergentes :
- Compréhension du contexte long (jusqu'à 128k tokens)
- Raisonnement multi-étapes (Chain-of-Thought)
- Généralisation zero-shot et few-shot
- Multilinguisme natif

Ces avancées rendent aujourd'hui possible des systèmes conversationnels hautement performants nécessitant moins de données d'entraînement spécifiques.

### 2. Large Language Models (LLM)

**Définition et caractéristiques**

Les Large Language Models sont des modèles de réseaux de neurones entraînés sur de vastes corpus textuels (plusieurs téraoctets) pour prédire le prochain token d'une séquence. Leur taille se mesure en nombre de paramètres, allant de quelques milliards à plusieurs centaines de milliards.

Caractéristiques clés :
- **Échelle massive** : Milliards de paramètres (GPT-4 : ~1.7T, Llama 3.1 70B : 70B)
- **Architecture Transformer** : Mécanismes d'attention multi-têtes
- **Pré-entraînement non supervisé** : Objectif de modélisation du langage
- **Tokenisation subword** : BPE, WordPiece, SentencePiece
- **Contextual embeddings** : Représentations dépendant du contexte

**Modèles de référence (2024)**

**GPT-4 (OpenAI)** :
- Modèle propriétaire multimodal (texte + images)
- Environ 1.7 trillion de paramètres
- Contexte jusqu'à 128k tokens
- Performances SOTA sur de nombreux benchmarks
- Coût : $0.03/1k tokens (input), $0.06/1k tokens (output)
- Limitation : API seulement, pas d'auto-hébergement

**Claude 3 (Anthropic)** :
- Famille de modèles (Opus, Sonnet, Haiku)
- Contexte jusqu'à 200k tokens
- Focus sur la sécurité et l'alignement
- Performances comparables à GPT-4
- Coût similaire à GPT-4

**Llama 3.1 (Meta)** :
- Open-source sous licence permissive
- Variantes : 8B, 70B, 405B paramètres
- Multilingue (10+ langues dont français)
- Auto-hébergement possible
- Performance : 70B comparable à GPT-3.5
- Avantage : Contrôle complet, personnalisation

**Mistral 7B et Mixtral 8x7B (Mistral AI)** :
- Modèles européens open-source
- Excellent rapport performance/taille
- Mistral 7B : Performant pour sa taille compacte
- Mixtral 8x7B : Architecture Mixture of Experts
- Optimisé pour le français

**Capacités des LLM pour le service client**

1. **Compréhension en langage naturel** :
   - Parsing de requêtes complexes et ambiguës
   - Gestion du code-switching et erreurs orthographiques
   - Extraction implicite d'intentions et entités

2. **Génération de réponses contextuelles** :
   - Réponses cohérentes et fluides en français
   - Adaptation du ton et du style
   - Personnalisation selon le profil utilisateur

3. **Raisonnement et résolution de problèmes** :
   - Chain-of-Thought reasoning
   - Décomposition de problèmes complexes
   - Accès à des bases de connaissances (via RAG)

4. **Apprentissage contextuel (In-Context Learning)** :
   - Few-shot learning : exemples dans le prompt
   - Adaptation rapide sans réentraînement
   - Personnalisation par utilisateur/session

**Limites et défis des LLM**

1. **Hallucinations** :
   - Génération de faits incorrects avec grande confiance
   - Particulièrement problématique pour le service client
   - Mitigation : RAG, calibration, fact-checking

2. **Coût computationnel** :
   - Inférence coûteuse (GPU nécessaire)
   - Latence proportionnelle à la longueur de génération
   - Coût par requête significatif en production

3. **Contrôle et prévisibilité** :
   - Comportement parfois imprévisible
   - Difficulté de garantir des contraintes strictes
   - Nécessité de guardrails et validation

4. **Biais et sécurité** :
   - Reproduction de biais du corpus d'entraînement
   - Potentiel de générer du contenu inapproprié
   - Nécessité de modération et filtrage

5. **Dépendance aux données** :
   - Knowledge cutoff (connaissances figées à la date d'entraînement)
   - Nécessité de mise à jour via RAG ou fine-tuning
   - Gestion des informations évolutives

### 3. Architectures d'agents intelligents

**Concept d'agent conversationnel**

Un agent conversationnel intelligent est un système qui :
- **Perçoit** son environnement (messages utilisateurs, contexte)
- **Raisonne** sur les actions à entreprendre
- **Agit** en générant des réponses ou en déclenchant des actions
- **Apprend** de ses interactions pour s'améliorer

**Architecture ReAct (Reason + Act)**

Proposée par Yao et al. (2023), l'architecture ReAct combine raisonnement et action dans un cycle itératif :

```
Cycle ReAct :
1. Thought (Pensée) : L'agent analyse la situation
2. Action : L'agent choisit un outil et l'exécute
3. Observation : L'agent examine le résultat
4. Répète jusqu'à avoir assez d'informations
5. Final Answer : L'agent génère la réponse finale
```

Avantages :
- Transparence du raisonnement (traces explicites)
- Capacité à utiliser des outils externes
- Décomposition de problèmes complexes
- Gestion d'erreurs et récupération

Application au service client :
- Outil 1 : Recherche dans la base de connaissances (FAQ, procédures)
- Outil 2 : Requête à l'API de transactions (vérification de statut)
- Outil 3 : Extraction d'entités (ID transaction, numéros, montants)
- Outil 4 : Gestion de mémoire conversationnelle (historique)

**Architecture RAG (Retrieval-Augmented Generation)**

Le RAG (Lewis et al., 2020) augmente la génération avec une récupération d'information :

```
Pipeline RAG :
1. Requête utilisateur
2. Vectorisation de la requête
3. Recherche de similarité dans la base vectorielle
4. Récupération des k documents les plus pertinents
5. Enrichissement du prompt avec les documents
6. Génération de la réponse par le LLM
```

Avantages :
- Réduction des hallucinations
- Traçabilité (citations de sources)
- Mise à jour facile de la base de connaissances
- Pas besoin de réentraîner le modèle

Composants techniques :
- **Embedding model** : paraphrase-multilingual-mpnet-base-v2 (768 dim)
- **Vector DB** : ChromaDB, Pinecone, Weaviate
- **Chunking strategy** : Découpage intelligent du texte (512-1024 tokens)
- **Retrieval** : Similarité cosinus, score de réranking

**Architecture hybride : RAG + ReAct**

La combinaison RAG + ReAct offre le meilleur des deux mondes :
- RAG fournit les connaissances factuelles
- ReAct permet le raisonnement multi-étapes et l'utilisation d'outils

Cette architecture est particulièrement adaptée au service client où il faut :
- Accéder à de la documentation à jour (RAG)
- Interroger des APIs externes (ReAct)
- Raisonner sur des problèmes complexes (ReAct)
- Maintenir le contexte conversationnel (ReAct)

## II. Techniques de Deep Learning pour le NLP

### 1. Réseaux de neurones récurrents (RNN, LSTM, GRU)

**Réseaux de Neurones Récurrents (RNN)**

Les RNN, introduits par Rumelhart et al. (1986), sont conçus pour traiter des séquences de longueur variable en maintenant un état caché :

```
Équations RNN :
h_t = tanh(W_hh * h_(t-1) + W_xh * x_t + b_h)
y_t = W_hy * h_t + b_y
```

Où :
- h_t : état caché au temps t
- x_t : entrée au temps t
- y_t : sortie au temps t
- W : matrices de poids
- b : vecteurs de biais

Avantages :
- Traitement de séquences de longueur variable
- Partage de paramètres dans le temps
- Mémoire des entrées précédentes

Limitations :
- **Vanishing gradient** : Difficulté d'apprentissage sur longues séquences
- **Biais récent** : Oubli des informations lointaines
- **Parallélisation limitée** : Calcul séquentiel obligatoire

**Long Short-Term Memory (LSTM)**

Les LSTM (Hochreiter & Schmidhuber, 1997) résolvent le problème du vanishing gradient avec des portes (gates) :

```
Équations LSTM :
f_t = σ(W_f * [h_(t-1), x_t] + b_f)   # Forget gate
i_t = σ(W_i * [h_(t-1), x_t] + b_i)   # Input gate
C̃_t = tanh(W_C * [h_(t-1), x_t] + b_C) # Candidate cell state
C_t = f_t ⊙ C_(t-1) + i_t ⊙ C̃_t       # Cell state update
o_t = σ(W_o * [h_(t-1), x_t] + b_o)   # Output gate
h_t = o_t ⊙ tanh(C_t)                 # Hidden state
```

Mécanismes clés :
- **Cell state** : Mémoire long-terme
- **Forget gate** : Décide quoi oublier
- **Input gate** : Décide quoi ajouter
- **Output gate** : Décide quoi exposer

Applications NLP :
- Modélisation du langage
- Traduction automatique (seq2seq)
- Classification de texte
- Extraction d'entités nommées

**Gated Recurrent Unit (GRU)**

Les GRU (Cho et al., 2014) simplifient les LSTM avec moins de paramètres :

```
Équations GRU :
r_t = σ(W_r * [h_(t-1), x_t])      # Reset gate
z_t = σ(W_z * [h_(t-1), x_t])      # Update gate
h̃_t = tanh(W * [r_t ⊙ h_(t-1), x_t]) # Candidate hidden state
h_t = (1 - z_t) ⊙ h_(t-1) + z_t ⊙ h̃_t # Final hidden state
```

Avantages vs LSTM :
- Moins de paramètres (plus rapide à entraîner)
- Performance comparable sur beaucoup de tâches
- Plus simple à implémenter et debugger

**Applications au service client**

Pour un système de service client, les RNN/LSTM/GRU sont utilisés pour :

1. **Classification d'intentions** :
   - BiLSTM pour capturer le contexte bidirectionnel
   - Softmax final sur les classes d'intentions
   - Exemple : "problème de transfert" → classe: PROBLEME_TRANSACTION

2. **Extraction d'entités** :
   - BiLSTM-CRF pour le NER
   - Identification de : numéros de téléphone, montants, opérateurs, IDs
   - Tagging BIO (Begin, Inside, Outside)

3. **Génération de réponses** :
   - Encoder-decoder avec LSTM
   - Attention mechanism pour focaliser sur les parties pertinentes
   - Génération token par token

### 2. Mécanismes d'attention et Transformers

**Mécanisme d'attention**

L'attention (Bahdanau et al., 2015) permet au modèle de se focaliser sur les parties pertinentes de l'entrée :

```
Attention score :
e_ij = a(s_i, h_j)  # score entre état décodeur i et encodeur j
α_ij = exp(e_ij) / Σ_k exp(e_ik)  # Softmax normalization
c_i = Σ_j α_ij * h_j  # Context vector (somme pondérée)
```

Types d'attention :
- **Additive (Bahdanau)** : a(s, h) = v^T * tanh(W_s * s + W_h * h)
- **Multiplicative (Luong)** : a(s, h) = s^T * W * h
- **Scaled dot-product** : a(Q, K) = softmax(Q*K^T / √d_k)

Avantages :
- Gestion des dépendances longue distance
- Interprétabilité (visualisation des poids d'attention)
- Performance améliorée sur seq2seq

**Architecture Transformer**

Les Transformers (Vaswani et al., 2017) reposent entièrement sur l'attention, éliminant la récurrence :

Composants clés :

1. **Multi-Head Self-Attention** :
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W^O
où head_i = Attention(Q*W_i^Q, K*W_i^K, V*W_i^V)
```

Permet au modèle d'attendre à différentes positions et sous-espaces.

2. **Position-wise Feed-Forward** :
```
FFN(x) = max(0, x*W_1 + b_1)*W_2 + b_2
```

Appliqué indépendamment à chaque position.

3. **Positional Encoding** :
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Injecte l'information de position (ordre des tokens).

4. **Layer Normalization et Residual Connections** :
```
Output = LayerNorm(x + Sublayer(x))
```

Stabilise l'entraînement et permet des réseaux profonds.

**Architecture complète** :

```
Encoder (N layers) :
  - Multi-Head Self-Attention
  - Add & Norm
  - Feed-Forward
  - Add & Norm

Decoder (N layers) :
  - Masked Multi-Head Self-Attention
  - Add & Norm
  - Multi-Head Cross-Attention (sur l'encodeur)
  - Add & Norm
  - Feed-Forward
  - Add & Norm
```

Avantages des Transformers :
- **Parallélisation** : Tous les tokens traités simultanément
- **Longue portée** : Attention directe entre tous les tokens
- **Scalabilité** : Performance croissante avec la taille
- **Transfert** : Pré-entraînement + fine-tuning efficace

Variantes architecturales :

- **Encoder-only** (BERT) : Bidirectionnel, pour tâches de compréhension
- **Decoder-only** (GPT) : Autorégressif, pour génération
- **Encoder-Decoder** (T5, BART) : Complet, pour seq2seq

### 3. Modèles pré-entraînés et Transfer Learning

**Paradigme du Transfer Learning en NLP**

Le transfer learning en NLP se déroule en deux phases :

**Phase 1 : Pré-entraînement (Pretraining)**

Objectifs non supervisés sur large corpus :
- **Masked Language Modeling (MLM)** : BERT
  - Masquer 15% des tokens
  - Prédire les tokens masqués
  - Apprentissage bidirectionnel

- **Causal Language Modeling (CLM)** : GPT
  - Prédire le prochain token
  - Apprentissage autorégressif gauche-à-droite

- **Sequence-to-Sequence** : T5
  - Diverses tâches de transformation de texte
  - Format unifié input → output

Corpus typiques :
- Common Crawl (plusieurs téraoctets web)
- Wikipedia (haute qualité, multilingue)
- Livres, articles scientifiques
- Code source (pour modèles de code)

Résultat : Modèle avec représentations linguistiques générales

**Phase 2 : Fine-tuning (Ajustement fin)**

Adaptation sur tâche spécifique avec données annotées :
- Classification : Ajouter une couche softmax finale
- NER : Ajouter une couche CRF
- Génération : Ajuster le decoder

Stratégies de fine-tuning :

1. **Full fine-tuning** :
   - Tous les paramètres mis à jour
   - Meilleure performance mais coûteux
   - Risque d'overfitting si peu de données

2. **Feature extraction** :
   - Poids gelés, entraînement seulement des nouvelles couches
   - Rapide mais performance limitée

3. **Parameter-Efficient Fine-Tuning (PEFT)** :
   - **LoRA** (Low-Rank Adaptation) : Décomposition matricielle rang faible
   - **Prefix Tuning** : Apprentissage de préfixes virtuels
   - **Adapter Layers** : Petits modules insérés entre couches
   - Avantages : Réduit paramètres à entraîner de 99%, garde performance

**LoRA en détail**

LoRA (Hu et al., 2021) décompose les mises à jour de poids :

```
W_new = W_0 + ΔW
où ΔW = B * A (décomposition rang faible)
```

- W_0 : Poids pré-entraînés (gelés)
- B ∈ R^(d × r), A ∈ R^(r × k) : Matrices de rang r << min(d,k)
- r : Rang (typiquement 8-64)

Paramètres à entraîner : 2*r*(d+k) au lieu de d*k

Avantages :
- Réduction massive des paramètres (0.1-1% de l'original)
- Entraînement plus rapide (moins de mémoire GPU)
- Plusieurs adaptateurs LoRA pour différentes tâches
- Pas de latence d'inférence supplémentaire

Application au service client :
- Pré-entraînement : Llama 3.2 3B sur corpus général
- Fine-tuning LoRA : Conversations EasyTransfert (3031 exemples)
- Résultat : Modèle spécialisé avec seulement ~50 MB de paramètres additionnels

**Modèles multilingues pour le français**

Modèles pré-entraînés pertinents pour notre contexte :

1. **CamemBERT** (Martin et al., 2020) :
   - BERT entraîné sur français (OSCAR corpus)
   - 110M ou 335M paramètres
   - Excellent pour tâches de compréhension

2. **mBERT** (Devlin et al., 2019) :
   - BERT multilingue (104 langues)
   - Bon pour code-switching
   - Performance française légèrement inférieure à CamemBERT

3. **XLM-RoBERTa** (Conneau et al., 2020) :
   - 100 langues, 270M ou 550M paramètres
   - Excellent cross-lingual
   - Robuste au code-switching

4. **Llama 3.1 (8B, 70B)** (Meta, 2024) :
   - Multilingue incluant français
   - Génération et compréhension
   - Open-source, fine-tunable

5. **Mistral 7B** (Mistral AI, 2023) :
   - Français renforcé
   - Excellent rapport qualité/taille
   - Open-source européen

## III. Applications dans le service client

### 1. Chatbots et assistants virtuels

**Taxonomie des chatbots**

Les chatbots peuvent être classés selon plusieurs dimensions :

**Par complexité** :
1. **Rule-based** : Arbre de décision fixe
2. **Retrieval-based** : Sélection dans une base de réponses
3. **Generative** : Génération de nouvelles réponses

**Par domaine** :
1. **Open-domain** : Conversation générale (difficile)
2. **Closed-domain** : Domaine spécifique (service client)

**Par architecture** :
1. **Intent-based** : Classification puis réponse template
2. **QA-based** : Question-answering sur documents
3. **Dialogue management** : Gestion d'état et de politique
4. **End-to-end neural** : Entrée → Réponse directe

**Architectures modernes pour le service client**

**Architecture 1 : Pipeline intent + NER + Response**

```
1. Intent Classification :
   User: "Mon transfert n'est pas arrivé"
   → Intent: PROBLEME_TRANSACTION (confidence: 0.95)

2. Entity Extraction :
   → Entities: {
       "probleme_type": "non_arrivé",
       "sujet": "transfert"
     }

3. Dialogue State Tracking :
   → State: MISSING_INFO
   → Slots requis: {transaction_id, phone_number, amount}

4. Response Generation :
   Template: "Je comprends que votre transfert n'est pas arrivé. 
              Pour vous aider, j'ai besoin de votre identifiant de 
              transaction. Pouvez-vous me le fournir ?"
```

Avantages :
- Contrôle précis du comportement
- Debugging facile
- Performance prévisible

Inconvénients :
- Rigide et peu flexible
- Nécessite beaucoup de règles manuelles
- Difficulté à gérer l'imprévu

**Architecture 2 : LLM avec prompting structuré**

```
System Prompt:
"Tu es un assistant du service client EasyTransfert.
 
Contexte: EasyTransfert permet des transferts entre 
MTN, Orange, Moov, Wave, Trésor Money.

Règles:
- Toujours demander l'ID de transaction pour les problèmes
- Frais: 1-2% du montant (min 25, max 500 FCFA)
- En cas de doute, rediriger vers 2522018730
- Ton empathique avec émojis

Exemples:
[few-shot examples...]
"

User: "Mon transfert n'est pas arrivé"

Response: "Je comprends votre inquiétude 😟. Pour vous 
aider efficacement, pourriez-vous me communiquer :
1. L'identifiant de votre transaction
2. Le montant transféré
3. Les opérateurs concernés (émetteur et récepteur)

Ainsi, je pourrai vérifier le statut de votre transfert."
```

Avantages :
- Très flexible et naturel
- Gère les variations linguistiques
- Adaptation contextuelle automatique

Inconvénients :
- Moins de contrôle
- Risque d'hallucinations
- Coût d'inférence plus élevé

**Architecture 3 : RAG pour service client**

```
1. User Query: "Quels sont les frais pour un transfert de 
                 50000 FCFA de MTN vers Orange?"

2. Query Vectorization:
   embedding_model(query) → vector(768 dim)

3. Retrieval from Knowledge Base:
   ChromaDB.search(vector, top_k=3)
   → Retrieved docs:
     - FAQ #4: "Frais de transfert"
     - Procédure: "Calcul des frais"
     - Doc technique: "Grilles tarifaires par opérateur"

4. Prompt Augmentation:
   """
   Contexte pertinent:
   [Docs récupérés...]
   
   Question: {user_query}
   
   Réponds en te basant UNIQUEMENT sur le contexte fourni.
   Cite tes sources.
   """

5. LLM Generation:
   "D'après notre grille tarifaire, pour un transfert de 
    50 000 FCFA de MTN vers Orange, les frais sont de 
    500 FCFA (soit 1% du montant, plafonné à 500 FCFA).
    
    Source: FAQ #4 - Frais de transfert"
```

Avantages :
- Réponses factuelles et traçables
- Mise à jour facile de la base de connaissances
- Réduit les hallucinations

**Cas d'usage typiques en service client**

1. **Information générale** :
   - Horaires, limites, frais
   - Procédures de base
   - Compatibilités opérateurs

2. **Support transactionnel** :
   - Suivi de transaction
   - Vérification de statut
   - Problèmes d'exécution

3. **Résolution de problèmes** :
   - Transaction échouée
   - Argent non reçu
   - Erreurs techniques

4. **Guidance procédurale** :
   - Comment faire un transfert
   - Comment s'inscrire
   - Comment récupérer un code oublié

### 2. Systèmes de tickets et de classification

**Classification automatique de tickets**

La classification de tickets permet de router automatiquement les requêtes vers les bonnes équipes ou workflows.

**Taxonomie des catégories** (EasyTransfert) :

```
1. INFORMATION_GENERALE (30% du volume)
   - info_frais
   - info_limites
   - info_operateurs
   - info_horaires

2. PROBLEME_TRANSACTION (40% du volume)
   - transaction_echouee
   - transaction_incomplete
   - argent_non_recu
   - erreur_montant

3. PROBLEME_TECHNIQUE (15% du volume)
   - probleme_connexion
   - erreur_application
   - bug_interface

4. COMPTE_UTILISATEUR (10% du volume)
   - inscription
   - connexion
   - code_oublie
   - modification_profil

5. RECLAMATION (5% du volume)
   - insatisfaction
   - contestation_frais
   - demande_remboursement
```

**Approches de classification**

**Approche 1 : Classification supervisée traditionnelle**

Pipeline :
1. Prétraitement : nettoyage, tokenisation, stemming
2. Vectorisation : TF-IDF ou count vectorizer
3. Classification : SVM, Naive Bayes, Random Forest
4. Post-traitement : seuil de confiance, classe par défaut

Exemple avec SVM :
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# Vectorisation
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2
)
X_train = vectorizer.fit_transform(texts_train)

# Classification
clf = SVC(kernel='linear', probability=True)
clf.fit(X_train, y_train)

# Prédiction
X_test = vectorizer.transform(texts_test)
predictions = clf.predict(X_test)
confidences = clf.predict_proba(X_test)
```

Performance typique :
- Accuracy : 0.82-0.88
- F1-score macro : 0.78-0.85
- Avantage : Rapide, peu de données nécessaires
- Limite : Pas de compréhension contextuelle

**Approche 2 : Classification avec embeddings + Deep Learning**

Architecture :
```
Input (text)
   ↓
Embedding Layer (pré-entraîné CamemBERT)
   ↓
BiLSTM(256 units)
   ↓
Dropout(0.3)
   ↓
BiLSTM(128 units)
   ↓
Attention Layer
   ↓
Dense(64, ReLU)
   ↓
Dropout(0.3)
   ↓
Dense(num_classes, Softmax)
```

Performance typique :
- Accuracy : 0.90-0.95
- F1-score macro : 0.88-0.93
- Avantage : Meilleure compréhension contextuelle
- Limite : Plus de données nécessaires, coût computationnel

**Approche 3 : Zero-shot classification avec LLM**

Prompting :
```
Classifie la requête client suivante dans l'une de ces catégories :
- INFORMATION_GENERALE
- PROBLEME_TRANSACTION
- PROBLEME_TECHNIQUE
- COMPTE_UTILISATEUR
- RECLAMATION

Requête : "{user_query}"

Réponds uniquement avec le nom de la catégorie la plus appropriée.
Si aucune ne correspond, réponds "AUTRE".

Catégorie :
```

Performance typique :
- Accuracy : 0.85-0.92 (dépend du LLM)
- Avantage : Pas besoin de données d'entraînement
- Limite : Coût d'inférence, latence

### 3. Analyse de sentiment et détection d'intention

**Analyse de sentiment**

L'analyse de sentiment permet de détecter l'émotion du client (positif, négatif, neutre) pour prioriser et personnaliser les réponses.

**Niveaux de granularité** :

1. **Document-level** : Sentiment global du message
2. **Sentence-level** : Sentiment par phrase
3. **Aspect-based** : Sentiment par aspect (ex: "bon service mais frais élevés")

**Approches pour le français** :

**Approche 1 : Lexiques de sentiments**

Utilisation de lexiques annotés (FEEL, Polyglot) :
```python
positive_words = {"content", "satisfait", "merci", "parfait", "rapide"}
negative_words = {"problème", "insatisfait", "lent", "erreur", "nul"}

def sentiment_score(text):
    words = text.lower().split()
    pos_count = sum(1 for w in words if w in positive_words)
    neg_count = sum(1 for w in words if w in negative_words)
    
    if pos_count > neg_count:
        return "POSITIF"
    elif neg_count > pos_count:
        return "NEGATIF"
    else:
        return "NEUTRE"
```

Avantages : Simple, rapide, interprétable
Limites : Pas de contexte, sensible aux négations

**Approche 2 : Classification avec CamemBERT**

Fine-tuning de CamemBERT sur corpus de sentiments :
```python
from transformers import CamembertForSequenceClassification

model = CamembertForSequenceClassification.from_pretrained(
    'camembert-base',
    num_labels=3  # positif, négatif, neutre
)

# Fine-tuning sur corpus annoté
# ...

# Prédiction
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
sentiment = torch.argmax(outputs.logits, dim=1)
```

Performance :
- Accuracy : 0.88-0.93 sur corpus français
- Robuste aux négations et sarcasme

**Application au service client** :

Priorisation des tickets :
- Sentiment NEGATIF + Mots clés urgents → Priorité HAUTE
- Sentiment POSITIF → Réponse standard
- Sentiment NEUTRE → Priorité NORMALE

Personnalisation des réponses :
- NEGATIF : Ton empathique, excuse, solution rapide
- POSITIF : Ton amical, remerciement
- NEUTRE : Ton professionnel et neutre

**Détection d'intention**

La détection d'intention identifie ce que l'utilisateur veut accomplir.

**Taxonomie d'intentions** (EasyTransfert) :

```
1. Informatives :
   - demander_frais
   - demander_limites
   - demander_horaires
   - demander_compatibilite

2. Transactionnelles :
   - verifier_statut_transaction
   - suivre_transaction
   - signaler_probleme_transaction

3. Procédurales :
   - demander_procedure_transfert
   - demander_procedure_inscription
   - demander_aide_generale

4. Support :
   - signaler_bug
   - contacter_agent
   - demander_remboursement
```

**Multi-intention** : Un message peut contenir plusieurs intentions :
```
"Bonjour, je veux savoir les frais pour un transfert de 
 100000 FCFA et aussi vérifier le statut de ma transaction 
 TX12345678"

Intentions détectées :
- demander_frais (confiance: 0.92)
- verifier_statut_transaction (confiance: 0.95)
```

**Approche hiérarchique** :

```
Niveau 1 (Macro-intention) :
- INFORMATION
- ACTION
- PROBLEME

Niveau 2 (Intention spécifique) :
Si macro = INFORMATION :
  - info_frais, info_limites, info_horaires...
Si macro = ACTION :
  - faire_transfert, consulter_solde...
Si macro = PROBLEME :
  - transaction_echouee, bug_app...
```

Cette approche hiérarchique améliore la précision et la robustesse.

---

*[Suite du mémoire dans le fichier MEMOIRE_COMPLET_PARTIE3.md...]*
