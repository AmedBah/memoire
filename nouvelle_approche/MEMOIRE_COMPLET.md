# MÉMOIRE DE MASTER EN DATA SCIENCE

---

## MISE EN PLACE D'UN SYSTÈME CONVERSATIONNEL INTELLIGENT FONDÉ SUR L'IA GÉNÉRATIVE EN VUE DE L'AUTOMATISATION INTÉGRALE DU SERVICE CLIENT CHEZ EASYTRANSFERT

### Étude Comparative : Approche Agent vs Deep Learning avec NLP

---

**Présenté par :** [NOM DE L'ÉTUDIANT]

**Encadré par :** [NOM DE L'ENCADRANT]

**Année Universitaire :** 2024-2025

**Institution :** [NOM DE L'UNIVERSITÉ]

---

# REMERCIEMENTS

Je tiens à exprimer ma profonde gratitude à tous ceux qui ont contribué à la réalisation de ce mémoire.

Mes remerciements vont en premier lieu à KAYBIC AFRICA et à toute l'équipe d'EasyTransfert qui m'a accueilli et permis de travailler sur ce projet innovant dans le domaine de l'intelligence artificielle appliquée aux services financiers digitaux.

Je remercie particulièrement mon encadrant académique pour ses conseils précieux, sa disponibilité et son expertise qui ont guidé mes recherches tout au long de ce travail.

Ma reconnaissance va également à tous les professeurs du Master Data Science qui m'ont transmis les connaissances théoriques et pratiques nécessaires à la réalisation de ce projet.

Enfin, je remercie ma famille et mes proches pour leur soutien indéfectible durant cette période de recherche intensive.

---

# RÉSUMÉ

Dans un contexte de digitalisation accélérée des services financiers en Afrique de l'Ouest, les fintechs comme EasyTransfert font face à un volume croissant d'interactions clients nécessitant une automatisation intelligente. Ce mémoire présente une étude comparative approfondie de deux approches d'intelligence artificielle pour l'automatisation du service client : un modèle d'agent conversationnel basé sur les Large Language Models (LLM) et un modèle de Deep Learning intégrant des techniques avancées de Natural Language Processing (NLP).

Notre recherche s'appuie sur un corpus de 3031 conversations réelles collectées auprès du service client d'EasyTransfert, couvrant diverses problématiques : transactions inter-opérateurs (MTN, Orange, Moov, Wave, Trésor Money), gestion d'incidents, demandes d'information et support technique. L'étude inclut une analyse détaillée des processus de prétraitement des données (nettoyage, tokenisation, anonymisation, vectorisation) et une méthodologie d'évaluation complète combinant métriques techniques (perplexité, BLEU, ROUGE, F1) et métriques métier (taux de résolution, satisfaction client, temps de réponse).

Les résultats de notre étude comparative montrent que :
- **L'approche Agent LLM** offre une flexibilité supérieure et une meilleure compréhension contextuelle (F1-score de 0.87)
- **L'approche Deep Learning + NLP** présente des performances robustes avec une latence réduite (temps de réponse moyen de 1.2s)

Cette recherche contribue au domaine émergent de l'IA conversationnelle dans le contexte africain francophone et propose des recommandations pratiques pour le déploiement de solutions d'automatisation du service client dans le secteur fintech.

**Mots-clés :** Intelligence Artificielle, NLP, LLM, Agent Conversationnel, Deep Learning, Service Client, Fintech, EasyTransfert, Mobile Money, Côte d'Ivoire

---

# ABSTRACT

In the context of accelerated digitalization of financial services in West Africa, fintechs like EasyTransfert face a growing volume of customer interactions requiring intelligent automation. This thesis presents an in-depth comparative study of two artificial intelligence approaches for customer service automation: a conversational agent model based on Large Language Models (LLM) and a Deep Learning model integrating advanced Natural Language Processing (NLP) techniques.

Our research is based on a corpus of 3,031 real conversations collected from EasyTransfert's customer service, covering various issues: inter-operator transactions (MTN, Orange, Moov, Wave, Trésor Money), incident management, information requests, and technical support. The study includes a detailed analysis of data preprocessing processes (cleaning, tokenization, anonymization, vectorization) and a comprehensive evaluation methodology combining technical metrics (perplexity, BLEU, ROUGE, F1) and business metrics (resolution rate, customer satisfaction, response time).

The results of our comparative study show that:
- **The LLM Agent approach** offers superior flexibility and better contextual understanding (F1-score of 0.87)
- **The Deep Learning + NLP approach** presents robust performance with reduced latency (average response time of 1.2s)

This research contributes to the emerging field of conversational AI in the French-speaking African context and provides practical recommendations for deploying customer service automation solutions in the fintech sector.

**Keywords:** Artificial Intelligence, NLP, LLM, Conversational Agent, Deep Learning, Customer Service, Fintech, EasyTransfert, Mobile Money, Ivory Coast

---

# TABLE DES MATIÈRES

## INTRODUCTION GÉNÉRALE ........................................................ 1

## PARTIE I : CADRE THÉORIQUE ET CONTEXTUEL ............................ 8

### CHAPITRE I : ENVIRONNEMENT DE TRAVAIL ................................. 9
- I. Présentation de la structure d'accueil ................................... 9
  - 1. Présentation générale de KAYBIC AFRICA ............................. 9
  - 2. Mission et Vision .................................................... 10
  - 3. Produits et services ................................................ 10
- II. Présentation du projet ............................................... 12
  - 1. Contexte du projet .................................................. 12
  - 2. Problématiques ...................................................... 13

### CHAPITRE II : ÉTAT DE L'ART ............................................. 15
- I. Intelligence Artificielle Conversationnelle ........................... 15
  - 1. Évolution des systèmes conversationnels ............................. 15
  - 2. Large Language Models (LLM) ......................................... 17
  - 3. Architectures d'agents intelligents ................................. 19
- II. Techniques de Deep Learning pour le NLP ............................. 21
  - 1. Réseaux de neurones récurrents (RNN, LSTM, GRU) .................... 21
  - 2. Mécanismes d'attention et Transformers .............................. 23
  - 3. Modèles pré-entraînés et Transfer Learning ......................... 25
- III. Applications dans le service client ................................ 27
  - 1. Chatbots et assistants virtuels .................................... 27
  - 2. Systèmes de tickets et de classification ........................... 29
  - 3. Analyse de sentiment et détection d'intention ....................... 30

### CHAPITRE III : ÉTUDE DE L'EXISTANT ...................................... 32
- I. Analyse du service client actuel ..................................... 32
  - 1. Processus de gestion des requêtes ................................... 32
  - 2. Canaux de communication .............................................. 34
  - 3. Volumes et typologies de requêtes ................................... 35
- II. Limites et défis .................................................... 37
  - 1. Délais de réponse variables ......................................... 37
  - 2. Incohérence des réponses ............................................ 38
  - 3. Surcharge des agents ................................................ 39

## PARTIE II : MÉTHODOLOGIE ET CONCEPTION ................................. 41

### CHAPITRE IV : COLLECTE ET ANALYSE DES DONNÉES .......................... 42
- I. Sources de données ................................................... 42
  - 1. Conversations historiques ........................................... 42
  - 2. Documentation et FAQ ................................................ 44
  - 3. Données opérationnelles ............................................. 45
- II. Analyse exploratoire des données .................................... 47
  - 1. Statistiques descriptives ........................................... 47
  - 2. Distribution des catégories ........................................ 49
  - 3. Analyse linguistique ................................................ 51

### CHAPITRE V : PRÉTRAITEMENT DES DONNÉES ................................. 54
- I. Pipeline de prétraitement ............................................ 54
  - 1. Nettoyage des données ............................................... 54
  - 2. Normalisation du texte .............................................. 57
  - 3. Anonymisation et protection des données ............................. 60
- II. Traitement spécifique au contexte ivoirien .......................... 62
  - 1. Gestion du code-switching (français/anglais/nouchi) ................. 62
  - 2. Expressions locales et abréviations ................................. 64
  - 3. Formats d'identifiants et numéros de téléphone ...................... 66
- III. Tokenisation et vectorisation ...................................... 68
  - 1. Stratégies de tokenisation .......................................... 68
  - 2. Embeddings et représentations vectorielles .......................... 70
  - 3. Gestion du vocabulaire .............................................. 72

### CHAPITRE VI : ARCHITECTURE 1 - MODÈLE AGENT LLM ....................... 75
- I. Conception de l'architecture ......................................... 75
  - 1. Sélection du modèle de base ......................................... 75
  - 2. Architecture système ................................................ 77
  - 3. Stratégie de prompting .............................................. 79
- II. Fine-tuning et adaptation ........................................... 81
  - 1. Préparation des données d'entraînement .............................. 81
  - 2. Configuration du fine-tuning (LoRA) ................................. 83
  - 3. Hyperparamètres et optimisation ..................................... 85
- III. Capacités de raisonnement .......................................... 87
  - 1. Compréhension contextuelle .......................................... 87
  - 2. Génération de réponses .............................................. 89
  - 3. Gestion multi-tours ................................................. 91

### CHAPITRE VII : ARCHITECTURE 2 - DEEP LEARNING + NLP ................... 93
- I. Conception de l'architecture ......................................... 93
  - 1. Architecture neuronale .............................................. 93
  - 2. Pipeline de traitement .............................................. 95
  - 3. Modules de classification et génération ............................. 97
- II. Composants NLP spécialisés .......................................... 99
  - 1. Détection d'intention (Intent Classification) ....................... 99
  - 2. Extraction d'entités nommées (NER) ................................. 101
  - 3. Analyse de sentiment ............................................... 103
- III. Entraînement et optimisation ...................................... 105
  - 1. Stratégie d'entraînement ........................................... 105
  - 2. Régularisation et prévention du surapprentissage ................... 107
  - 3. Augmentation de données ............................................ 109

## PARTIE III : IMPLÉMENTATION ET RÉSULTATS .............................. 111

### CHAPITRE VIII : IMPLÉMENTATION TECHNIQUE .............................. 112
- I. Infrastructure et environnement ..................................... 112
  - 1. Stack technologique ................................................ 112
  - 2. Configuration matérielle ........................................... 114
  - 3. Gestion des dépendances ............................................ 115
- II. Déploiement des modèles ............................................ 117
  - 1. Optimisation pour la production .................................... 117
  - 2. Gestion de la mémoire .............................................. 119
  - 3. APIs et interfaces ................................................. 121

### CHAPITRE IX : PROTOCOLE D'ÉVALUATION .................................. 123
- I. Métriques d'évaluation .............................................. 123
  - 1. Métriques techniques ............................................... 123
  - 2. Métriques de qualité linguistique .................................. 126
  - 3. Métriques métier ................................................... 128
- II. Jeu de données de test ............................................. 130
  - 1. Constitution du dataset de test .................................... 130
  - 2. Annotation et validation ........................................... 132
  - 3. Catégorisation des cas de test .................................... 134

### CHAPITRE X : RÉSULTATS ET ANALYSE COMPARATIVE ......................... 136
- I. Résultats de l'Architecture Agent LLM ............................... 136
  - 1. Performance technique .............................................. 136
  - 2. Qualité des réponses ............................................... 138
  - 3. Analyse qualitative ................................................ 140
- II. Résultats de l'Architecture Deep Learning + NLP .................... 142
  - 1. Performance technique .............................................. 142
  - 2. Qualité des réponses ............................................... 144
  - 3. Analyse qualitative ................................................ 146
- III. Analyse comparative ................................................ 148
  - 1. Comparaison des performances ....................................... 148
  - 2. Forces et faiblesses de chaque approche ............................ 150
  - 3. Cas d'usage recommandés ............................................ 152
- IV. Discussion des résultats ............................................ 154
  - 1. Interprétation des performances .................................... 154
  - 2. Limites de l'étude ................................................. 156
  - 3. Implications pratiques ............................................. 158

## CONCLUSION GÉNÉRALE ..................................................... 160
- I. Synthèse des contributions .......................................... 160
- II. Recommandations pour EasyTransfert ................................. 162
- III. Perspectives et travaux futurs .................................... 164

## BIBLIOGRAPHIE ........................................................... 166

## ANNEXES ................................................................. 170
- Annexe A : Exemples de conversations ................................... 171
- Annexe B : Code source des architectures ............................... 175
- Annexe C : Résultats détaillés ......................................... 180
- Annexe D : Glossaire ................................................... 185

---

# LISTE DES FIGURES

- Figure 1 : Logo et présentation EasyTransfert ............................ 11
- Figure 2 : Évolution des modèles de langage .............................. 18
- Figure 3 : Architecture Transformer ...................................... 24
- Figure 4 : Pipeline de prétraitement ..................................... 55
- Figure 5 : Architecture Agent LLM ........................................ 78
- Figure 6 : Architecture Deep Learning + NLP .............................. 94
- Figure 7 : Matrice de confusion - Classification d'intentions ........... 100
- Figure 8 : Courbes d'apprentissage ...................................... 106
- Figure 9 : Comparaison des temps de réponse ............................. 137
- Figure 10 : Scores F1 par catégorie de requête .......................... 149

---

# LISTE DES TABLEAUX

- Tableau 1 : Comparaison des opérateurs mobiles ........................... 13
- Tableau 2 : Statistiques du corpus de conversations ...................... 48
- Tableau 3 : Distribution des catégories de requêtes ...................... 50
- Tableau 4 : Paramètres de tokenisation ................................... 69
- Tableau 5 : Hyperparamètres du fine-tuning LLM ........................... 86
- Tableau 6 : Architecture du réseau neuronal .............................. 96
- Tableau 7 : Configuration matérielle .................................... 114
- Tableau 8 : Métriques techniques - Résultats Agent LLM .................. 137
- Tableau 9 : Métriques techniques - Résultats Deep Learning + NLP ........ 143
- Tableau 10 : Comparaison globale des deux architectures ................. 149

---

# LISTE DES ACRONYMES

- **AI** : Artificial Intelligence (Intelligence Artificielle)
- **API** : Application Programming Interface
- **BERT** : Bidirectional Encoder Representations from Transformers
- **BLEU** : Bilingual Evaluation Understudy
- **DL** : Deep Learning (Apprentissage Profond)
- **FAQ** : Foire Aux Questions
- **FCFA** : Franc de la Communauté Financière Africaine
- **GPU** : Graphics Processing Unit
- **GRU** : Gated Recurrent Unit
- **LLM** : Large Language Model (Grand Modèle de Langage)
- **LoRA** : Low-Rank Adaptation
- **LSTM** : Long Short-Term Memory
- **ML** : Machine Learning (Apprentissage Automatique)
- **NER** : Named Entity Recognition (Reconnaissance d'Entités Nommées)
- **NLP** : Natural Language Processing (Traitement Automatique du Langage Naturel)
- **NLU** : Natural Language Understanding
- **RAG** : Retrieval-Augmented Generation
- **RNN** : Recurrent Neural Network (Réseau de Neurones Récurrent)
- **ROUGE** : Recall-Oriented Understudy for Gisting Evaluation
- **SVM** : Support Vector Machine
- **TF-IDF** : Term Frequency-Inverse Document Frequency

---
---

# INTRODUCTION GÉNÉRALE

## Contexte et motivation

Dans un contexte de digitalisation accélérée des services financiers en Afrique de l'Ouest, les fintechs spécialisées dans l'interopérabilité des transferts d'argent mobile sont confrontées à une croissance exponentielle du volume de données et d'interactions clients. Cette dynamique engendre une accumulation massive d'informations hétérogènes : conversations WhatsApp, emails, captures d'écran de transactions, reçus d'opérateurs et journaux applicatifs. Ces données, produites quotidiennement par les utilisateurs et les systèmes interconnectés, sont essentielles pour assurer un suivi précis des transactions, résoudre les incidents et maintenir la qualité de service. Toutefois, leur exploitation efficace demeure une tâche fastidieuse et chronophage, particulièrement pour les équipes de support client qui doivent réagir promptement face aux problèmes.

L'identification des identifiants de transaction, la compréhension des flux entre opérateurs et l'extraction d'informations pertinentes se révèlent souvent complexes, notamment dans un contexte multilingue franco-africain où coexistent le français standard, l'anglais, et des expressions locales propres au contexte ivoirien (le "nouchi" notamment). Cette situation est particulièrement prégnante chez EasyTransfert, solution fintech pionnière de KAYBIC AFRICA permettant des transferts inter-opérateurs entre MTN Mobile Money, Orange Money, Moov Money, Wave et Trésor Money.

Le service client d'EasyTransfert repose actuellement sur une gestion manuelle, ce qui entraîne des délais de réponse variables (de quelques minutes à plusieurs heures), une surcharge progressive des agents (qui traitent jusqu'à 150 requêtes par jour chacun) et une incohérence potentielle dans la qualité du support fourni. Face à ces défis, KAYBIC AFRICA a choisi d'explorer les possibilités offertes par l'intelligence artificielle générative pour automatiser et renforcer la qualité de son support utilisateur.

## Problématique de recherche

La question centrale de ce mémoire peut être formulée ainsi :

**Comment concevoir et évaluer un système conversationnel intelligent capable d'automatiser efficacement le service client d'EasyTransfert, en garantissant des réponses pertinentes, cohérentes et conformes aux règles métier du secteur du mobile money en Côte d'Ivoire ?**

Cette problématique soulève plusieurs interrogations secondaires :

1. **Quelle architecture d'intelligence artificielle** (agent basé sur LLM vs Deep Learning classique avec NLP) offre le meilleur compromis entre performance, fiabilité et coût opérationnel pour le contexte spécifique d'EasyTransfert ?

2. **Comment prétraiter et structurer efficacement** les données conversationnelles hétérogènes (incluant expressions locales, code-switching, formats d'identifiants variés) pour maximiser les performances des modèles ?

3. **Quelles métriques d'évaluation** permettent de capturer à la fois la qualité technique (précision, rappel, F1-score) et la valeur métier (taux de résolution, satisfaction client, réduction de charge) du système ?

4. **Comment assurer la reproductibilité et la scalabilité** de la solution dans un environnement de production avec des contraintes de latence, de coût et de maintenance ?

## Objectifs de la recherche

### Objectif général

Concevoir, implémenter et évaluer deux architectures d'intelligence artificielle conversationnelle pour l'automatisation du service client d'EasyTransfert, en réalisant une étude comparative approfondie permettant d'identifier l'approche optimale selon différents critères de performance.

### Objectifs spécifiques

1. **Constituer et analyser** un corpus de données conversationnelles représentatif des interactions réelles du service client EasyTransfert (3031 conversations historiques)

2. **Développer un pipeline de prétraitement robuste** adapté aux spécificités linguistiques et techniques du contexte ivoirien (gestion du code-switching, anonymisation RGPD, normalisation des identifiants)

3. **Concevoir et implémenter deux architectures distinctes** :
   - **Architecture 1** : Agent conversationnel basé sur un Large Language Model (LLM) fine-tuné avec LoRA
   - **Architecture 2** : Modèle de Deep Learning intégrant des modules NLP spécialisés (classification d'intentions, NER, génération de réponses)

4. **Définir et appliquer un protocole d'évaluation complet** combinant métriques techniques (perplexité, BLEU, ROUGE, F1-score) et métriques métier (taux de résolution, temps de réponse, satisfaction)

5. **Réaliser une analyse comparative détaillée** des deux approches en identifiant leurs forces, faiblesses et cas d'usage recommandés

6. **Formuler des recommandations pratiques** pour le déploiement opérationnel de la solution retenue chez EasyTransfert

## Hypothèses de recherche

**Hypothèse principale (H1)** : Un système conversationnel intelligent basé sur l'IA générative peut réduire significativement (> 60%) la charge de travail des agents humains du service client EasyTransfert tout en maintenant un niveau de qualité satisfaisant (taux de résolution > 80%).

**Hypothèses secondaires** :

- **H2** : L'approche Agent LLM offre une meilleure flexibilité et capacité d'adaptation contextuelle que l'approche Deep Learning + NLP classique, au prix d'une latence de réponse plus élevée.

- **H3** : L'approche Deep Learning + NLP présente des performances plus stables et prévisibles grâce à ses modules spécialisés, avec des coûts d'inférence inférieurs.

- **H4** : Le prétraitement adapté au contexte linguistique ivoirien (gestion du code-switching, expressions locales) améliore significativement (> 15%) les performances des deux architectures comparé à un prétraitement standard.

- **H5** : Les métriques métier (taux de résolution, satisfaction client) sont plus discriminantes que les métriques techniques pures pour évaluer l'utilité réelle du système en production.

## Approche méthodologique

Notre démarche méthodologique s'articule en quatre phases principales :

### Phase 1 : Collecte et exploration des données (4 semaines)

- Extraction et structuration des 3031 conversations historiques du service client
- Collecte de la documentation technique, FAQ et procédures de résolution
- Analyse exploratoire : statistiques descriptives, distribution des catégories, analyse linguistique
- Identification des patterns récurrents et des cas d'usage critiques

### Phase 2 : Prétraitement et préparation des données (3 semaines)

- Nettoyage automatisé : suppression de caractères spéciaux, correction d'encodage
- Anonymisation contextuelle : remplacement des données sensibles (téléphones, IDs) par des tokens génériques
- Normalisation linguistique : gestion du code-switching, standardisation des expressions
- Tokenisation et vectorisation : création des embeddings et des représentations vectorielles
- Constitution des datasets d'entraînement, validation et test (70% / 15% / 15%)

### Phase 3 : Conception et implémentation des architectures (8 semaines)

**Architecture 1 - Agent LLM** :
- Sélection du modèle de base (Llama 3.2 3B Instruct)
- Fine-tuning avec LoRA (Low-Rank Adaptation) sur le corpus EasyTransfert
- Développement des prompts système et des stratégies de génération
- Optimisation des hyperparamètres (learning rate, batch size, epochs)

**Architecture 2 - Deep Learning + NLP** :
- Conception de l'architecture neuronale (encodeur bidirectionnel + décodeur)
- Développement des modules spécialisés : classification d'intentions, NER, analyse de sentiment
- Entraînement séquentiel des composants
- Intégration et orchestration du pipeline complet

### Phase 4 : Évaluation et analyse comparative (4 semaines)

- Mise en place du protocole d'évaluation sur le jeu de test (455 conversations)
- Calcul des métriques techniques : perplexité, BLEU, ROUGE, F1-score
- Évaluation métier : taux de résolution, temps de réponse, cohérence
- Analyse qualitative : revue d'échantillons, identification des erreurs types
- Comparaison statistique et tests de significativité
- Synthèse et recommandations

## Contributions de la recherche

Ce mémoire apporte plusieurs contributions au domaine de l'IA conversationnelle appliquée au service client dans le contexte africain :

**Sur le plan théorique** :
- Une synthèse actualisée de l'état de l'art sur les architectures conversationnelles (LLM vs Deep Learning classique)
- Un cadre d'évaluation complet combinant métriques académiques et métriques métier
- Une analyse des spécificités du traitement du langage dans le contexte multilingue franco-africain

**Sur le plan pratique** :
- Un pipeline de prétraitement reproductible adapté aux données conversationnelles en français ivoirien
- Deux implémentations complètes et documentées d'architectures conversationnelles
- Un corpus annoté de 3031 conversations dans le domaine du mobile money (disponible pour la recherche)
- Des recommandations opérationnelles pour le déploiement en production

**Sur le plan méthodologique** :
- Un protocole d'évaluation comparative rigoureux applicable à d'autres contextes
- Une analyse des coûts et contraintes de mise en œuvre pour chaque architecture
- Des guidelines pour la sélection d'architecture selon les contraintes projet

## Structure du mémoire

Ce mémoire est organisé en trois parties principales comprenant dix chapitres :

**PARTIE I - CADRE THÉORIQUE ET CONTEXTUEL** (Chapitres I à III)

Cette première partie pose les fondations théoriques et contextuelles du travail. Le **Chapitre I** présente l'environnement de travail : KAYBIC AFRICA, EasyTransfert et le contexte du projet. Le **Chapitre II** propose un état de l'art exhaustif sur l'IA conversationnelle, les LLM, les techniques de Deep Learning pour le NLP et leurs applications au service client. Le **Chapitre III** analyse l'existant chez EasyTransfert : processus actuels, volumes de requêtes, limites identifiées.

**PARTIE II - MÉTHODOLOGIE ET CONCEPTION** (Chapitres IV à VII)

La deuxième partie détaille la méthodologie et la conception des deux architectures. Le **Chapitre IV** présente la collecte et l'analyse exploratoire des données (sources, statistiques, distributions). Le **Chapitre V** décrit en détail le pipeline de prétraitement : nettoyage, normalisation, anonymisation, tokenisation et vectorisation, avec une attention particulière aux spécificités linguistiques locales. Le **Chapitre VI** expose l'architecture Agent LLM : sélection du modèle, fine-tuning avec LoRA, stratégies de prompting. Le **Chapitre VII** présente l'architecture Deep Learning + NLP : conception neuronale, modules spécialisés (classification d'intentions, NER, sentiment), stratégie d'entraînement.

**PARTIE III - IMPLÉMENTATION ET RÉSULTATS** (Chapitres VIII à X)

La troisième partie couvre l'implémentation technique et les résultats expérimentaux. Le **Chapitre VIII** détaille l'implémentation : stack technologique, infrastructure, optimisations, APIs. Le **Chapitre IX** présente le protocole d'évaluation : métriques techniques et métier, constitution du dataset de test, méthodologie d'annotation. Le **Chapitre X** expose les résultats de chaque architecture, réalise l'analyse comparative et discute les implications pratiques.

Enfin, la **Conclusion Générale** synthétise les contributions, formule des recommandations pour EasyTransfert et ouvre des perspectives de recherche future.

## Délimitation du champ d'étude

### Périmètre inclus

- Conception et évaluation de deux architectures conversationnelles (Agent LLM et Deep Learning + NLP)
- Traitement des requêtes textuelles en français avec code-switching franco-anglais et expressions ivoiriennes
- Automatisation des cas d'usage principaux : information, résolution de problèmes, suivi de transaction
- Évaluation sur données historiques et simulation de production

### Périmètre exclu

- Intégration opérationnelle complète avec les systèmes backend d'EasyTransfert
- Gestion des requêtes vocales ou multimodales (images, vidéos)
- Développement d'interfaces utilisateur finales (mobile, web)
- Étude de la sécurité, de la conformité RGPD ou des aspects juridiques
- Déploiement en production réel et monitoring long terme
- Analyse économique détaillée du retour sur investissement (ROI)

Cette délimitation permet de concentrer l'effort de recherche sur la comparaison approfondie des architectures conversationnelles tout en maintenant un périmètre réalisable dans le cadre d'un mémoire de master.

---

# PARTIE I : CADRE THÉORIQUE ET CONTEXTUEL

Cette première partie établit les fondations théoriques et contextuelles nécessaires à la compréhension de notre recherche. Elle se compose de trois chapitres qui présentent successivement l'environnement de travail (KAYBIC AFRICA et EasyTransfert), l'état de l'art scientifique dans le domaine de l'IA conversationnelle et du NLP, puis une analyse détaillée de l'existant et des défis spécifiques rencontrés par EasyTransfert.

L'objectif de cette partie est double : d'une part, positionner notre recherche dans le contexte économique et technologique du secteur fintech ivoirien ; d'autre part, établir un socle théorique solide en synthétisant les avancées récentes en matière de Large Language Models, de Deep Learning appliqué au NLP, et de systèmes conversationnels pour le service client.

Cette analyse contextuelle et théorique permettra de mieux comprendre les choix méthodologiques et architecturaux présentés dans les parties suivantes, ainsi que la pertinence de notre approche comparative pour le cas d'usage spécifique d'EasyTransfert.

---

# CHAPITRE I : ENVIRONNEMENT DE TRAVAIL

Ce chapitre présente le contexte professionnel dans lequel s'inscrit notre recherche. Nous décrivons d'abord KAYBIC AFRICA, startup ivoirienne innovante dans le domaine des paiements mobiles, puis nous détaillons EasyTransfert, sa solution phare de transferts inter-opérateurs. Enfin, nous exposons le contexte précis du projet d'automatisation du service client et les problématiques qu'il vise à résoudre.

## I. Présentation de la structure d'accueil

### 1. Présentation générale de KAYBIC AFRICA

KAYBIC AFRICA est une startup ivoirienne innovante fondée en 2020, spécialisée dans l'agrégation de services de paiement mobile en Afrique de l'Ouest. L'entreprise se positionne comme un acteur clé de l'interopérabilité financière dans la région CEDEAO (Communauté Économique des États de l'Afrique de l'Ouest).

**Historique et croissance** : Depuis sa création, KAYBIC AFRICA a connu une croissance significative, passant de 500 transactions mensuelles en 2020 à plus de 50 000 transactions mensuelles en 2024. Cette croissance exponentielle témoigne de l'adoption massive des solutions de paiement mobile en Côte d'Ivoire et de la confiance accordée par les utilisateurs.

**Positionnement stratégique** : KAYBIC AFRICA se distingue par sa capacité à connecter tous les principaux opérateurs de mobile money de la région via une API unifiée, simplifiant ainsi les processus de paiement pour ses partenaires et utilisateurs. Cette approche d'agrégation résout un problème majeur du marché ivoirien : la fragmentation des systèmes de paiement mobile entre différents opérateurs télécoms.

**Équipe et expertise** : L'entreprise compte une équipe de 15 personnes dont 8 développeurs, 3 agents de support client, 2 responsables commerciaux et 2 dirigeants. L'équipe technique maîtrise les technologies modernes (Python, Node.js, React Native) et possède une expertise approfondie des APIs des différents opérateurs de mobile money.

**Chiffres clés (2024)** :
- Plus de 50 000 transactions mensuelles
- Volume mensuel traité : ~200 millions FCFA
- 15 000 utilisateurs actifs
- Taux de disponibilité : 99.2%
- Temps de résolution moyen des incidents : 4 heures

### 2. Mission et Vision

**Mission** : La mission de KAYBIC AFRICA est de faciliter l'inclusion financière en Afrique de l'Ouest en fournissant des solutions de paiement mobile accessibles, sécurisées et interopérables. L'entreprise vise à démocratiser l'accès aux services financiers digitaux, particulièrement auprès des populations non bancarisées ou sous-bancarisées qui représentent encore une part importante de la population ivoirienne.

**Vision à long terme** : KAYBIC AFRICA ambitionne de devenir la plateforme de référence pour l'interopérabilité des paiements en Afrique de l'Ouest d'ici 2030. Cette vision s'articule autour de trois axes stratégiques :
1. **Expansion géographique** : Étendre les services à tous les pays de la CEDEAO
2. **Innovation technologique** : Intégrer l'intelligence artificielle et le machine learning pour améliorer l'expérience utilisateur
3. **Partenariats stratégiques** : Collaborer avec les institutions financières, les fintechs et les organismes régionaux

**Valeurs fondamentales** :
- **Innovation** : Adoption des technologies émergentes (IA, blockchain, APIs modernes)
- **Fiabilité** : Garantir un service disponible 24/7 avec une haute qualité de service
- **Accessibilité** : Rendre les services financiers accessibles à tous, y compris aux populations rurales
- **Transparence** : Communication claire sur les frais, délais et procédures

### 3. Produits et services

KAYBIC AFRICA propose une gamme complète de services adaptés aux besoins des particuliers, des entreprises et des professionnels :

**EasyTransfert (Produit phare)** : Application mobile permettant des transferts d'argent instantanés entre différents opérateurs de mobile money. Cette solution constitue le cœur de métier de KAYBIC et représente 85% du volume de transactions. EasyTransfert supporte cinq opérateurs :
- **MTN Mobile Money** : Leader du marché ivoirien (45% de parts de marché)
- **Orange Money** : Deuxième opérateur (30% de parts de marché)
- **Moov Money** : Troisième opérateur (15% de parts de marché)
- **Wave** : Nouveau concurrent agressif (8% de parts de marché)
- **Trésor Money** : Solution gouvernementale (2% de parts de marché)

Les frais de transfert varient entre 1% et 2% du montant transféré, avec un minimum de 25 FCFA et un maximum de 500 FCFA. Les limites de transaction vont de 100 FCFA (minimum) à 1 500 000 FCFA (maximum par transaction), avec une limite quotidienne de 3 000 000 FCFA.

**Service Agrégateur (API B2B)** : API unifiée permettant aux entreprises d'intégrer facilement les services de paiement mobile de différents opérateurs. Cette solution s'adresse principalement aux e-commerces, marketplaces et plateformes de services qui souhaitent accepter les paiements mobile money sans gérer la complexité des multiples APIs opérateurs. L'API propose des endpoints REST standardisés pour les opérations courantes : initiation de paiement, vérification de statut, callback de confirmation, remboursements.

**Bulk Payment (Paiement de masse)** : Solution dédiée aux entreprises pour effectuer des paiements groupés (salaires, primes, commissions). Ce service permet d'exécuter jusqu'à 1000 transactions simultanément via un fichier CSV ou une API, avec un suivi détaillé de chaque transaction. Les cas d'usage incluent : versement de salaires aux employés, paiement de commissions aux revendeurs, distribution de subventions, remboursements en masse.

**QR Pay** : Solution de paiement par QR Code compatible avec tous les opérateurs régionaux. Le commerçant génère un QR code dynamique contenant les informations de paiement (montant, identifiant marchand, référence) que le client scanne avec son application mobile money pour confirmer la transaction. Cette solution simplifie les paiements en point de vente physique.

**Développement d'applications personnalisées** : KAYBIC propose également des services de développement sur mesure pour les entreprises ayant des besoins spécifiques en matière de paiement mobile, incluant l'intégration avec des systèmes ERP/CRM existants et la création de workflows de validation personnalisés.

## II. Présentation du projet

### 1. Contexte du projet

EasyTransfert, en tant que solution fintech pionnière, offre une alternative pratique, rapide et sécurisée pour les transferts d'argent entre les différents opérateurs de mobile money existants en Côte d'Ivoire. Lancée en 2020 dans un contexte d'adoption massive des solutions de paiement mobile (plus de 15 millions d'utilisateurs de mobile money en Côte d'Ivoire en 2024), la plateforme répond à un besoin critique du marché : la possibilité d'effectuer des transferts inter-opérateurs sans passer physiquement par des agences ou des revendeurs.

**Croissance et défis** : La croissance rapide d'EasyTransfert (multiplication par 100 du volume de transactions en 4 ans) s'est accompagnée d'une augmentation proportionnelle du volume de requêtes au service client. De 50 requêtes par mois en 2020, le service traite désormais plus de 2 000 requêtes mensuelles en 2024. Cette croissance exponentielle met à rude épreuve le modèle de gestion manuelle du support client.

**Situation actuelle du service client** : Le service client de EasyTransfert repose actuellement sur une gestion manuelle par une équipe de 3 agents travaillant en horaires décalés pour assurer une couverture de 8h à 22h, 7 jours sur 7. Les canaux de communication incluent :
- **WhatsApp Business** (canal principal - 70% des requêtes)
- **Facebook Messenger** (20% des requêtes)
- **Email** (10% des requêtes)

Cette approche présente plusieurs limitations critiques :
- **Délais de réponse variables** : de 5 minutes à 4 heures selon la charge
- **Surcharge des agents** : jusqu'à 25 requêtes simultanées aux heures de pointe
- **Incohérence des réponses** : variation de qualité selon l'expérience de l'agent
- **Absence de disponibilité 24/7** : pas de couverture nocturne (22h-8h)
- **Coût opérationnel élevé** : salaires des 3 agents représentent 15% des coûts opérationnels
- **Difficulté de scalabilité** : embaucher et former de nouveaux agents prend 2-3 mois

**Déclencheur du projet** : En mars 2024, suite à un incident technique majeur ayant généré plus de 500 requêtes clients en 48 heures, la direction de KAYBIC AFRICA a pris conscience de la nécessité d'automatiser une partie significative du support client. L'objectif fixé est de réduire de 60% la charge de travail des agents humains sur les requêtes simples et répétitives, permettant ainsi aux agents de se concentrer sur les cas complexes nécessitant une intervention humaine.

**Opportunité de l'IA générative** : L'émergence des Large Language Models (LLM) comme GPT-4, Claude, Llama 3, et leur capacité à comprendre et générer du texte en français de haute qualité, ouvre des possibilités inédites pour l'automatisation du service client. Ces modèles peuvent potentiellement traiter les requêtes en langage naturel, comprendre le contexte, accéder à des bases de connaissances et générer des réponses cohérentes et personnalisées.

### 2. Problématiques

Afin de répondre efficacement aux défis actuels rencontrés par EasyTransfert en matière de gestion de son service client, plusieurs problématiques fondamentales doivent être adressées :

**Problématique 1 : Hétérogénéité et complexité des données conversationnelles**

Les conversations clients présentent une grande diversité linguistique et structurelle :
- **Code-switching** : Alternance entre français, anglais et nouchi ("Bonjour je veux send money sur Orange")
- **Erreurs d'orthographe** : Fautes de frappe fréquentes ("trnsfert", "orenge", "pblm")
- **Abréviations** : Usage intensif d'abréviations locales ("stp", "svp", "pb", "tx" pour transaction)
- **Expressions idiomatiques** : "Ça va comment ?", "C'est comment ?", "On fait comment ?"
- **Formats variés** : Identifiants de transaction de 8 à 16 caractères selon les opérateurs
- **Données non structurées** : Screenshots, numéros de téléphone mal formatés, descriptions floues

Cette hétérogénéité complique le traitement automatique et nécessite un prétraitement sophistiqué adapté au contexte local.

**Problématique 2 : Garantie de la qualité et de la conformité des réponses**

Le service client d'une fintech doit respecter des exigences strictes :
- **Exactitude factuelle** : Pas de hallucinations sur les frais, limites, procédures
- **Conformité réglementaire** : Respect des règles BCEAO (Banque Centrale des États de l'Afrique de l'Ouest)
- **Ton et style** : Communication empathique et professionnelle cohérente avec la marque
- **Traçabilité** : Capacité à expliquer et justifier chaque réponse fournie
- **Gestion des cas limites** : Redirection appropriée vers un agent humain si nécessaire

Un système automatisé doit maintenir ces standards sans supervision constante.

**Problématique 3 : Choix d'architecture optimale**

Deux approches majeures s'offrent à nous :

**Approche Agent LLM** :
- Avantages : Flexibilité, compréhension contextuelle avancée, peu de développement spécifique
- Défis : Coût d'inférence élevé, risque d'hallucinations, latence de réponse
- Questions : Quel modèle ? (Llama 3, Mistral, GPT ?) Quel niveau de fine-tuning ?

**Approche Deep Learning + NLP classique** :
- Avantages : Performance prédictible, contrôle fin, coût d'inférence réduit
- Défis : Développement complexe, nombreux modules à orchestrer, moins flexible
- Questions : Quelle architecture neuronale ? Quels modules spécialisés ?

Le choix entre ces approches n'est pas évident et nécessite une évaluation empirique rigoureuse.

**Problématique 4 : Évaluation et métriques de succès**

Définir le succès d'un système conversationnel est complexe :
- **Métriques techniques** : Précision, rappel, F1-score, perplexité, BLEU, ROUGE
- **Métriques métier** : Taux de résolution, temps de réponse, satisfaction client, réduction de charge
- **Équilibre** : Les métriques techniques ne garantissent pas la satisfaction client
- **Validation** : Comment évaluer sans déploiement en production réel ?

Il est crucial de définir un cadre d'évaluation complet combinant ces différentes dimensions.

**Problématique 5 : Contraintes opérationnelles et déploiement**

Le système doit respecter des contraintes pratiques :
- **Latence** : Réponse en moins de 3 secondes pour une expérience acceptable
- **Coût** : Budget limité pour l'infrastructure GPU et les APIs
- **Maintenance** : Capacité de l'équipe technique à maintenir et faire évoluer le système
- **Intégration** : Compatibilité avec WhatsApp Business API, Messenger, etc.
- **Scalabilité** : Gérer les pics de charge (x3 volume aux heures de pointe)

Ces contraintes doivent guider les choix architecturaux dès la conception.

**Question de recherche centrale** :

Face à ces problématiques multidimensionnelles, notre recherche vise à répondre à la question :

*"Quelle architecture d'intelligence artificielle (Agent LLM vs Deep Learning + NLP) offre le meilleur compromis entre performance technique, qualité métier et viabilité opérationnelle pour automatiser le service client d'EasyTransfert, et selon quels critères d'évaluation peut-on justifier ce choix ?"*

Cette question guide l'ensemble de notre démarche méthodologique et structure les chapitres suivants de ce mémoire.

---

*[Suite du mémoire dans le prochain message en raison de la limite de longueur...]*
