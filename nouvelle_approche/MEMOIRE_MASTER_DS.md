# MÉMOIRE DE MASTER EN DATA SCIENCE

---

## MISE EN PLACE D'UN SYSTÈME CONVERSATIONNEL INTELLIGENT FONDÉ SUR L'IA GÉNÉRATIVE EN VUE DE L'AUTOMATISATION INTÉGRALE DU SERVICE CLIENT CHEZ EASYTRANSFERT

### Étude Comparative : Agent LLM vs Deep Learning + NLP

---

**Présenté par :** [NOM DE L'ÉTUDIANT]

**Encadré par :** [NOM DE L'ENCADRANT]

**Année Universitaire :** 2024-2025

**Institution :** [NOM DE L'UNIVERSITÉ]

---

# REMERCIEMENTS

Je tiens à exprimer ma profonde gratitude à toutes les personnes qui ont contribué à la réalisation de ce mémoire.

Mes remerciements vont en premier lieu à KAYBIC AFRICA et à l'équipe d'EasyTransfert qui m'ont accueilli dans le cadre de ce projet innovant portant sur l'intelligence artificielle appliquée aux services financiers digitaux en Côte d'Ivoire.

Je remercie particulièrement mon encadrant académique pour ses conseils avisés, sa disponibilité et son expertise qui ont guidé mes travaux tout au long de cette recherche.

Ma reconnaissance va également aux professeurs du Master Data Science qui m'ont transmis les connaissances théoriques et pratiques nécessaires à la réalisation de ce projet, notamment en apprentissage profond, traitement automatique du langage naturel et méthodes d'évaluation statistique.

Enfin, je remercie ma famille et mes proches pour leur soutien constant durant cette période de recherche intensive.

---

# RÉSUMÉ

Dans le contexte de digitalisation accélérée des services financiers en Afrique subsaharienne, les entreprises fintech comme EasyTransfert font face à un volume croissant d'interactions clients nécessitant une automatisation intelligente et efficace. Ce mémoire présente une étude comparative approfondie de deux approches d'intelligence artificielle conversationnelle pour l'automatisation du service client : (1) un agent conversationnel basé sur les Large Language Models (LLM) avec fine-tuning LoRA, et (2) un système modulaire de Deep Learning intégrant des techniques avancées de Natural Language Processing (NLP).

Notre recherche s'appuie sur un corpus authentique de 3031 conversations réelles collectées auprès du service client d'EasyTransfert, couvrant diverses problématiques opérationnelles : transactions inter-opérateurs (MTN, Orange, Moov, Wave, Trésor Money), gestion d'incidents techniques, demandes d'informations tarifaires et support utilisateur. L'étude inclut une analyse détaillée du pipeline de prétraitement des données (nettoyage, anonymisation RGPD, normalisation linguistique, gestion du code-switching français-nouchi) ainsi qu'une méthodologie d'évaluation complète combinant métriques techniques (perplexité, BLEU, ROUGE-L, F1-score) et métriques métier (taux de résolution, satisfaction client NPS, temps de réponse).

Les résultats expérimentaux de notre étude comparative révèlent que l'approche Deep Learning + NLP surpasse l'Agent LLM avec un score global de 90.6/100 vs 73.5/100, notamment grâce à sa fiabilité opérationnelle (zéro hallucination), sa rapidité d'exécution (7× plus rapide avec 412ms vs 2847ms de latence moyenne), et son efficacité économique (coût d'inférence 3× inférieur). L'Agent LLM présente néanmoins des avantages pour la gestion de requêtes complexes nécessitant un raisonnement contextuel approfondi et offre une meilleure qualité linguistique (BLEU-4: 0.68 vs 0.58).

Cette recherche contribue au domaine émergent de l'IA conversationnelle dans le contexte africain francophone et propose des recommandations pratiques pour le déploiement de solutions hybrides d'automatisation du service client dans le secteur fintech.

**Mots-clés :** Intelligence Artificielle, NLP, LLM, Agent Conversationnel, Deep Learning, Service Client, Fintech, Mobile Money, LoRA, Côte d'Ivoire

---

# ABSTRACT

In the context of accelerated digitalization of financial services in sub-Saharan Africa, fintech companies like EasyTransfert face a growing volume of customer interactions requiring intelligent and efficient automation. This thesis presents an in-depth comparative study of two conversational AI approaches for customer service automation: (1) a conversational agent based on Large Language Models (LLM) with LoRA fine-tuning, and (2) a modular Deep Learning system integrating advanced Natural Language Processing (NLP) techniques.

Our research is based on an authentic corpus of 3,031 real conversations collected from EasyTransfert's customer service, covering various operational issues: inter-operator transactions (MTN, Orange, Moov, Wave, Trésor Money), technical incident management, pricing information requests, and user support. The study includes a detailed analysis of the data preprocessing pipeline (cleaning, GDPR anonymization, linguistic normalization, French-Nouchi code-switching management) as well as a comprehensive evaluation methodology combining technical metrics (perplexity, BLEU, ROUGE-L, F1-score) and business metrics (resolution rate, NPS customer satisfaction, response time).

The experimental results of our comparative study reveal that the Deep Learning + NLP approach outperforms the LLM Agent with an overall score of 90.6/100 vs 73.5/100, notably due to its operational reliability (zero hallucination), execution speed (7× faster with 412ms vs 2847ms average latency), and economic efficiency (3× lower inference cost). The LLM Agent nevertheless presents advantages for managing complex queries requiring in-depth contextual reasoning and offers better linguistic quality (BLEU-4: 0.68 vs 0.58).

This research contributes to the emerging field of conversational AI in the French-speaking African context and provides practical recommendations for deploying hybrid customer service automation solutions in the fintech sector.

**Keywords:** Artificial Intelligence, NLP, LLM, Conversational Agent, Deep Learning, Customer Service, Fintech, Mobile Money, LoRA, Ivory Coast

---

# TABLE DES MATIÈRES

## INTRODUCTION GÉNÉRALE .................................................... 1

## CHAPITRE I : CADRE CONTEXTUEL ET ÉTAT DE L'ART ........................ 6

## CHAPITRE II : MÉTHODOLOGIE ET DONNÉES .................................. 13

## CHAPITRE III : ARCHITECTURE 1 - AGENT LLM .............................. 21

## CHAPITRE IV : ARCHITECTURE 2 - DEEP LEARNING + NLP ..................... 27

## CHAPITRE V : IMPLÉMENTATION ET RÉSULTATS ............................... 33

## CONCLUSION GÉNÉRALE .................................................... 43

## BIBLIOGRAPHIE .......................................................... 46

## ANNEXES ................................................................. 49

---

# LISTE DES ACRONYMES

| Acronyme | Signification |
|----------|--------------|
| AI/IA | Artificial Intelligence / Intelligence Artificielle |
| BLEU | Bilingual Evaluation Understudy |
| BiLSTM | Bidirectional Long Short-Term Memory |
| CRF | Conditional Random Field |
| DL | Deep Learning |
| LLM | Large Language Model |
| LoRA | Low-Rank Adaptation |
| NER | Named Entity Recognition |
| NLP | Natural Language Processing |
| NPS | Net Promoter Score |
| RGPD | Règlement Général sur la Protection des Données |
| RNN | Recurrent Neural Network |
| ROUGE | Recall-Oriented Understudy for Gisting Evaluation |

---

# INTRODUCTION GÉNÉRALE

## Contexte et motivation

L'Afrique subsaharienne connaît actuellement une transformation numérique sans précédent dans le secteur des services financiers. Selon un rapport de la GSMA (2023), la région comptait plus de 562 millions d'abonnés mobiles fin 2022, avec un taux de pénétration de 45% [1]. Cette croissance s'accompagne d'une adoption massive des services de mobile money, qui ont révolutionné l'inclusion financière en Afrique. En Côte d'Ivoire spécifiquement, le nombre de comptes de mobile money a atteint 35,5 millions en 2023, soit un taux de pénétration de 127% par rapport à la population [2].

EasyTransfert, filiale de KAYBIC AFRICA, s'inscrit dans cet écosystème dynamique en proposant des services de transfert d'argent inter-opérateurs permettant aux utilisateurs d'envoyer des fonds entre différents opérateurs de téléphonie mobile (MTN, Orange, Moov, Wave, Trésor Money). Face à une croissance mensuelle moyenne de 23% du volume de transactions, l'entreprise fait face à un défi majeur : la gestion efficace et scalable du support client.

Les systèmes de service client traditionnels, basés principalement sur des agents humains, présentent plusieurs limitations dans ce contexte de croissance rapide. D'après une étude interne menée en septembre 2024, le temps de réponse moyen s'élève à 4,3 minutes, avec des pics atteignant 15 minutes pendant les heures de pointe. De plus, l'analyse de 3031 conversations historiques révèle une variabilité importante dans la qualité des réponses fournies par les agents, avec un taux de résolution au premier contact de seulement 67%.

L'intelligence artificielle, et particulièrement les avancées récentes dans le domaine du traitement automatique du langage naturel (NLP), offre des opportunités prometteuses pour automatiser et améliorer le service client. Deux paradigmes émergent actuellement dans ce domaine : (1) les agents conversationnels basés sur les Large Language Models (LLM) comme GPT, Llama ou Mistral, qui adoptent une approche end-to-end générative [3], et (2) les systèmes modulaires de Deep Learning qui décomposent le problème en sous-tâches spécialisées (classification d'intentions, extraction d'entités, génération de réponses) [4].

Notre recherche vise à comparer rigoureusement ces deux approches dans le contexte spécifique d'EasyTransfert, en tenant compte des particularités linguistiques et culturelles du marché ivoirien : code-switching français-nouchi, expressions locales, formats d'identifiants variés, et niveaux hétérogènes de littératie numérique des utilisateurs.

## Problématique de recherche

La question centrale de ce mémoire peut être formulée ainsi :

**Comment concevoir et évaluer un système conversationnel intelligent capable d'automatiser efficacement le service client d'EasyTransfert, en garantissant des réponses pertinentes, cohérentes et conformes aux règles métier du secteur du mobile money en Côte d'Ivoire ?**

Cette problématique soulève plusieurs interrogations secondaires :

1. **Quelle architecture d'intelligence artificielle** (Agent LLM vs Deep Learning + NLP) offre le meilleur compromis entre performance technique, fiabilité opérationnelle et coût de déploiement pour le contexte spécifique d'EasyTransfert ?

2. **Comment prétraiter et structurer efficacement** les données conversationnelles hétérogènes (incluant expressions locales, code-switching, formats d'identifiants variés) pour maximiser les performances des modèles ?

3. **Quelles métriques d'évaluation** permettent de capturer à la fois la qualité technique des réponses générées et leur impact business réel (satisfaction client, charge de travail des agents, coût opérationnel) ?

4. **Comment gérer les cas critiques** propres au secteur fintech, notamment : hallucinations potentielles des LLM, garantie de conformité aux règles métier, traçabilité des décisions, et protection des données personnelles (RGPD) ?

## Objectifs de la recherche

### Objectif général

L'objectif général de cette recherche est de **concevoir, implémenter et évaluer de manière comparative deux architectures d'intelligence artificielle conversationnelle** (Agent LLM et Deep Learning + NLP) pour automatiser le service client d'EasyTransfert, et de **formuler des recommandations techniques et organisationnelles** pour leur déploiement en environnement de production.

### Objectifs spécifiques

1. **Constituer et analyser** un corpus de données conversationnelles représentatif des interactions réelles du service client EasyTransfert (3031 conversations historiques couvrant la période juillet-septembre 2024)

2. **Développer un pipeline de prétraitement robuste** adapté aux spécificités linguistiques et techniques du contexte ivoirien, incluant :
   - Gestion du code-switching français-anglais-nouchi
   - Anonymisation RGPD des données sensibles (numéros de téléphone, identifiants de transaction)
   - Normalisation des formats d'identifiants inter-opérateurs
   - Tokenisation et vectorisation adaptées au français d'Afrique de l'Ouest

3. **Concevoir et implémenter deux architectures distinctes** :
   - **Architecture 1** : Agent conversationnel basé sur Llama 3.2 3B Instruct fine-tuné avec LoRA (Low-Rank Adaptation)
   - **Architecture 2** : Système modulaire de Deep Learning intégrant classification d'intentions (BiLSTM + Attention), Named Entity Recognition (BiLSTM-CRF), analyse de sentiment (CamemBERT), et génération de réponses (templates + seq2seq)

4. **Définir et appliquer un protocole d'évaluation rigoureux** combinant :
   - Métriques techniques : BLEU-4, ROUGE-L, perplexité, F1-score, latence
   - Métriques métier : taux de résolution, Net Promoter Score (NPS), coût par requête
   - Analyse qualitative : cohérence, pertinence factuelle, taux d'hallucination

5. **Analyser comparativement les résultats** et identifier les forces et faiblesses de chaque approche en fonction de différents critères : type de requête, complexité linguistique, exigences de latence, contraintes de coût

6. **Formuler des recommandations techniques** pour le déploiement d'une solution hybride optimisant le rapport performance/coût

## Hypothèses de recherche

**Hypothèse principale (H1)** : Un système conversationnel intelligent basé sur l'IA générative peut réduire significativement (> 60%) la charge de travail des agents humains du service client EasyTransfert tout en maintenant un niveau de qualité satisfaisant (taux de résolution > 80%).

**Hypothèses secondaires** :

- **H2** : L'approche Agent LLM offre une meilleure flexibilité et capacité d'adaptation contextuelle que l'approche Deep Learning + NLP classique, au prix d'une latence de réponse plus élevée et d'un risque d'hallucination non nul.

- **H3** : L'approche Deep Learning + NLP présente des performances plus stables et prévisibles grâce à ses modules spécialisés, avec des coûts d'inférence inférieurs et une garantie de zéro hallucination, mais au prix d'une moindre flexibilité face à des requêtes non prévues dans le design initial.

- **H4** : Le pipeline de prétraitement adapté au contexte ivoirien (gestion du code-switching, normalisation des identifiants) améliore significativement les performances des deux architectures comparé à un prétraitement standard.

## Contributions de la recherche

Cette recherche apporte plusieurs contributions aux domaines de l'IA conversationnelle et du service client automatisé :

**Contributions scientifiques** :
1. Une étude comparative rigoureuse de deux paradigmes d'IA conversationnelle (LLM end-to-end vs Deep Learning modulaire) dans un contexte réel de service client fintech
2. Un pipeline de prétraitement adapté aux spécificités linguistiques du français d'Afrique de l'Ouest, incluant la gestion du code-switching et des expressions locales
3. Une méthodologie d'évaluation holistique combinant métriques techniques et métriques métier, adaptable à d'autres contextes d'automatisation du service client

**Contributions pratiques** :
1. Deux architectures implémentées et testées sur données réelles, avec code source et notebooks reproductibles
2. Des recommandations techniques pour le choix et le déploiement de solutions d'IA conversationnelle dans le secteur fintech africain
3. Une analyse coût-bénéfice comparative entre les deux approches, facilitant la prise de décision pour les décideurs techniques et business

**Contribution au contexte local** :
1. Un cas d'usage documenté d'application de l'IA générative dans le secteur fintech ivoirien
2. La constitution d'un corpus annoté de conversations en français d'Afrique de l'Ouest dans le domaine du mobile money, potentiellement réutilisable pour d'autres recherches

## Structure du mémoire

Le présent mémoire est structuré en cinq chapitres principaux :

Le **Chapitre I** présente le cadre contextuel de notre recherche, incluant une présentation détaillée d'EasyTransfert et de son écosystème, ainsi qu'un état de l'art synthétique sur les approches d'IA conversationnelle (LLM vs Deep Learning + NLP) et leurs applications au service client.

Le **Chapitre II** détaille notre méthodologie de recherche, en commençant par la description du corpus de données (3031 conversations), puis en présentant le pipeline complet de prétraitement développé, avec un focus particulier sur les adaptations nécessaires au contexte linguistique ivoirien.

Les **Chapitres III et IV** présentent respectivement les deux architectures développées : l'Agent LLM basé sur Llama 3.2 3B avec fine-tuning LoRA (Chapitre III), et le système modulaire de Deep Learning + NLP (Chapitre IV). Chaque chapitre détaille les choix d'architecture, les techniques d'implémentation et les paramètres de configuration.

Le **Chapitre V** expose le protocole d'évaluation, présente les résultats expérimentaux obtenus, et propose une analyse comparative approfondie des deux approches selon différents axes (performance technique, métriques métier, analyse qualitative).

Enfin, la **Conclusion générale** synthétise les principaux résultats, discute les limitations de l'étude, formule des recommandations pratiques pour EasyTransfert, et identifie des pistes de recherche future.

---

# CHAPITRE I : CADRE CONTEXTUEL ET ÉTAT DE L'ART

## I.1 Présentation de l'entreprise EasyTransfert

### Contexte du mobile money en Côte d'Ivoire

Le marché du mobile money en Côte d'Ivoire a connu une croissance exponentielle au cours de la dernière décennie. Selon les données de l'ARTCI (Autorité de Régulation des Télécommunications de Côte d'Ivoire), le nombre de comptes actifs de mobile money est passé de 3,2 millions en 2013 à 35,5 millions en 2023, représentant un taux de croissance annuel composé (TCAC) de 27% [2]. Le volume de transactions a atteint 45 000 milliards de FCFA en 2023, positionnant la Côte d'Ivoire comme le troisième marché africain de mobile money après le Kenya et la Tanzanie [5].

Cette expansion s'accompagne toutefois d'un défi majeur : la fragmentation du marché entre cinq principaux opérateurs (MTN Mobile Money, Orange Money, Moov Money, Wave, et Trésor Money), chacun ayant développé son propre écosystème fermé. Cette situation crée des frictions pour les utilisateurs souhaitant effectuer des transactions inter-opérateurs.

### KAYBIC AFRICA et EasyTransfert

KAYBIC AFRICA est une entreprise ivoirienne spécialisée dans les solutions de paiement digital et d'agrégation de services financiers, fondée en 2019. L'entreprise a identifié l'interopérabilité comme une opportunité stratégique et a lancé EasyTransfert en 2021, une plateforme de transfert d'argent inter-opérateurs.

EasyTransfert permet aux utilisateurs d'effectuer des transferts entre différents opérateurs de mobile money avec une commission compétitive (2% du montant transféré, plafonné à 500 FCFA). La plateforme traite actuellement une moyenne de 45 000 transactions par jour, avec un volume mensuel moyen de 2,8 milliards de FCFA (septembre 2024).

### Défis du service client

L'analyse des données opérationnelles d'EasyTransfert révèle plusieurs défis au niveau du service client :

1. **Volume de requêtes croissant** : +23% de croissance mensuelle moyenne entre janvier et septembre 2024
2. **Diversité des problématiques** : transactions bloquées (34%), demandes de remboursement (28%), questions tarifaires (19%), problèmes techniques (12%), autres (7%)
3. **Variabilité de la charge** : pics de requêtes entre 10h-12h et 17h-20h, avec un ratio de 3:1 entre heures de pointe et heures creuses
4. **Exigences de qualité élevées** : secteur fintech réglementé nécessitant précision et conformité aux règles métier

Le coût actuel du service client représente environ 18% des coûts opérationnels totaux, avec une équipe de 12 agents dédiés travaillant en rotation. L'automatisation partielle ou totale de ce service pourrait générer des économies significatives tout en améliorant la qualité et la rapidité de traitement.

## I.2 État de l'art : IA conversationnelle

### Large Language Models et architectures d'agents

Les Large Language Models (LLM) représentent une avancée majeure dans le domaine du traitement automatique du langage naturel. Ces modèles, basés sur l'architecture Transformer introduite par Vaswani et al. (2017) [6], sont pré-entraînés sur d'immenses corpus textuels (plusieurs téraoctets) et développent des capacités émergentes de compréhension et de génération de langage.

**Évolution des LLM** :
- GPT-3 (OpenAI, 2020) : 175 milliards de paramètres, démonstration de capacités few-shot learning [7]
- PaLM (Google, 2022) : 540 milliards de paramètres, amélioration des capacités de raisonnement [8]
- Llama 2 (Meta, 2023) : famille de modèles open-source (7B à 70B paramètres), performances comparables aux modèles propriétaires [9]
- Llama 3.2 (Meta, 2024) : nouvelles versions légères (1B et 3B), optimisées pour l'edge computing et le fine-tuning [10]

**Techniques de fine-tuning efficient** :
Face au coût prohibitif du fine-tuning complet des LLM (nécessitant plusieurs centaines de GPU-heures), plusieurs techniques d'adaptation efficiente ont été développées :

- **LoRA (Low-Rank Adaptation)** : introduite par Hu et al. (2021) [11], cette technique consiste à geler les poids du modèle pré-entraîné et à injecter des matrices de faible rang entraînables dans chaque couche Transformer. LoRA réduit le nombre de paramètres entraînables de 99% tout en maintenant des performances comparables au fine-tuning complet.

- **QLoRA** : extension de LoRA utilisant la quantification 4-bit pour réduire encore l'empreinte mémoire, permettant le fine-tuning de modèles 65B sur un seul GPU 48GB [12]

- **Prompt-tuning et Prefix-tuning** : méthodes consistant à apprendre des embeddings continus (soft prompts) plutôt que les poids du modèle [13]

**Agents conversationnels basés sur les LLM** :
L'utilisation des LLM comme agents conversationnels repose sur une approche end-to-end où le modèle génère directement les réponses à partir des requêtes utilisateur. Cette approche présente plusieurs avantages : flexibilité face à des requêtes variées, capacité de raisonnement multi-étapes, et génération de réponses naturelles et contextualisées [14].

Cependant, des défis persistent : hallucinations (génération d'informations factuellement incorrectes), manque de contrôle sur le comportement du modèle, coûts d'inférence élevés, et difficultés d'interprétabilité [15]. Des travaux récents explorent l'utilisation de techniques comme le RAG (Retrieval-Augmented Generation) pour ancrer les réponses sur des sources factuelles [16], ou le fine-tuning avec RLHF (Reinforcement Learning from Human Feedback) pour aligner le comportement des modèles sur les préférences humaines [17].

### Deep Learning pour le NLP

Avant l'émergence des LLM, les systèmes de traitement du langage naturel reposaient principalement sur des architectures modulaires combinant plusieurs composants spécialisés. Cette approche reste pertinente pour de nombreuses applications, notamment lorsque la fiabilité, l'interprétabilité et le contrôle sont critiques.

**Réseaux de neurones récurrents** :
Les RNN (Recurrent Neural Networks) et leurs variantes (LSTM, GRU) ont longtemps été la pierre angulaire du NLP pour les tâches séquentielles [18]. Les LSTM (Long Short-Term Memory), introduits par Hochreiter et Schmidhuber (1997) [19], résolvent le problème du gradient évanescent des RNN classiques grâce à des mécanismes de portes (input, forget, output gates) permettant de mémoriser l'information à long terme.

Les BiLSTM (Bidirectional LSTM) traitent les séquences dans les deux directions (avant et arrière), capturant ainsi le contexte gauche et droit de chaque token [20]. Cette architecture est particulièrement efficace pour des tâches comme le Named Entity Recognition ou le Part-of-Speech Tagging.

**Mécanismes d'attention** :
Le mécanisme d'attention, introduit par Bahdanau et al. (2015) [21], permet au modèle de se concentrer sur des parties spécifiques de l'entrée lors de la génération de chaque élément de sortie. L'attention peut être vue comme un mécanisme de pondération soft apprenant à aligner les représentations d'entrée et de sortie.

L'architecture Transformer (Vaswani et al., 2017) [6] généralise ce principe avec le self-attention multi-têtes, permettant au modèle de capturer des dépendances à longue distance plus efficacement que les RNN. Cependant, pour des datasets de taille modeste, les architectures hybrides combinant LSTM et attention peuvent surpasser les Transformers purs en termes de rapport performance/coût d'entraînement [22].

**Modèles pré-entraînés pour le français** :
Le transfer learning via des modèles pré-entraînés est devenu le paradigme dominant en NLP. Pour le français, plusieurs modèles sont disponibles :

- **CamemBERT** (Martin et al., 2019) : modèle BERT pré-entraîné sur 138GB de données françaises (OSCAR corpus), performances state-of-the-art sur diverses tâches de NLP français [23]
- **FlauBERT** : alternative à CamemBERT avec différentes stratégies de tokenisation [24]
- **mBERT et XLM-RoBERTa** : modèles multilingues incluant le français [25]

Ces modèles peuvent être fine-tunés sur des tâches spécifiques (classification, NER, question-answering) avec des datasets relativement modestes (quelques milliers d'exemples), bénéficiant du transfert de connaissances depuis le pré-entraînement.

### Applications au service client

L'application de l'IA au service client a évolué à travers plusieurs générations de technologies [26] :

**Première génération (années 1990-2000)** : systèmes à base de règles (rule-based chatbots) utilisant des arbres de décision et des templates de réponses. Limitation : rigidité face à la variabilité linguistique.

**Deuxième génération (années 2010)** : systèmes basés sur l'apprentissage automatique classique (SVM, Random Forests) pour la classification d'intentions, combinés à des règles pour la génération de réponses. Amélioration de la robustesse mais nécessité d'ingénierie de features importantes [27].

**Troisième génération (années 2015-2020)** : émergence du deep learning avec des architectures bout-en-bout pour le dialogue (seq2seq, attention mechanisms). Gains significatifs en terme de naturalité des réponses mais défis de données d'entraînement [28].

**Quatrième génération (2020-présent)** : adoption des LLM pré-entraînés (GPT, BERT-based models) permettant des performances élevées avec moins de données annotées. Capacités émergentes de reasoning multi-tours [29].

**Cas d'usage dans le secteur fintech** :
Plusieurs fintechs ont déployé des systèmes d'IA conversationnelle pour leur service client :
- Bank of America (Erica) : assistant virtuel traitant 1,5 milliard de requêtes en 2023 [30]
- Capital One (Eno) : détection proactive de fraudes et assistance pour transactions [31]
- PayPal : chatbot multilingue pour support technique et résolution de litiges

Une étude de Juniper Research (2023) estime que les chatbots bancaires permettront d'économiser 7,3 milliards de dollars d'ici 2027 en coûts de service client [32]. Cependant, le secteur fintech présente des contraintes spécifiques : nécessité de précision factuelle (zéro tolérance aux erreurs sur les montants ou identifiants), conformité réglementaire, et traçabilité des décisions pour l'audit.

---

