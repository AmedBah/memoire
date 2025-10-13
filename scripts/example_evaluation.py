"""
Exemple d'utilisation du système d'évaluation des architectures.
"""
import random
import time
from compare_architectures import ArchitectureEvaluator, load_test_data, compare_architectures


# Fonctions de chat simulées pour chaque architecture
def architecture_1_chat(query: str) -> str:
    """
    Architecture 1: Agent LLM simple (fine-tuned).
    Simulation avec latence et réponses basiques.
    """
    # Simuler une latence élevée (LLM inference)
    time.sleep(random.uniform(2.5, 3.2))
    
    # Réponse simulée
    responses = [
        "Bonjour, Service client EasyTransfert 🤗. Comment puis-je vous aider aujourd'hui ?",
        "Pour effectuer un transfert, veuillez suivre ces étapes : 1) Connectez-vous à l'application, 2) Sélectionnez le destinataire, 3) Entrez le montant.",
        "Les frais de transfert dépendent du montant et de l'opérateur. Pour plus d'informations, consultez notre grille tarifaire.",
        "Je comprends votre problème. Pouvez-vous me donner plus de détails sur votre transaction ?"
    ]
    return random.choice(responses)


def architecture_2_chat(query: str) -> str:
    """
    Architecture 2: RAG standard.
    Simulation avec latence moyenne et réponses plus précises.
    """
    # Simuler une latence moyenne (retrieval + LLM)
    time.sleep(random.uniform(1.0, 1.5))
    
    # Réponse simulée avec sources
    responses = [
        "Bonjour ! Service client EasyTransfert. Je peux vous aider avec vos questions.\n\nSources: FAQ EasyTransfert",
        "Pour transférer de l'argent:\n1. Connectez-vous à votre compte\n2. Sélectionnez 'Transfert'\n3. Choisissez l'opérateur et le montant\n\nSources: Guide d'utilisation",
        "Les frais varient selon l'opérateur:\n- MTN vers Orange: 2% (min 100 FCFA)\n- Orange vers MTN: 2% (min 100 FCFA)\n\nSources: Grille tarifaire",
        "Je peux vous aider avec ce problème. Voici les étapes recommandées...\n\nSources: Base de connaissances"
    ]
    return random.choice(responses)


def architecture_3_chat(query: str) -> str:
    """
    Architecture 3: RAG-Agentique.
    Simulation avec latence variable (ReAct cycle) et réponses détaillées.
    """
    # Simuler une latence variable selon la complexité
    if "frais" in query.lower() or "tarif" in query.lower():
        time.sleep(random.uniform(2.0, 3.0))  # Utilise l'outil operator_info
    else:
        time.sleep(random.uniform(1.5, 2.2))
    
    # Réponse simulée avec raisonnement
    responses = [
        "Bonjour ! Je vais vous aider avec EasyTransfert.\n\nÉtapes:\n1. Analyse de votre demande\n2. Recherche dans la base de connaissances\n\nRéponse: ...",
        "Pour effectuer ce transfert:\n\nAction 1: Vérification des informations opérateur\nAction 2: Consultation de la grille tarifaire\n\nRésultat: Voici les détails...",
        "J'ai analysé votre demande:\n\nPensée: L'utilisateur demande des informations sur les frais\nAction: Utilisation de l'outil operator_info\nRésultat: Les frais sont...",
        "Réponse complète basée sur:\n- FAQ EasyTransfert\n- Documentation opérateurs\n- Données en temps réel\n\nDétails: ..."
    ]
    return random.choice(responses)


def main():
    """Fonction principale pour l'exemple d'évaluation."""
    print("="*80)
    print("🚀 EXEMPLE D'ÉVALUATION DES ARCHITECTURES")
    print("="*80)
    
    # Charger les données de test
    print("\n📂 Chargement des données de test...")
    test_data = load_test_data('data/conversations/splits/conversations_test.jsonl')
    print(f"✓ {len(test_data)} conversations de test chargées")
    
    # Évaluer chaque architecture sur un sous-ensemble
    max_samples = 10  # Limiter pour l'exemple
    num_runs = 2      # Nombre de runs par requête
    
    architectures = [
        ("Architecture 1 - Fine-tuning Simple", architecture_1_chat),
        ("Architecture 2 - RAG Standard", architecture_2_chat),
        ("Architecture 3 - RAG-Agentique", architecture_3_chat)
    ]
    
    results_files = []
    
    for arch_name, chat_function in architectures:
        evaluator = ArchitectureEvaluator(arch_name)
        
        print(f"\n{'='*80}")
        print(f"🔍 Évaluation de {arch_name}")
        print(f"{'='*80}")
        
        evaluator.evaluate_dataset(
            test_data,
            chat_function,
            num_runs=num_runs,
            max_samples=max_samples
        )
        
        evaluator.print_summary()
        
        # Sauvegarder les résultats
        output_file = f"results/{arch_name.lower().replace(' ', '_').replace('-', '_')}_results.json"
        evaluator.save_results(output_file)
        results_files.append(output_file)
    
    # Comparer toutes les architectures
    print("\n" + "="*80)
    print("📊 COMPARAISON FINALE")
    print("="*80)
    
    compare_architectures(results_files)
    
    print("\n✅ Évaluation terminée !")
    print("\nLes résultats détaillés sont disponibles dans le dossier 'results/'")


if __name__ == "__main__":
    # Créer le dossier results s'il n'existe pas
    from pathlib import Path
    Path("results").mkdir(exist_ok=True)
    
    main()
