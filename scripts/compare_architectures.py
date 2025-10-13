"""
Script pour comparer les performances des différentes architectures sur le test set.
"""
import json
import time
from pathlib import Path
from typing import List, Dict, Callable
import numpy as np
from evaluation_metrics import evaluate_response, calculate_response_relevance


class ArchitectureEvaluator:
    """Classe pour évaluer une architecture conversationnelle."""
    
    def __init__(self, architecture_name: str):
        """
        Initialise l'évaluateur.
        
        Args:
            architecture_name: Nom de l'architecture à évaluer
        """
        self.architecture_name = architecture_name
        self.results = []
    
    def evaluate_conversation(
        self,
        conversation: Dict,
        chat_function: Callable,
        num_runs: int = 3
    ) -> Dict:
        """
        Évalue une conversation avec l'architecture.
        
        Args:
            conversation: Dictionnaire contenant les messages
            chat_function: Fonction de chat de l'architecture
            num_runs: Nombre d'exécutions pour mesurer la latence
        
        Returns:
            Dictionnaire avec les métriques de performance
        """
        messages = conversation['messages']
        
        # Séparer user et assistant messages
        user_messages = [m for m in messages if m['role'] == 'user']
        assistant_messages = [m for m in messages if m['role'] == 'assistant']
        
        if len(user_messages) == 0 or len(assistant_messages) == 0:
            return None
        
        # Prendre le dernier échange user-assistant
        last_user_msg = user_messages[-1]['content']
        reference_response = assistant_messages[-1]['content']
        
        # Mesurer la latence (moyenne sur plusieurs runs)
        latencies = []
        generated_response = None
        
        for _ in range(num_runs):
            start_time = time.time()
            try:
                generated_response = chat_function(last_user_msg)
            except Exception as e:
                print(f"Erreur lors de la génération: {e}")
                generated_response = ""
            latency = (time.time() - start_time) * 1000  # en ms
            latencies.append(latency)
        
        avg_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        
        # Calculer les métriques de qualité
        quality_metrics = evaluate_response(reference_response, generated_response)
        
        # Calculer la pertinence
        relevance = calculate_response_relevance(generated_response, last_user_msg)
        
        result = {
            'query': last_user_msg,
            'reference': reference_response,
            'generated': generated_response,
            'latency_ms': avg_latency,
            'latency_std_ms': std_latency,
            'relevance': relevance,
            **quality_metrics
        }
        
        self.results.append(result)
        return result
    
    def evaluate_dataset(
        self,
        test_data: List[Dict],
        chat_function: Callable,
        num_runs: int = 3,
        max_samples: int = None
    ):
        """
        Évalue l'architecture sur un dataset complet.
        
        Args:
            test_data: Liste de conversations de test
            chat_function: Fonction de chat de l'architecture
            num_runs: Nombre d'exécutions par requête
            max_samples: Nombre maximum d'échantillons à évaluer (None = tous)
        """
        print(f"\n{'='*80}")
        print(f"🔍 Évaluation de {self.architecture_name}")
        print(f"{'='*80}\n")
        
        samples = test_data[:max_samples] if max_samples else test_data
        
        for idx, conversation in enumerate(samples, 1):
            print(f"Conversation {idx}/{len(samples)}...", end=' ')
            
            result = self.evaluate_conversation(conversation, chat_function, num_runs)
            
            if result:
                print(f"✓ (Latence: {result['latency_ms']:.0f}ms, BLEU-4: {result['bleu-4']:.3f})")
            else:
                print("⚠️  Ignorée (pas de messages)")
    
    def compute_summary_statistics(self) -> Dict:
        """
        Calcule les statistiques résumées de l'évaluation.
        
        Returns:
            Dictionnaire avec les statistiques moyennes
        """
        if not self.results:
            return {}
        
        # Calculer les moyennes pour toutes les métriques
        metrics = {}
        
        for key in self.results[0].keys():
            if key not in ['query', 'reference', 'generated']:
                values = [r[key] for r in self.results if key in r]
                if values:
                    metrics[f"{key}_mean"] = np.mean(values)
                    metrics[f"{key}_std"] = np.std(values)
                    metrics[f"{key}_min"] = np.min(values)
                    metrics[f"{key}_max"] = np.max(values)
        
        return metrics
    
    def print_summary(self):
        """Affiche un résumé des résultats."""
        if not self.results:
            print("Aucun résultat disponible.")
            return
        
        stats = self.compute_summary_statistics()
        
        print(f"\n{'='*80}")
        print(f"📊 Résumé - {self.architecture_name}")
        print(f"{'='*80}\n")
        
        print(f"Nombre de conversations évaluées: {len(self.results)}\n")
        
        # Métriques de performance
        print("🚀 Performance:")
        print(f"  Latence moyenne: {stats.get('latency_ms_mean', 0):.2f} ± {stats.get('latency_ms_std', 0):.2f} ms")
        print(f"  Latence min/max: {stats.get('latency_ms_min', 0):.2f} / {stats.get('latency_ms_max', 0):.2f} ms")
        
        # Calculer le throughput
        if stats.get('latency_ms_mean', 0) > 0:
            throughput = 1000 / stats['latency_ms_mean']  # req/s
            print(f"  Throughput: {throughput:.2f} req/s")
        
        # Métriques de qualité
        print("\n📈 Qualité (scores 0-1):")
        print(f"  BLEU-1: {stats.get('bleu-1_mean', 0):.4f} ± {stats.get('bleu-1_std', 0):.4f}")
        print(f"  BLEU-2: {stats.get('bleu-2_mean', 0):.4f} ± {stats.get('bleu-2_std', 0):.4f}")
        print(f"  BLEU-3: {stats.get('bleu-3_mean', 0):.4f} ± {stats.get('bleu-3_std', 0):.4f}")
        print(f"  BLEU-4: {stats.get('bleu-4_mean', 0):.4f} ± {stats.get('bleu-4_std', 0):.4f}")
        print(f"  ROUGE-1: {stats.get('rouge-1_mean', 0):.4f} ± {stats.get('rouge-1_std', 0):.4f}")
        print(f"  ROUGE-2: {stats.get('rouge-2_mean', 0):.4f} ± {stats.get('rouge-2_std', 0):.4f}")
        print(f"  ROUGE-L: {stats.get('rouge-l_mean', 0):.4f} ± {stats.get('rouge-l_std', 0):.4f}")
        print(f"  Similarité sémantique: {stats.get('semantic_similarity_mean', 0):.4f} ± {stats.get('semantic_similarity_std', 0):.4f}")
        
        # Pertinence
        print("\n🎯 Pertinence:")
        print(f"  Score moyen: {stats.get('relevance_mean', 0):.4f} ± {stats.get('relevance_std', 0):.4f}")
        
        print(f"\n{'='*80}\n")
    
    def save_results(self, output_file: str):
        """
        Sauvegarde les résultats dans un fichier JSON.
        
        Args:
            output_file: Chemin du fichier de sortie
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        summary = {
            'architecture': self.architecture_name,
            'num_samples': len(self.results),
            'statistics': self.compute_summary_statistics(),
            'detailed_results': self.results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Résultats sauvegardés dans: {output_file}")


def load_test_data(test_file: str) -> List[Dict]:
    """
    Charge les données de test.
    
    Args:
        test_file: Chemin vers le fichier JSONL de test
    
    Returns:
        Liste de conversations
    """
    conversations = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            conversations.append(json.loads(line))
    return conversations


def compare_architectures(results_files: List[str]):
    """
    Compare les résultats de plusieurs architectures.
    
    Args:
        results_files: Liste des fichiers de résultats JSON
    """
    print("\n" + "="*80)
    print("📊 COMPARAISON DES ARCHITECTURES")
    print("="*80 + "\n")
    
    architectures = []
    
    for file in results_files:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            architectures.append(data)
    
    if not architectures:
        print("Aucune architecture à comparer.")
        return
    
    # Tableau comparatif
    print(f"{'Métrique':<30} | " + " | ".join([f"{a['architecture'][:20]:>20}" for a in architectures]))
    print("-" * 80)
    
    # Métriques de performance
    print("\n🚀 PERFORMANCE:")
    metric_keys = [
        ('latency_ms_mean', 'Latence moyenne (ms)', '{:.2f}'),
        ('throughput', 'Throughput (req/s)', '{:.2f}')
    ]
    
    for key, label, fmt in metric_keys:
        values = []
        for arch in architectures:
            stats = arch.get('statistics', {})
            if key == 'throughput':
                latency = stats.get('latency_ms_mean', 0)
                value = 1000 / latency if latency > 0 else 0
            else:
                value = stats.get(key, 0)
            values.append(value)
        
        # Trouver le meilleur (plus bas pour latence, plus haut pour throughput)
        if 'latency' in key:
            best_idx = np.argmin(values) if any(v > 0 for v in values) else -1
        else:
            best_idx = np.argmax(values)
        
        row = f"{label:<30} | "
        for i, v in enumerate(values):
            marker = " 🏆" if i == best_idx and v > 0 else "   "
            row += f"{fmt.format(v):>20}{marker} | "
        print(row)
    
    # Métriques de qualité
    print("\n📈 QUALITÉ:")
    quality_metrics = [
        ('bleu-1_mean', 'BLEU-1'),
        ('bleu-4_mean', 'BLEU-4'),
        ('rouge-1_mean', 'ROUGE-1'),
        ('rouge-l_mean', 'ROUGE-L'),
        ('semantic_similarity_mean', 'Similarité sémantique'),
        ('relevance_mean', 'Pertinence')
    ]
    
    for key, label in quality_metrics:
        values = []
        for arch in architectures:
            value = arch.get('statistics', {}).get(key, 0)
            values.append(value)
        
        best_idx = np.argmax(values)
        
        row = f"{label:<30} | "
        for i, v in enumerate(values):
            marker = " 🏆" if i == best_idx else "   "
            row += f"{v:>20.4f}{marker} | "
        print(row)
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Évaluer et comparer les architectures conversationnelles'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commandes disponibles')
    
    # Commande pour évaluer une architecture
    eval_parser = subparsers.add_parser('evaluate', help='Évaluer une architecture')
    eval_parser.add_argument('--name', type=str, required=True, help='Nom de l\'architecture')
    eval_parser.add_argument('--test-file', type=str, required=True, help='Fichier de test JSONL')
    eval_parser.add_argument('--output', type=str, required=True, help='Fichier de sortie JSON')
    eval_parser.add_argument('--max-samples', type=int, default=None, help='Nombre max d\'échantillons')
    eval_parser.add_argument('--num-runs', type=int, default=3, help='Nombre de runs par requête')
    
    # Commande pour comparer les architectures
    compare_parser = subparsers.add_parser('compare', help='Comparer plusieurs architectures')
    compare_parser.add_argument('results', nargs='+', help='Fichiers de résultats JSON')
    
    args = parser.parse_args()
    
    if args.command == 'evaluate':
        print("Pour évaluer une architecture, vous devez fournir une fonction de chat.")
        print("Utilisez ce script comme module dans un notebook ou script Python.")
        print("\nExemple d'utilisation:")
        print("```python")
        print("from compare_architectures import ArchitectureEvaluator, load_test_data")
        print("")
        print("# Charger les données de test")
        print("test_data = load_test_data('data/conversations/splits/conversations_test.jsonl')")
        print("")
        print("# Créer un évaluateur")
        print("evaluator = ArchitectureEvaluator('Architecture 1')")
        print("")
        print("# Définir votre fonction de chat")
        print("def my_chat_function(query):")
        print("    # Votre code ici")
        print("    return response")
        print("")
        print("# Évaluer")
        print("evaluator.evaluate_dataset(test_data, my_chat_function, max_samples=10)")
        print("evaluator.print_summary()")
        print("evaluator.save_results('results/arch1_results.json')")
        print("```")
    
    elif args.command == 'compare':
        compare_architectures(args.results)
    
    else:
        parser.print_help()
