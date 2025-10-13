"""
Script pour diviser les données de conversation en ensembles train/validation/test.
"""
import json
import random
from pathlib import Path

def split_conversations(input_file, output_dir, train_ratio=0.80, val_ratio=0.15, test_ratio=0.05, seed=42):
    """
    Divise les conversations en ensembles train, validation et test.
    
    Args:
        input_file: Chemin vers le fichier JSONL des conversations
        output_dir: Répertoire de sortie pour les fichiers divisés
        train_ratio: Proportion pour l'ensemble d'entraînement (défaut: 0.80)
        val_ratio: Proportion pour l'ensemble de validation (défaut: 0.15)
        test_ratio: Proportion pour l'ensemble de test (défaut: 0.05)
        seed: Graine aléatoire pour la reproductibilité
    """
    # Vérifier que les ratios totalisent 1.0
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Les ratios doivent totaliser 1.0"
    
    # Fixer la graine aléatoire
    random.seed(seed)
    
    # Charger toutes les conversations
    conversations = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            conversations.append(json.loads(line))
    
    total = len(conversations)
    print(f"Total de conversations chargées: {total}")
    
    # Mélanger les conversations
    random.shuffle(conversations)
    
    # Calculer les tailles des ensembles
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size
    
    print(f"\nDivision des données:")
    print(f"  Train: {train_size} conversations ({train_size/total*100:.1f}%)")
    print(f"  Validation: {val_size} conversations ({val_size/total*100:.1f}%)")
    print(f"  Test: {test_size} conversations ({test_size/total*100:.1f}%)")
    
    # Diviser les données
    train_data = conversations[:train_size]
    val_data = conversations[train_size:train_size + val_size]
    test_data = conversations[train_size + val_size:]
    
    # Créer le répertoire de sortie s'il n'existe pas
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Sauvegarder les ensembles
    splits = {
        'train': train_data,
        'validation': val_data,
        'test': test_data
    }
    
    for split_name, split_data in splits.items():
        output_file = output_path / f'conversations_{split_name}.jsonl'
        with open(output_file, 'w', encoding='utf-8') as f:
            for conversation in split_data:
                f.write(json.dumps(conversation, ensure_ascii=False) + '\n')
        print(f"\n✓ {split_name.capitalize()}: {len(split_data)} conversations sauvegardées dans {output_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Diviser les conversations en ensembles train/val/test')
    parser.add_argument(
        '--input',
        type=str,
        default='data/conversations/conversation_1000_finetune.jsonl',
        help='Chemin vers le fichier d\'entrée JSONL'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/conversations/splits',
        help='Répertoire de sortie pour les fichiers divisés'
    )
    parser.add_argument('--train-ratio', type=float, default=0.80, help='Ratio pour train')
    parser.add_argument('--val-ratio', type=float, default=0.15, help='Ratio pour validation')
    parser.add_argument('--test-ratio', type=float, default=0.05, help='Ratio pour test')
    parser.add_argument('--seed', type=int, default=42, help='Graine aléatoire')
    
    args = parser.parse_args()
    
    split_conversations(
        args.input,
        args.output,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed
    )
