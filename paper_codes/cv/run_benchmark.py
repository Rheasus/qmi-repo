"""
Batch Benchmark Execution Script
=================================

Run complete benchmark across multiple configurations.

Usage:
    python run_benchmark.py --datasets cifar10 cifar100 --models SimpleCNN ResNet18 
                            --optimizers Adam SGD --seeds 42 1337 --epochs 50

Author: Gorkem Yilmaz
Institution: University of Sussex
Contact: gy74@sussex.ac.uk
"""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path
from datetime import datetime
import pandas as pd


def run_single_experiment(config, data_dir, results_dir):
    """Run a single experiment configuration."""
    
    cmd = [
        sys.executable, "trainer.py",
        "--dataset", config['dataset'],
        "--model", config['model'],
        "--optimizer", config['optimizer'],
        "--seed", str(config['seed']),
        "--epochs", str(config['epochs']),
        "--lr", str(config['lr']),
        "--batch_size", str(config['batch_size']),
        "--data_dir", data_dir,
        "--results_dir", results_dir
    ]
    
    print(f"\n{'='*80}")
    print(f"Running: {config['dataset']} | {config['model']} | {config['optimizer']} | Seed {config['seed']}")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Run complete CV benchmark'
    )
    
    parser.add_argument('--datasets', nargs='+', required=True,
                        choices=['cifar10', 'cifar100', 'fashion_mnist'],
                        help='Datasets to benchmark')
    parser.add_argument('--models', nargs='+', required=True,
                        choices=['SimpleCNN', 'ResNet18'],
                        help='Models to benchmark')
    parser.add_argument('--optimizers', nargs='+', required=True,
                        choices=['SGD', 'Adam', 'AdamW', 'SPSA', 'QNG'],
                        help='Optimizers to benchmark')
    parser.add_argument('--seeds', nargs='+', type=int, required=True,
                        help='Random seeds')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--data_dir', type=str, default='./datasets',
                        help='Directory for datasets')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Directory for results')
    
    args = parser.parse_args()
    
    # Generate all configurations
    configs = []
    for dataset in args.datasets:
        for model in args.models:
            for optimizer in args.optimizers:
                for seed in args.seeds:
                    configs.append({
                        'dataset': dataset,
                        'model': model,
                        'optimizer': optimizer,
                        'seed': seed,
                        'epochs': args.epochs,
                        'lr': args.lr,
                        'batch_size': args.batch_size
                    })
    
    print(f"\n{'='*80}")
    print(f"BENCHMARK CONFIGURATION")
    print(f"{'='*80}")
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Optimizers: {', '.join(args.optimizers)}")
    print(f"Seeds: {', '.join(map(str, args.seeds))}")
    print(f"Epochs: {args.epochs}")
    print(f"Total experiments: {len(configs)}")
    print(f"{'='*80}\n")
    
    # Confirm before starting
    response = input("Start benchmark? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Benchmark cancelled.")
        return
    
    # Run experiments
    start_time = datetime.now()
    results = []
    
    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] Starting experiment...")
        success = run_single_experiment(config, args.data_dir, args.results_dir)
        
        results.append({
            **config,
            'success': success
        })
    
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    
    # Summary
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    print(f"\n{'='*80}")
    print(f"BENCHMARK COMPLETE")
    print(f"{'='*80}")
    print(f"Total experiments: {len(results)}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"⏱️  Total time: {elapsed/60:.1f} minutes")
    print(f"{'='*80}\n")
    
    # Save summary
    summary_path = Path(args.results_dir) / f"benchmark_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'configs': results,
            'summary': {
                'total': len(results),
                'successful': successful,
                'failed': failed,
                'elapsed_seconds': elapsed
            }
        }, f, indent=2)
    
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
