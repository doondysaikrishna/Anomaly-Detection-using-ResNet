"""
Industrial Visual Anomaly Detection - Evaluation & Benchmarking CLI
Computes Image-level and Pixel-level AUROC, Precision, Recall, F1, and Confusion Matrices.
"""

import argparse
import os
import json
from core.pipeline import IndustrialAnomalyPipeline
from core.dataset.industrial_dataset import get_available_categories


def main():
    parser = argparse.ArgumentParser(description="Evaluate Industrial Visual Anomaly Detection Models")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["cable", "capsule", "circuit_board", "transistor"],
        help="Categories to evaluate",
    )
    parser.add_argument("--dataset_root", type=str, default="test", help="Root dataset directory")
    parser.add_argument("--weights_dir", type=str, default="weights", help="Directory containing trained weights")
    parser.add_argument("--output_file", type=str, default="evaluation_results.json", help="Path to save results JSON")
    
    args = parser.parse_args()
    
    print(f"\n=======================================================")
    print(f"  INDUSTRIAL VISUAL ANOMALY DETECTION - EVALUATION     ")
    print(f"=======================================================\n")
    
    results = {}
    
    for category in args.categories:
        print(f"[*] Evaluating category: '{category}'...")
        try:
            pipeline = IndustrialAnomalyPipeline(
                category=category,
                dataset_root=args.dataset_root,
                weights_dir=args.weights_dir,
            )
            
            if not pipeline.is_ready:
                print(f"[!] No weights found for '{category}'. Training on the fly...")
                pipeline.train(calibrate=True)
                
            metrics = pipeline.calibrate_and_evaluate()
            results[category] = metrics
            
        except Exception as e:
            print(f"[!] Error evaluating category '{category}': {e}")
            results[category] = {"error": str(e)}
            
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"\n==========================================================================================")
    print(f"                               BENCHMARK EVALUATION RESULTS                               ")
    print(f"==========================================================================================")
    print(f"{'Category':<16} | {'Samples':<8} | {'Image AUROC':<12} | {'Pixel AUROC':<12} | {'F1-Score':<10} | {'Precision':<10} | {'Recall':<8}")
    print("-" * 88)
    for cat, data in results.items():
        if "image_auroc" in data:
            pix_auroc = f"{data['pixel_auroc']:.4f}" if data.get("pixel_auroc") is not None else "N/A"
            print(f"{cat:<16} | {data.get('total_samples', 0):<8} | {data.get('image_auroc', 0.0):<12.4f} | {pix_auroc:<12} | {data.get('f1_score', 0.0):<10.4f} | {data.get('precision', 0.0):<10.4f} | {data.get('recall', 0.0):<8.4f}")
        else:
            print(f"{cat:<16} | {'ERROR':<8} | {'ERROR':<12} | {'ERROR':<12} | {'ERROR':<10} | {'ERROR':<10} | {'ERROR':<8}")
            
    print(f"\n[+] Results exported to: {os.path.abspath(args.output_file)}\n")


if __name__ == "__main__":
    main()
