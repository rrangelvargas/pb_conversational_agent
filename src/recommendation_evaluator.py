"""
Recommendation Evaluator
Evaluates the recommendation agent and tests different category/keyword/moral weight combinations
using a structured ground truth JSON file.
It includes comprehensive evaluation metrics (NDCG@5, F1@5), as well as graphs
and heatmaps for visualisation.
"""

import json
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import List, Dict
from conversational_agent import ProjectRecommender
from constants import SYNTHETIC_DATA_PATH, POLAND_DATA_PATH, WORLDWIDE_DATA_PATH

class RecommendationEvaluator:
    """
    Evaluates the recommendation agent using a structured ground truth JSON file.
    This version runs tests ONLY on a curated pool of projects defined in the ground truth.
    """

    def __init__(self, ground_truth_path="../data/ground_truth.json"):
        """
        Initializes the evaluator by loading the ground truth, all datasets,
        and the conversational agent.
        """
        self.agent = ProjectRecommender(dataset_type="synthetic")
        self.ground_truth = self._load_json(ground_truth_path)
        self.full_datasets = {
            "synthetic": pd.read_csv(SYNTHETIC_DATA_PATH),
            "poland": pd.read_csv(POLAND_DATA_PATH),
            "worldwide": pd.read_csv(WORLDWIDE_DATA_PATH),
        }
        print("Evaluator initialized successfully with the new ground truth structure.")

    def _load_json(self, file_path):
        """Loads and parses the ground truth JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _get_relevance_score(self, project_id, matches):
        """Assigns a relevance score based on the ground truth lists."""
        if project_id in matches.get("perfect", []):
            return 4
        if project_id in matches.get("good", []):
            return 3
        if project_id in matches.get("poor", []):
            return 1
        return 2

    def _calculate_ndcg_at_5(self, recommended_ids, matches):
        """Calculates NDCG@5 using graded relevance scores."""
        dcg = 0
        for i, proj_id in enumerate(recommended_ids[:5]):
            relevance = self._get_relevance_score(proj_id, matches)
            dcg += relevance / math.log2(i + 2)

        ground_truth_projects = matches.get("perfect", []) + matches.get("good", [])
        ideal_ranking = sorted(
            ground_truth_projects,
            key=lambda pid: self._get_relevance_score(pid, matches),
            reverse=True
        )
        
        idcg = 0
        for i, proj_id in enumerate(ideal_ranking[:5]):
            relevance = self._get_relevance_score(proj_id, matches)
            idcg += relevance / math.log2(i + 2)
            
        return dcg / idcg if idcg > 0 else 0.0

    def _calculate_top_k_f1_at_5(self, recommended_ids, matches):
        """Calculates F1@5, considering 'perfect' and 'good' projects as relevant."""
        top_5_recs = set(recommended_ids[:5])
        relevant_set = set(matches.get("perfect", []) + matches.get("good", []))

        if not relevant_set:
            return 0.0, 0.0, 0.0

        true_positives = len(top_5_recs.intersection(relevant_set))
        if true_positives == 0:
            return 0.0, 0.0, 0.0

        precision = true_positives / 5
        recall = true_positives / len(relevant_set)
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return precision, recall, f1_score

    def run_evaluation(self):
        """Runs the full evaluation loop, structured by dataset."""
        test_cases = self.ground_truth["test_cases"]

        for dataset_name, project_ids_to_use in self.ground_truth["projects"].items():
            print("\n" + "#" * 80)
            print(f"## EVALUATING DATASET: {dataset_name.upper()}")
            print("#" * 80)

            full_dataset = self.full_datasets[dataset_name]
            curated_pool_df = full_dataset[full_dataset["project_id"].isin(project_ids_to_use)].copy()
            print(f"  - Using a curated pool of {len(curated_pool_df)} projects for this evaluation.")

            all_ndcg_scores = []
            all_f1_scores = []

            # Set the agent's internal state for this dataset's evaluation run
            self.agent.projects_df = curated_pool_df
            self.agent.keyword_weight = 7
            self.agent.moral_weight = 3

            for prompt, matches_by_dataset in test_cases.items():
                matches = matches_by_dataset.get(dataset_name)

                print(f"\n--- Test Case: \"{prompt[:50]}...\" ---")
                
                if not matches or not (matches.get("perfect") or matches.get("good")):
                    print("  No 'perfect' or 'good' matches defined for this case. Skipping.")
                    continue

                # Use the correct public method from the agent
                results = self.agent.generate_recommendations(user_input=prompt, top_n=5)
                recommendations_df = results['recommendations']
                
                recommended_project_ids = [rec['project_id'] for rec in recommendations_df]
                
                ndcg_score = self._calculate_ndcg_at_5(recommended_project_ids, matches)
                precision, recall, f1_score = self._calculate_top_k_f1_at_5(recommended_project_ids, matches)
                
                all_ndcg_scores.append(ndcg_score)
                all_f1_scores.append(f1_score)

                print(f"  - Top 5 Recommended Project IDs: {recommended_project_ids}")
                print(f"  - Metrics for this case:")
                print(f"    - NDCG@5:   {ndcg_score:.4f}")
                print(f"    - F1@5:     {f1_score:.4f} (Precision: {precision:.2f}, Recall: {recall:.2f})")

            avg_ndcg = sum(all_ndcg_scores) / len(all_ndcg_scores) if all_ndcg_scores else 0.0
            avg_f1 = sum(all_f1_scores) / len(all_f1_scores) if all_f1_scores else 0.0
            
            print("\n" + "=" * 80)
            print(f"AVERAGE SCORES FOR '{dataset_name.upper()}' DATASET:")
            print(f"  - Average NDCG@5: {avg_ndcg:.4f}")
            print(f"  - Average F1@5:   {avg_f1:.4f}")
            print("=" * 80)

    def save_results_to_csv(self, all_results: Dict, test_cases: List[str]):
        """Save evaluation results to CSV files."""
        try:
            # Create results directory
            os.makedirs("../results/evaluation", exist_ok=True)
            
            print("Saving evaluation results to CSV...")
            
            # 1. Summary results CSV
            summary_data = []
            for dataset_name, results in all_results.items():
                summary_data.append({
                    'dataset': dataset_name,
                    'ndcg5_average': results['ndcg5']['average'],
                    'ndcg5_max': results['ndcg5']['max'],
                    'ndcg5_min': results['ndcg5']['min'],
                    'f1_5_average': results['f1_5']['average'],
                    'f1_5_max': results['f1_5']['max'],
                    'f1_5_min': results['f1_5']['min'],
                    'total_test_cases': results['total_test_cases']
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_csv_path = "../results/evaluation/evaluation_summary.csv"
            summary_df.to_csv(summary_csv_path, index=False)
            print(f"[OK] Summary results saved to: {summary_csv_path}")
            
            # 2. Detailed results CSV (individual test case results)
            detailed_data = []
            for dataset_name, results in all_results.items():
                individual_ndcg = results['ndcg5']['individual_scores']
                individual_f1 = results['f1_5']['individual_scores']
                
                for i, test_case in enumerate(test_cases):
                    row_data = {
                        'dataset': dataset_name,
                        'test_case': test_case,
                        'ndcg5_score': individual_ndcg[i] if i < len(individual_ndcg) else 0.0,
                        'f1_5_score': individual_f1[i] if i < len(individual_f1) else 0.0
                    }
                    detailed_data.append(row_data)
            
            detailed_df = pd.DataFrame(detailed_data)
            detailed_csv_path = "../results/evaluation/evaluation_detailed.csv"
            detailed_df.to_csv(detailed_csv_path, index=False)
            print(f"[OK] Detailed results saved to: {detailed_csv_path}")
            
        except Exception as e:
            print(f"[ERROR] Error saving CSV results: {e}")

    def create_evaluation_graphs(self, all_results: Dict, test_cases: List[str]):
        """Create evaluation graphs for all datasets."""
        try:
            # Create results directory
            os.makedirs("../results/evaluation", exist_ok=True)
            
            print("Generating evaluation graphs...")
            print("   Saving graphs to: results/evaluation/")
            
            # Test if we can save files
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            
            # Set style
            plt.style.use('default')
            sns.set_palette("husl")
            
            datasets = list(all_results.keys())
            
            # 1. NDCG@5 Comparison
            fig, ax = plt.subplots(figsize=(12, 8))
            
            avg_scores = [all_results[dataset]['ndcg5']['average'] for dataset in datasets]
            max_scores = [all_results[dataset]['ndcg5']['max'] for dataset in datasets]
            min_scores = [all_results[dataset]['ndcg5']['min'] for dataset in datasets]
            
            x = np.arange(len(datasets))
            width = 0.25
            
            bars_avg = ax.bar(x - width, avg_scores, width, label='Average NDCG@5', alpha=0.8)
            bars_max = ax.bar(x, max_scores, width, label='Max NDCG@5', alpha=0.8)
            bars_min = ax.bar(x + width, min_scores, width, label='Min NDCG@5', alpha=0.8)
            
            ax.set_xlabel('Dataset', fontsize=12)
            ax.set_ylabel('NDCG@5 Score', fontsize=12)
            ax.set_title('NDCG@5 Performance Comparison Across Datasets', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([d.capitalize() for d in datasets])
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bars, scores in [(bars_avg, avg_scores), (bars_max, max_scores), (bars_min, min_scores)]:
                for bar, score in zip(bars, scores):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, height + 0.01, 
                           f'{score:.3f}', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            plt.savefig('../results/evaluation/ndcg5_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. F1@5 Comparison
            fig, ax = plt.subplots(figsize=(12, 8))
            
            avg_scores = [all_results[dataset]['f1_5']['average'] for dataset in datasets]
            max_scores = [all_results[dataset]['f1_5']['max'] for dataset in datasets]
            min_scores = [all_results[dataset]['f1_5']['min'] for dataset in datasets]
            
            bars_avg = ax.bar(x - width, avg_scores, width, label='Average F1@5', alpha=0.8)
            bars_max = ax.bar(x, max_scores, width, label='Max F1@5', alpha=0.8)
            bars_min = ax.bar(x + width, min_scores, width, label='Min F1@5', alpha=0.8)
            
            ax.set_xlabel('Dataset', fontsize=12)
            ax.set_ylabel('F1@5 Score', fontsize=12)
            ax.set_title('F1@5 Performance Comparison Across Datasets', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([d.capitalize() for d in datasets])
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bars, scores in [(bars_avg, avg_scores), (bars_max, max_scores), (bars_min, min_scores)]:
                for bar, score in zip(bars, scores):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, height + 0.01, 
                           f'{score:.3f}', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            plt.savefig('../results/evaluation/f1_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. Metrics Heatmap
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create metrics matrix
            metrics_data = []
            for dataset in datasets:
                row_data = [
                    all_results[dataset]['ndcg5']['average'],
                    all_results[dataset]['f1_5']['average']
                ]
                metrics_data.append(row_data)
            
            metrics_matrix = np.array(metrics_data)
            
            sns.heatmap(metrics_matrix, 
                        xticklabels=['NDCG@5', 'F1@5'],
                        yticklabels=[d.capitalize() for d in datasets],
                        annot=True, 
                        fmt='.3f', 
                        cmap='YlOrRd',
                        ax=ax)
            
            ax.set_title('Evaluation Metrics Heatmap', fontsize=14, fontweight='bold')
            ax.set_xlabel('Metrics', fontsize=12)
            ax.set_ylabel('Datasets', fontsize=12)
            
            plt.tight_layout()
            plt.savefig('../results/evaluation/metrics_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("[OK] All evaluation graphs saved to results/evaluation/")
            print("   - ndcg5_comparison.png - NDCG@5 performance comparison")
            print("   - f1_comparison.png - F1@5 performance comparison")
            print("   - metrics_heatmap.png - All metrics heatmap")
            
        except Exception as e:
            print(f"[ERROR] Error creating graphs: {e}")

    def optimize_weights(self, dataset_type: str = "synthetic") -> Dict:
        """
        Optimize keyword and moral weights for a specific dataset.
        
        Args:
            dataset_type: Dataset to optimize ('synthetic', 'poland', 'worldwide')
            
        Returns:
            Dictionary with optimization results and best weights
        """
        print(f"WEIGHT OPTIMIZATION FOR {dataset_type.upper()} DATASET")
        print("=" * 70)
        print("Testing different keyword/moral weight combinations...")
        print("=" * 70)
        
        # Define weight combinations to test
        # Test different weight combinations for three-component system
        # Format: (category_weight, keyword_weight, moral_weight)
        weight_combinations = [
            (8.0, 1.0, 1.0),   # Heavy category bias
            (6.0, 2.0, 2.0),   # Strong category bias
            (5.0, 3.0, 2.0),   # Default weights
            (4.0, 4.0, 2.0),   # Balanced category/keyword
            (3.0, 5.0, 2.0),   # Moderate keyword bias
            (2.0, 6.0, 2.0),   # Strong keyword bias
            (2.0, 4.0, 4.0),   # Balanced keyword/moral
            (1.0, 3.0, 6.0),   # Strong moral bias
            (1.0, 1.0, 8.0),   # Heavy moral bias
        ]
        
        # Results storage
        optimization_results = []
        test_cases = list(self.ground_truth["test_cases"].keys())
        
        for i, (category_weight, keyword_weight, moral_weight) in enumerate(weight_combinations):
            print(f"\nTesting weights {i+1}/{len(weight_combinations)}: Category={category_weight}, Keyword={keyword_weight}, Moral={moral_weight}")
            
            try:
                # Initialize recommender with specific weights
                recommender = ProjectRecommender(
                    dataset_type=dataset_type,
                    category_weight=category_weight,
                    keyword_weight=keyword_weight,
                    moral_weight=moral_weight
                )
                
                # Get dataset-specific project IDs
                project_ids_to_use = self.ground_truth["projects"].get(dataset_type, [])
                if not project_ids_to_use:
                    print(f"  No projects defined for {dataset_type} dataset. Skipping.")
                    continue
                
                # Filter dataset to ground truth projects
                full_dataset = self.full_datasets[dataset_type]
                curated_pool_df = full_dataset[full_dataset["project_id"].isin(project_ids_to_use)].copy()
                recommender.projects_df = curated_pool_df
                
                # Run evaluation for this weight combination
                all_ndcg_scores = []
                all_f1_scores = []
                
                for prompt, matches_by_dataset in self.ground_truth["test_cases"].items():
                    matches = matches_by_dataset.get(dataset_type)
                    
                    if not matches or not (matches.get("perfect") or matches.get("good")):
                        continue
                    
                    # Generate recommendations
                    results = recommender.generate_recommendations(user_input=prompt, top_n=5)
                    recommendations_df = results['recommendations']
                    recommended_project_ids = [rec['project_id'] for rec in recommendations_df]
                    
                    # Calculate metrics
                    ndcg_score = self._calculate_ndcg_at_5(recommended_project_ids, matches)
                    precision, recall, f1_score = self._calculate_top_k_f1_at_5(recommended_project_ids, matches)
                    
                    all_ndcg_scores.append(ndcg_score)
                    all_f1_scores.append(f1_score)
                
                # Calculate average scores
                avg_ndcg = sum(all_ndcg_scores) / len(all_ndcg_scores) if all_ndcg_scores else 0.0
                avg_f1 = sum(all_f1_scores) / len(all_f1_scores) if all_f1_scores else 0.0
                
                # Store results
                optimization_results.append({
                    'category_weight': category_weight,
                    'keyword_weight': keyword_weight,
                    'moral_weight': moral_weight,
                    'ndcg5_avg': avg_ndcg,
                    'f1_5_avg': avg_f1,
                    'combined_score': avg_ndcg + avg_f1
                })
                
                print(f"  NDCG@5: {avg_ndcg:.3f}")
                print(f"  F1@5: {avg_f1:.3f}")
                print(f"  Combined: {avg_ndcg + avg_f1:.3f}")
                
            except Exception as e:
                print(f"  [ERROR] Error with weights {category_weight}/{keyword_weight}/{moral_weight}: {e}")
                optimization_results.append({
                    'category_weight': category_weight,
                    'keyword_weight': keyword_weight,
                    'moral_weight': moral_weight,
                    'ndcg5_avg': 0.0,
                    'f1_5_avg': 0.0,
                    'combined_score': 0.0
                })
        
        # Find best weights
        df_results = pd.DataFrame(optimization_results)
        
        # Best by NDCG@5
        best_ndcg_idx = df_results['ndcg5_avg'].idxmax()
        best_ndcg = df_results.iloc[best_ndcg_idx]
        
        # Best by F1@5
        best_f1_idx = df_results['f1_5_avg'].idxmax()
        best_f1 = df_results.iloc[best_f1_idx]
        
        # Best by combined score
        best_combined_idx = df_results['combined_score'].idxmax()
        best_combined = df_results.iloc[best_combined_idx]
        
        # Print optimization summary
        print(f"\n{'='*70}")
        print("WEIGHT OPTIMIZATION RESULTS")
        print(f"{'='*70}")
        
        print(f"\nBEST BY NDCG@5:")
        print(f"   Weights: Category={best_ndcg['category_weight']}, Keyword={best_ndcg['keyword_weight']}, Moral={best_ndcg['moral_weight']}")
        print(f"   NDCG@5: {best_ndcg['ndcg5_avg']:.3f}")
        print(f"   F1@5: {best_ndcg['f1_5_avg']:.3f}")
        
        print(f"\nBEST BY F1@5:")
        print(f"   Weights: Category={best_f1['category_weight']}, Keyword={best_f1['keyword_weight']}, Moral={best_f1['moral_weight']}")
        print(f"   NDCG@5: {best_f1['ndcg5_avg']:.3f}")
        print(f"   F1@5: {best_f1['f1_5_avg']:.3f}")
        
        print(f"\nBEST BY COMBINED SCORE (NDCG@5 + F1@5):")
        print(f"   Weights: Category={best_combined['category_weight']}, Keyword={best_combined['keyword_weight']}, Moral={best_combined['moral_weight']}")
        print(f"   NDCG@5: {best_combined['ndcg5_avg']:.3f}")
        print(f"   F1@5: {best_combined['f1_5_avg']:.3f}")
        print(f"   Combined: {best_combined['combined_score']:.3f}")
        
        # Save results to CSV
        self.save_optimization_results_to_csv(df_results, dataset_type)
        
        return {
            'optimization_results': optimization_results,
            'best_ndcg': best_ndcg.to_dict(),
            'best_f1': best_f1.to_dict(),
            'best_combined': best_combined.to_dict(),
            'dataframe': df_results
        }

    def save_optimization_results_to_csv(self, df_results: pd.DataFrame, dataset_type: str):
        """Save weight optimization results to CSV file."""
        try:
            # Create results directory
            os.makedirs("../results/weight_optimization", exist_ok=True)
            
            # Add additional calculated columns
            df_export = df_results.copy()
            df_export['total_weight'] = df_export['category_weight'] + df_export['keyword_weight'] + df_export['moral_weight']
            df_export['dataset'] = dataset_type
            
            # Reorder columns for better readability
            column_order = [
                'dataset', 'category_weight', 'keyword_weight', 'moral_weight', 'total_weight',
                'ndcg5_avg', 'f1_5_avg', 'combined_score'
            ]
            
            df_export = df_export[column_order]
            
            # Save to CSV
            csv_path = f"../results/weight_optimization/weight_optimization_results_{dataset_type}.csv"
            df_export.to_csv(csv_path, index=False)
            
            print(f"[OK] Weight optimization results saved to: {csv_path}")
            
        except Exception as e:
            print(f"[ERROR] Error saving CSV results: {e}")

    def run_comprehensive_evaluation(self):
        """Run comprehensive evaluation with graphs, CSV saving, and weight optimization."""
        print("COMPREHENSIVE RECOMMENDATION EVALUATION")
        print("=" * 70)
        print("Testing conversational agent across all datasets with:")
        print("  • Standard evaluation metrics (NDCG@5, F1@5)")
        print("  • Graph generation and visualization")
        print("  • CSV results saving")
        print("  • Weight optimization for each dataset")
        print("=" * 70)
        
        # Run standard evaluation and collect results
        all_results = {}
        test_cases = list(self.ground_truth["test_cases"].keys())
        
        for dataset_name, project_ids_to_use in self.ground_truth["projects"].items():
            print(f"\n{'='*60}")
            print(f"Evaluating {dataset_name.upper()} dataset...")
            print(f"{'='*60}")
            
            full_dataset = self.full_datasets[dataset_name]
            curated_pool_df = full_dataset[full_dataset["project_id"].isin(project_ids_to_use)].copy()
            print(f"  - Using a curated pool of {len(curated_pool_df)} projects for this evaluation.")
            
            all_ndcg_scores = []
            all_f1_scores = []
            
            # Set the agent's internal state for this dataset's evaluation run
            self.agent.projects_df = curated_pool_df
            self.agent.keyword_weight = 7
            self.agent.moral_weight = 3
            
            for prompt, matches_by_dataset in self.ground_truth["test_cases"].items():
                matches = matches_by_dataset.get(dataset_name)
                
                print(f"\n--- Test Case: \"{prompt[:50]}...\" ---")
                
                if not matches or not (matches.get("perfect") or matches.get("good")):
                    print("  No 'perfect' or 'good' matches defined for this case. Skipping.")
                    continue
                
                # Use the correct public method from the agent
                results = self.agent.generate_recommendations(user_input=prompt, top_n=5)
                recommendations_df = results['recommendations']
                
                recommended_project_ids = [rec['project_id'] for rec in recommendations_df]
                
                ndcg_score = self._calculate_ndcg_at_5(recommended_project_ids, matches)
                precision, recall, f1_score = self._calculate_top_k_f1_at_5(recommended_project_ids, matches)
                
                all_ndcg_scores.append(ndcg_score)
                all_f1_scores.append(f1_score)
                
                print(f"  - Top 5 Recommended Project IDs: {recommended_project_ids}")
                print(f"  - Metrics for this case:")
                print(f"    - NDCG@5:   {ndcg_score:.4f}")
                print(f"    - F1@5:     {f1_score:.4f} (Precision: {precision:.2f}, Recall: {recall:.2f})")
            
            # Calculate summary statistics
            avg_ndcg = sum(all_ndcg_scores) / len(all_ndcg_scores) if all_ndcg_scores else 0.0
            max_ndcg = max(all_ndcg_scores) if all_ndcg_scores else 0.0
            min_ndcg = min(all_ndcg_scores) if all_ndcg_scores else 0.0
            
            avg_f1 = sum(all_f1_scores) / len(all_f1_scores) if all_f1_scores else 0.0
            max_f1 = max(all_f1_scores) if all_f1_scores else 0.0
            min_f1 = min(all_f1_scores) if all_f1_scores else 0.0
            
            # Store results
            all_results[dataset_name] = {
                'ndcg5': {
                    'average': avg_ndcg,
                    'max': max_ndcg,
                    'min': min_ndcg,
                    'individual_scores': all_ndcg_scores
                },
                'f1_5': {
                    'average': avg_f1,
                    'max': max_f1,
                    'min': min_f1,
                    'individual_scores': all_f1_scores
                },
                'total_test_cases': len(all_ndcg_scores)
            }
            
            print("\n" + "=" * 80)
            print(f"AVERAGE SCORES FOR '{dataset_name.upper()}' DATASET:")
            print(f"  - Average NDCG@5: {avg_ndcg:.4f}")
            print(f"  - Average F1@5:   {avg_f1:.4f}")
            print("=" * 80)
        
        # Generate graphs and save CSV results
        print(f"\n{'='*60}")
        print("Generating evaluation visualizations and saving results...")
        print(f"{'='*60}")
        
        self.create_evaluation_graphs(all_results, test_cases)
        self.save_results_to_csv(all_results, test_cases)
        
        # Run weight optimization for each dataset
        print(f"\n{'='*60}")
        print("Running weight optimization for each dataset...")
        print(f"{'='*60}")
        
        optimization_results = {}
        for dataset_name in self.ground_truth["projects"].keys():
            print(f"\nOptimizing weights for {dataset_name.upper()} dataset...")
            try:
                opt_results = self.optimize_weights(dataset_name)
                optimization_results[dataset_name] = opt_results
                print(f"[OK] {dataset_name.capitalize()} optimization completed!")
            except Exception as e:
                print(f"[ERROR] Error optimizing {dataset_name} dataset: {e}")
                optimization_results[dataset_name] = None
        
        # Print final summary
        print(f"\n{'='*70}")
        print("FINAL COMPREHENSIVE EVALUATION SUMMARY")
        print(f"{'='*70}")
        
        for dataset_name in self.ground_truth["projects"].keys():
            results = all_results[dataset_name]
            print(f"\n{dataset_name.upper()}:")
            print(f"  NDCG@5: {results['ndcg5']['average']:.3f} (max: {results['ndcg5']['max']:.3f}, min: {results['ndcg5']['min']:.3f})")
            print(f"  F1@5: {results['f1_5']['average']:.3f} (max: {results['f1_5']['max']:.3f}, min: {results['f1_5']['min']:.3f})")
            
            if optimization_results[dataset_name]:
                opt = optimization_results[dataset_name]
                print(f"  Best weights by combined score: K={opt['best_combined']['keyword_weight']}, M={opt['best_combined']['moral_weight']}")
        
        return {
            'evaluation_results': all_results,
            'optimization_results': optimization_results
        }

if __name__ == "__main__":
    evaluator = RecommendationEvaluator()
    evaluator.run_comprehensive_evaluation()