"""
Run Comprehensive Model Comparison

This script runs a comprehensive comparison of different moral value classification models
and generates detailed analytics and visualizations.
"""

import os
import sys
import logging
import pandas as pd
from datetime import datetime
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model_comparison import ModelComparisonFramework
from src.enhanced_moral_classifier import EnhancedMoralValueClassifier, create_classifier_comparison
from src.utils import print_separator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def run_comprehensive_comparison():
    """
    Run comprehensive model comparison with detailed analytics.
    """
    print("COMPREHENSIVE MODEL COMPARISON")
    print("=" * 60)
    
    # Initialize framework
    framework = ModelComparisonFramework()
    
    # Check if test data exists
    if not os.path.exists(framework.test_data_path):
        print(f"Test data not found at {framework.test_data_path}")
        print("Creating synthetic test data...")
        create_synthetic_test_data()
    
    # Run comparison
    print("Running model comparison...")
    comparison_results = framework.compare_models()
    
    if 'error' in comparison_results:
        print(f"Error: {comparison_results['error']}")
        return
    
    # Generate visualizations
    print("Generating visualizations...")
    framework.generate_visualizations()
    
    # Save results
    print("Saving results...")
    results_file, summary_file = framework.save_results()
    
    # Print summary
    print("\nComparison completed!")
    print(f"Results saved to: {results_file}")
    print(f"Summary saved to: {summary_file}")
    
    # Print detailed recommendations
    if 'comparison_analysis' in comparison_results:
        analysis = comparison_results['comparison_analysis']
        print("\n" + "="*60)
        print("DETAILED RECOMMENDATIONS")
        print("="*60)
        
        for rec in analysis.get('recommendations', []):
            print(f"  • {rec}")
        
        # Print detailed metrics comparison
        if 'summary' in analysis and 'metrics_comparison' in analysis['summary']:
            print("\n" + "="*60)
            print("DETAILED METRICS COMPARISON")
            print("="*60)
            
            metrics = analysis['summary']['metrics_comparison']
            for model_name, model_metrics in metrics.items():
                print(f"\n{framework.models[model_name]['name']}:")
                print(f"  • Distribution Entropy: {model_metrics['distribution_entropy']:.3f}")
                print(f"  • Mean Confidence: {model_metrics['mean_confidence']:.3f}")
                print(f"  • Consistency: {model_metrics['consistency']:.3f}")
                print(f"  • Diversity: {model_metrics['diversity']:.3f}")
                print(f"  • Max Foundation %: {model_metrics['max_foundation_percentage']:.1f}%")

def create_synthetic_test_data():
    """
    Create synthetic test data for comparison if none exists.
    """
    print("Creating synthetic test data...")
    
    # Sample project descriptions covering different moral foundations
    test_descriptions = [
        # Care/Harm examples
        "Providing healthcare services and medical assistance to underserved communities",
        "Child safety programs and protective services for vulnerable families",
        "Mental health support and counseling services for residents",
        "Emergency response and disaster relief infrastructure",
        
        # Fairness/Cheating examples
        "Equal access to education and training programs for all residents",
        "Anti-discrimination initiatives and equal opportunity programs",
        "Fair housing policies and affordable housing development",
        "Workplace equality and fair employment practices",
        
        # Loyalty/Betrayal examples
        "Community building events and neighborhood cohesion programs",
        "Cultural heritage preservation and traditional values support",
        "Local business support and community loyalty initiatives",
        "Veteran support services and patriotic community programs",
        
        # Authority/Subversion examples
        "Law enforcement training and public safety programs",
        "Government transparency and civic education initiatives",
        "Traffic law enforcement and road safety regulations",
        "Building code enforcement and regulatory compliance",
        
        # Sanctity/Degradation examples
        "Religious and spiritual community center development",
        "Environmental conservation and natural resource protection",
        "Cultural preservation and sacred site maintenance",
        "Clean water and air quality protection programs",
        
        # Liberty/Oppression examples
        "Civil rights advocacy and freedom of expression support",
        "Individual privacy protection and data rights initiatives",
        "Personal autonomy and choice in healthcare decisions",
        "Freedom of movement and transportation accessibility"
    ]
    
    # Create DataFrame
    test_data = pd.DataFrame({
        'description': test_descriptions,
        'category': ['test'] * len(test_descriptions)
    })
    
    # Save to file
    os.makedirs(os.path.dirname("data/generated/"), exist_ok=True)
    test_data.to_csv("data/generated/content.csv", index=False)
    print(f"Created test data with {len(test_descriptions)} samples")

def run_individual_model_analysis():
    """
    Run detailed analysis for each individual model.
    """
    print("\n" + "="*60)
    print("INDIVIDUAL MODEL ANALYSIS")
    print("="*60)
    
    models = [
        "facebook/bart-large-mnli",
        "roberta-large-mnli"
    ]
    
    # Load test data
    test_data_path = "data/generated/content.csv"
    if os.path.exists(test_data_path):
        test_data = pd.read_csv(test_data_path)
        test_texts = test_data['description'].tolist()[:20]  # Use first 20 for detailed analysis
    else:
        print("Test data not found, using sample texts")
        test_texts = [
            "Equal access to education for all students regardless of background",
            "Community safety and crime prevention programs",
            "Environmental protection and sustainability initiatives",
            "Cultural diversity and inclusion programs",
            "Supporting local businesses and economic development"
        ]
    
    for model_name in models:
        print(f"\n{'='*50}")
        print(f"Analyzing: {model_name}")
        print(f"{'='*50}")
        
        try:
            # Create classifier
            classifier = EnhancedMoralValueClassifier(model_name)
            
            # Classify all test texts
            results = []
            for text in test_texts:
                result = classifier.classify_text(text)
                results.append(result)
            
            # Get metrics
            metrics = classifier.get_evaluation_metrics()
            
            # Print results
            print(f"Total samples processed: {metrics['total_samples']}")
            print(f"Mean confidence: {metrics['mean_confidence']:.3f}")
            print(f"Distribution entropy: {metrics['distribution_entropy']:.3f}")
            print(f"Max foundation percentage: {metrics['max_foundation_percentage']:.1f}%")
            
            print("\nFoundation Distribution:")
            for foundation, count in metrics['foundation_distribution'].items():
                percentage = metrics['foundation_percentages'][foundation]
                print(f"  • {foundation}: {count} ({percentage:.1f}%)")
            
            # Save individual report
            report_path = classifier.save_evaluation_report()
            print(f"\nDetailed report saved to: {report_path}")
            
            # Show sample classifications
            print(f"\nSample Classifications:")
            for i, result in enumerate(results[:3]):
                if 'error' not in result:
                    print(f"  {i+1}. '{test_texts[i][:50]}...'")
                    print(f"     → {result['dominant_foundation']} ({result['confidence']:.3f})")
                    print(f"     → {result['analysis']}")
            
        except Exception as e:
            print(f"Error analyzing {model_name}: {e}")

def generate_comparison_report():
    """
    Generate a comprehensive comparison report.
    """
    print("\n" + "="*60)
    print("GENERATING COMPREHENSIVE REPORT")
    print("="*60)
    
    # Create output directory
    output_dir = "results/comparisons"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"{output_dir}/comprehensive_comparison_report_{timestamp}.md"
    
    with open(report_file, 'w') as f:
        f.write("# Moral Value Classification Model Comparison Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Overview\n\n")
        f.write("This report compares three different models for moral value classification:\n\n")
        f.write("1. **facebook/bart-large-mnli** - Current baseline model\n")
        f.write("2. **roberta-large-mnli** - Alternative transformer model\n")
        f.write("3. **microsoft/DialoGPT-medium** - Conversational model\n\n")
        
        f.write("## Evaluation Approach\n\n")
        f.write("Since we don't have ground truth labels, we evaluate models using:\n\n")
        f.write("- **Distribution Analysis**: How balanced are the moral foundation assignments?\n")
        f.write("- **Confidence Analysis**: How confident are the models in their predictions?\n")
        f.write("- **Consistency Analysis**: How similar are classifications for similar texts?\n")
        f.write("- **Diversity Analysis**: How well do models cover all moral foundations?\n\n")
        
        f.write("## Key Metrics\n\n")
        f.write("- **Distribution Entropy**: Higher values indicate more balanced distributions\n")
        f.write("- **Mean Confidence**: Average confidence across all classifications\n")
        f.write("- **Consistency**: Average pairwise similarity between classifications\n")
        f.write("- **Diversity**: Percentage of moral foundations actually used\n")
        f.write("- **Max Foundation %**: Percentage of most common foundation (lower is better)\n\n")
        
        f.write("## Recommendations\n\n")
        f.write("1. **For Balanced Results**: Choose models with high distribution entropy and low max foundation percentage\n")
        f.write("2. **For High Confidence**: Choose models with high mean confidence scores\n")
        f.write("3. **For Consistency**: Choose models with high consistency scores\n")
        f.write("4. **For Diversity**: Choose models with high diversity scores\n\n")
        
        f.write("## Files Generated\n\n")
        f.write("- `model_comparison_results_*.json`: Detailed results for each model\n")
        f.write("- `comparison_summary_*.txt`: Text summary of comparison\n")
        f.write("- `foundation_distributions.png`: Visual comparison of foundation distributions\n")
        f.write("- `confidence_comparison.png`: Visual comparison of confidence distributions\n")
        f.write("- `consistency_analysis.png`: PCA visualization of classification consistency\n")
        f.write("- `overall_metrics_radar.png`: Radar chart of overall metrics\n\n")
        
        f.write("## Next Steps\n\n")
        f.write("1. Review the visualizations to understand model behavior\n")
        f.write("2. Consider the trade-offs between different metrics\n")
        f.write("3. Test the best-performing model on your specific use case\n")
        f.write("4. Consider ensemble approaches combining multiple models\n\n")
    
    print(f"Comprehensive report saved to: {report_file}")

def main():
    """
    Main function to run the complete comparison.
    """
    print("Starting comprehensive model comparison...")
    
    # Run comprehensive comparison
    run_comprehensive_comparison()
    
    # Run individual model analysis
    run_individual_model_analysis()
    
    # Generate comprehensive report
    generate_comparison_report()
    
    print("\n" + "="*60)
    print("COMPARISON COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nCheck the following directories for results:")
    print("  • results/comparisons/ - Comparison results and visualizations")
    print("  • results/evaluation/ - Individual model evaluation reports")
    print("\nKey files to review:")
    print("  • foundation_distributions.png - Visual comparison of model outputs")
    print("  • overall_metrics_radar.png - Overall performance comparison")
    print("  • comparison_summary_*.txt - Text summary of findings")

if __name__ == "__main__":
    main()
