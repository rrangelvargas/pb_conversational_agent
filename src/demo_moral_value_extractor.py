"""
Demo script for the Moral Value Extractor system.
"""

from moral_value_extractor import MoralValueExtractor


def main():
    """
    Main function to demonstrate the moral value extractor.
    """
    print("Moral Reasoning Value Extraction Model")
    print("=" * 40)
    
    # Initialize the model
    try:
        extractor = MoralValueExtractor()
        print("✓ Model initialized successfully")
    except Exception as e:
        print(f"✗ Error initializing model: {e}")
        return
    
    # Example texts for testing
    test_texts = [
        "The judge showed great compassion while maintaining justice in the courtroom.",
        "It's important to be honest even when telling the truth might hurt someone's feelings.",
        "The government should respect individual liberties while ensuring public safety.",
        "We must take responsibility for our actions and their consequences.",
        "Showing respect for different cultures promotes understanding and harmony."
    ]
    
    print("\nTesting Value Extraction:")
    print("-" * 30)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nText {i}: {text}")
        values = extractor.extract_values(text)
        
        if values:
            print("Detected values:")
            for value, score in sorted(values.items(), key=lambda x: x[1], reverse=True):
                print(f"  • {value}: {score:.3f}")
        else:
            print("No significant values detected.")
    
    # Test moral dilemma analysis
    print("\n" + "=" * 40)
    print("Moral Dilemma Analysis:")
    print("-" * 30)
    
    dilemma_text = "A doctor must choose between saving one critically ill patient or using limited resources to help many others with less severe conditions."
    
    print(f"Dilemma: {dilemma_text}")
    analysis = extractor.analyze_moral_dilemma(dilemma_text)
    
    print(f"Primary values: {', '.join(analysis['primary_values'])}")
    print(f"Secondary values: {', '.join(analysis['secondary_values'])}")
    print(f"Potential conflicts: {', '.join(analysis['potential_conflicts'])}")
    print(f"Moral complexity: {analysis['moral_complexity']:.2f}")
    print(f"Recommendation: {analysis['recommendation']}")
    
    # Batch processing example
    print("\n" + "=" * 40)
    print("Batch Processing Example:")
    print("-" * 30)
    
    batch_results = extractor.batch_extract_values(test_texts[:3])
    for i, (text, values) in enumerate(zip(test_texts[:3], batch_results)):
        print(f"\nBatch {i+1}: {text[:50]}...")
        top_values = extractor.get_top_values(text, top_k=3)
        for value, score in top_values:
            print(f"  • {value}: {score:.3f}")


if __name__ == "__main__":
    main()
