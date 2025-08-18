"""
Demo script for the Project Recommendation Agent.
"""

from conversational_agent import ProjectRecommendationAgent

def demo_conversational_agent():
    """Demo the conversational agent with example inputs."""
    
    print("🚀 Starting Project Recommendation Agent Demo...")
    
    # Initialize agent
    agent = ProjectRecommendationAgent(
        votes_csv_path='data/parsed/worldwide_mechanical-turk/votes.csv',
        projects_csv_path='data/parsed/worldwide_mechanical-turk/projects.csv'
    )
    
    # Example user inputs
    example_inputs = [
        "I'm a young environmentalist who cares about sustainability and green energy",
        "As a senior citizen, I value safety and accessibility in our community",
        "I'm a college student who believes in education and community development",
        "I care about public health and want to make our streets safer for everyone"
    ]
    
    print(f"\n💬 Demo Examples:")
    print("=" * 60)
    
    for i, example in enumerate(example_inputs, 1):
        print(f"\n📝 Example {i}: {example}")
        print("-" * 40)
        
        try:
            response = agent.chat_and_recommend(example)
            print(response)
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("\n" + "=" * 60)
    
    print("\n🎉 Demo completed! Run 'python conversational_agent.py' for interactive chat.")

if __name__ == "__main__":
    demo_conversational_agent()
