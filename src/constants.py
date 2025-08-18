"""
Constants for the moral reasoning value extraction system.
"""

# Moral Foundations Theory labels (common in moral reasoning)
MORAL_FOUNDATIONS = [
    "Care/Harm",
    "Fairness/Cheating", 
    "Loyalty/Betrayal",
    "Authority/Subversion",
    "Sanctity/Degradation",
    "Liberty/Oppression"
]

# Additional value categories
VALUE_CATEGORIES = [
    "Justice", "Compassion", "Honesty", "Respect", "Responsibility",
    "Courage", "Integrity", "Generosity", "Forgiveness", "Humility"
]

# Default model configuration
DEFAULT_MODEL_NAME = "moralstories/roberta-large_action-context-consequence"
DEFAULT_THRESHOLD = 0.5
DEFAULT_TOP_K = 5

# Keyword mappings for moral foundations
MORAL_FOUNDATION_KEYWORDS = {
    "Care/Harm": ['care', 'harm', 'hurt', 'protect', 'nurture', 'damage', 'safety', 'wellbeing'],
    "Fairness/Cheating": ['fair', 'unfair', 'cheat', 'equal', 'justice', 'rights', 'discrimination'],
    "Loyalty/Betrayal": ['loyal', 'betray', 'trust', 'faithful', 'treason', 'commitment', 'allegiance'],
    "Authority/Subversion": ['authority', 'obey', 'respect', 'disobey', 'leadership', 'hierarchy', 'command'],
    "Sanctity/Degradation": ['sacred', 'pure', 'holy', 'degrading', 'vulgar', 'profane', 'spiritual'],
    "Liberty/Oppression": ['freedom', 'liberty', 'oppress', 'restrict', 'autonomy', 'independence', 'control']
}

# Keyword mappings for general values
GENERAL_VALUE_KEYWORDS = {
    "Justice": ['justice', 'fair', 'equality', 'rights', 'law', 'court'],
    "Compassion": ['compassion', 'kindness', 'empathy', 'help', 'care', 'support'],
    "Honesty": ['honest', 'truth', 'lie', 'deceive', 'transparent', 'sincere'],
    "Respect": ['respect', 'dignity', 'honor', 'courtesy', 'polite', 'considerate'],
    "Responsibility": ['responsibility', 'duty', 'obligation', 'accountable', 'liable']
}

# Confidence scores for keyword-based detection
KEYWORD_CONFIDENCE_SCORES = {
    "moral_foundations": 0.75,
    "general_values": 0.8
}

# Moral recommendations for different values
MORAL_RECOMMENDATIONS = {
    'Care/Harm': "Prioritize actions that promote well-being and avoid harm to others.",
    'Fairness/Cheating': "Ensure equitable treatment and avoid exploiting others.",
    'Loyalty/Betrayal': "Maintain commitments and trust in relationships.",
    'Authority/Subversion': "Respect legitimate authority while questioning unjust practices.",
    'Sanctity/Degradation': "Honor sacred aspects of life and avoid degradation.",
    'Liberty/Oppression': "Support individual freedoms while preventing oppression.",
    'Justice': "Strive for fairness and equal treatment for all.",
    'Compassion': "Act with empathy and kindness toward others.",
    'Honesty': "Maintain truthfulness and transparency in all actions.",
    'Respect': "Treat others with dignity and consideration."
}

# Model configuration
MODEL_CONFIG = {
    "max_length": 512,
    "truncation": True,
    "padding": True
}

# Analysis thresholds
ANALYSIS_THRESHOLDS = {
    "high_confidence": 0.7,
    "medium_confidence": 0.4,
    "low_confidence": 0.3
}
