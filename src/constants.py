"""
Constants for the moral reasoning value extraction system
and project recommendation system.
"""

# List of moral foundation names
MORAL_FOUNDATIONS = [
    "Authority",
    "Care", 
    "Fairness",
    "Loyalty",
    "Non-Moral",
    "Sanctity"
]

MORAL_FOUNDATIONS_TO_USE = [
    "Authority",
    "Care", 
    "Fairness",
    "Loyalty",
    "Sanctity"
]

# Category keywords for classification and matching
CATEGORY_KEYWORDS = {
    "Education": [
        'education', 'educational', 'workshops', 'school', 'classes', 'integration', 'social', 'library', 'english', 'youth', 
        'primary', 'reading', 'learning', 'students', 'parents', 'books', 'schools', 
        'equipment', 'training', 'children', 'instruction', 'college', 'high school', 'mentoring', 'writing',
        'diploma', 'curriculum', 'university', 'courses', 'parent', 'math', 'steam', 'kindergarten',
        'secondary', 'young', 'course', 'parenting', 'literate', 'tutoring', 'scholarship', 'literacy', 'knowledge',
        'learn', 'classroom', 'classrooms', 'class'
    ],
    "Environment, Public heath and Safety": [
        'health', 'medical', 'safety', 'meadow', 'greenery', 'protect', 'park', 'planting', 'garden', 'shrubs', 'protection', 'green', 
        'eco', 'forest', 'birds', 'trees', 'compassion', 'reduce', 'revitalization', 'treatment', 'habitat', 
        'preparedness', 'assistance', 'chronic', 'safe', 'medicine', 'wellness', 'emergency', 'ecosystem', 
        'landfill', 'pharmacy', 'fitness', 'infection', 'pollution', 'illness', 'medication', 'vaccines', 
        'vaccinations', 'air quality', 'monitoring', 'environmental', 'sustainability', 'solar', 'energy', 
        'waste', 'reduction', 'recycling', 'composting', 'roof', 'efficiency', 'stormwater', 'runoff', 
        'heat island', 'shade', 'aesthetics', 'need'
    ],
    "Culture and Community": [
        'community', 'culture', 'cultural', 'events', 'meetings', 'benches', 'lighting', 'care', 'promotes', 'recreation', 'gym', 
        'women', 'kindness', 'outdoor', 'people', 'modernization', 'revitalization', 'children', 'local', 
        'traditional', 'parent', 'gatherings', 'assistance', 'museum', 'society', 'social', 'exhibit', 
        'theater', 'youth', 'resident', 'exhibition', 'groups', 'festival', 'services', 'unity', 'festivals', 
        'organization', 'gathering', 'communities', 'center', 'cultural center', 'diversity' 'family', 'need'
    ],
    "Transportation": [
        'transportation', 'safety', 'parking', 'crossing', 'bike', 'path', 'road', 'construction', 'surface', 'bicycle', 'safe', 
        'streets', 'lighting', 'street', 'pavement', 'improving', 'improve', 'bus', 'traffic', 'pedestrian', 
        'modernization', 'intersection', 'replacement', 'roads', 'ramps', 'signals', 'crossings', 
        'marking', 'tunnel', 'cyclist', 'transport', 'lifts', 'stations', 'station', 'walkways', 'bicycles', 
        'transit', 'garages', 'lots', 'lift', 'lane', 'sidewalk', 'repair', 'installation', 'accessibility'
    ],
    "Recreation": [
        'recreation', 'programs', 'activities', 'playground', 'yoga', 'classes', 'tennis', 'workout', 'park', 'fitness', 'seniors',
        'gym', 'outdoor', 'sports', 'family', 'equipment', 'training', 'children', 'volleyball', 
        'lessons', 'indoors', 'splash pad', 'gear', 'leisure', 'center', 'tables', 'activity', 'picnicking',
        'materials', 'fields', 'barbecue', 'water', 'fun', 'playing', 'gymnasium', 'gymnasiums', 'swim', 'court', 
        'playgrounds', 'recreational', 'leisure activities', 'sports equipment', 'playground equipment'
    ],
    "Other": [] # "Other" category has no keywords and is used as a fallback
}

OPTIMAL_WEIGHTS = {
    "synthetic": {"keyword": 7.0, "moral": 3.0},
    "poland": {"keyword": 1.0, "moral": 9.0},
    "worldwide": {"keyword": 6.0, "moral": 4.0}
}

# Paths to data files
SYNTHETIC_DATA_PATH = "../data/balanced_synthetic_projects_with_moral_scores.csv"
POLAND_DATA_PATH = "../data/poland_warszawa_projects_with_moral_scores.csv"
WORLDWIDE_DATA_PATH = "../data/worldwide_mechanical_projects_with_moral_scores.csv"

# Standardized categories list
STANDARDIZED_CATEGORIES = [
    "Education", 
    "Environment, Public heath and Safety", 
    "Culture and Community", 
    "Transportation", 
    "Recreation", 
    "Other"
]

# Category mapping dictionary (case-insensitive)
CATEGORY_MAPPING = {
    # Warsaw Dataset Mappings
    "public transit and roads": "Transportation",
    "education": "Education", 
    "education and science": "Education",
    "culture": "Culture and Community",
    "sport": "Recreation",
    "sport and recreation": "Recreation", 
    "public space": "Culture and Community",
    "environmental protection": "Environment, Public heath and Safety",
    "nature and greenery": "Environment, Public heath and Safety",
    "urban greenery": "Environment, Public heath and Safety",
    "community building": "Culture and Community",
    "health and welfare": "Environment, Public heath and Safety",
    "welfare": "Environment, Public heath and Safety",
    "health": "Environment, Public heath and Safety",
    
    # Worldwide Dataset Mappings
    "environment, public health & safety": "Environment, Public heath and Safety",
    "culture & community": "Culture and Community",
    "streets, sidewalks & transit": "Transportation",
    "facilities, parks & recreation": "Recreation",
    
    "transportation": "Transportation",
    "culture and community": "Culture and Community",
    "recreation": "Recreation",
    "environment, public heath and safety": "Environment, Public heath and Safety",
    "other": "Other"
}