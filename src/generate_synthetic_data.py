"""
Generate comprehensive synthetic data for the conversational agent project.
Combines both project and voting data generation with diverse categories and moral values.
"""

import pandas as pd
import numpy as np
import random
from typing import List, Dict, Tuple
import json
from utils import save_csv_data, print_separator

# Define all categories from both datasets
ALL_CATEGORIES = [
    # From worldwide_mechanical-turk
    "Culture & community",
    "Education", 
    "Environment, public health & safety",
    "Facilities, parks & recreation",
    "Streets, Sidewalks & Transit",
    
    # From Warsaw dataset
    "urban greenery",
    "sport",
    "public space", 
    "public transit and roads",
    "welfare",
    "environmental protection",
    "health",
    "culture"
]

# Moral foundations
MORAL_FOUNDATIONS = [
    "Care/Harm", "Fairness/Cheating", "Loyalty/Betrayal",
    "Authority/Subversion", "Sanctity/Degradation", "Liberty/Oppression"
]

# Target demographics
TARGET_DEMOGRAPHICS = [
    "children", "youth", "adults", "seniors", "people with disabilities"
]

# Education levels
EDUCATION_LEVELS = [
    "high school", "some college", "college", "graduate degree"
]

# Age groups
AGE_GROUPS = [
    (18, 25), (26, 35), (36, 45), (46, 55), (56, 65), (66, 75), (76, 85)
]

# Genders
GENDERS = ["M", "F"]

# Income levels
INCOME_LEVELS = [
    "under 25000", "25000-50000", "50000-75000", "75000-100000", "over 100000"
]

# Political orientations
POLITICAL_ORIENTATIONS = [
    "very liberal", "liberal", "moderate", "conservative", "very conservative"
]

def generate_specific_project_templates():
    """Generate specific project templates for each category with detailed descriptions."""
    
    project_templates = {
        "Culture & community": [
            {
                "name": "Multilingual Community Center for New Immigrants",
                "description": "A comprehensive community center offering language classes, cultural exchange programs, job training, and social services for recent immigrants. The center will provide a welcoming space for cultural integration while preserving heritage.",
                "cost_range": (500000, 2000000),
                "moral_value": "Care/Harm"
            },
            {
                "name": "Intergenerational Storytelling Festival",
                "description": "Annual festival bringing together seniors and youth to share stories, traditions, and life experiences. Includes workshops, performances, and digital storytelling projects to bridge generational gaps.",
                "cost_range": (50000, 200000),
                "moral_value": "Loyalty/Betrayal"
            },
            {
                "name": "Public Art Mural Program",
                "description": "Commission local artists to create murals celebrating community diversity, history, and values. Each mural tells a story about the neighborhood and its people.",
                "cost_range": (25000, 100000),
                "moral_value": "Sanctity/Degradation"
            },
            {
                "name": "Community Cultural Heritage Museum",
                "description": "Interactive museum showcasing local history, immigrant stories, and cultural artifacts. Features rotating exhibits, educational programs, and community event spaces.",
                "cost_range": (800000, 2500000),
                "moral_value": "Loyalty/Betrayal"
            },
            {
                "name": "Neighborhood Block Party Program",
                "description": "Organized community events bringing together residents for food, music, and activities. Includes funding for permits, equipment, and promotional materials.",
                "cost_range": (15000, 75000),
                "moral_value": "Loyalty/Betrayal"
            }
        ],
        
        "Education": [
            {
                "name": "STEM Innovation Lab for Underprivileged Youth",
                "description": "State-of-the-art science, technology, engineering, and math laboratory providing hands-on learning experiences for students from low-income families. Includes robotics, coding, and maker space equipment.",
                "cost_range": (300000, 800000),
                "moral_value": "Fairness/Cheating"
            },
            {
                "name": "Digital Literacy Program for Seniors",
                "description": "Comprehensive computer and internet training program for elderly residents, helping them stay connected with family, access online services, and participate in the digital economy.",
                "cost_range": (75000, 150000),
                "moral_value": "Care/Harm"
            },
            {
                "name": "Outdoor Environmental Education Center",
                "description": "Nature-based learning facility with trails, observation decks, and interactive exhibits teaching environmental science, sustainability, and conservation to students of all ages.",
                "cost_range": (200000, 500000),
                "moral_value": "Sanctity/Degradation"
            },
            {
                "name": "Adult Vocational Training Center",
                "description": "Skills development programs for adults seeking career changes or job training. Offers courses in healthcare, technology, construction, and service industries.",
                "cost_range": (400000, 1000000),
                "moral_value": "Fairness/Cheating"
            },
            {
                "name": "Early Childhood Development Program",
                "description": "Comprehensive early learning initiative for children ages 0-5, including parent education, developmental screenings, and school readiness activities.",
                "cost_range": (250000, 600000),
                "moral_value": "Care/Harm"
            }
        ],
        
        "Environment, public health & safety": [
            {
                "name": "Urban Air Quality Monitoring Network",
                "description": "Installation of 50 air quality sensors across the city to monitor pollution levels, provide real-time data to residents, and help identify areas needing environmental intervention.",
                "cost_range": (150000, 400000),
                "moral_value": "Care/Harm"
            },
            {
                "name": "Community Emergency Response Training Program",
                "description": "Comprehensive disaster preparedness training for residents, including first aid, emergency communication, evacuation procedures, and community coordination during crises.",
                "cost_range": (100000, 250000),
                "moral_value": "Care/Harm"
            },
            {
                "name": "Green Infrastructure for Stormwater Management",
                "description": "Installation of rain gardens, permeable pavements, and green roofs to reduce flooding, improve water quality, and create urban habitats for wildlife.",
                "cost_range": (300000, 800000),
                "moral_value": "Sanctity/Degradation"
            },
            {
                "name": "Community Noise Reduction Initiative",
                "description": "Sound barrier installations, traffic calming measures, and noise monitoring systems to reduce urban noise pollution and improve quality of life.",
                "cost_range": (200000, 500000),
                "moral_value": "Care/Harm"
            },
            {
                "name": "Public Health Surveillance System",
                "description": "Integrated health monitoring network tracking disease outbreaks, environmental health risks, and community health trends for proactive public health interventions.",
                "cost_range": (300000, 700000),
                "moral_value": "Care/Harm"
            }
        ],
        
        "Facilities, parks & recreation": [
            {
                "name": "Universal Access Playground for All Abilities",
                "description": "Inclusive playground designed for children of all physical and cognitive abilities, featuring wheelchair-accessible equipment, sensory play areas, and quiet spaces for children with autism.",
                "cost_range": (400000, 1000000),
                "moral_value": "Fairness/Cheating"
            },
            {
                "name": "Community Fitness Center with Senior Programs",
                "description": "Multi-purpose fitness facility offering specialized exercise programs for seniors, including low-impact aerobics, strength training, balance classes, and social fitness groups.",
                "cost_range": (600000, 1500000),
                "moral_value": "Care/Harm"
            },
            {
                "name": "Urban Farm and Community Garden Complex",
                "description": "Educational urban agriculture center with community plots, greenhouses, composting facilities, and programs teaching sustainable food production and nutrition.",
                "cost_range": (250000, 600000),
                "moral_value": "Sanctity/Degradation"
            },
            {
                "name": "Indoor Sports Complex",
                "description": "Multi-sport facility with basketball courts, indoor soccer fields, swimming pool, and fitness areas for year-round athletic activities regardless of weather.",
                "cost_range": (800000, 2000000),
                "moral_value": "Loyalty/Betrayal"
            },
            {
                "name": "Community Arts and Crafts Studio",
                "description": "Creative space with pottery wheels, painting supplies, woodworking tools, and textile equipment for community art classes and individual projects.",
                "cost_range": (150000, 400000),
                "moral_value": "Sanctity/Degradation"
            }
        ],
        
        "Streets, Sidewalks & Transit": [
            {
                "name": "Complete Streets Initiative for Pedestrian Safety",
                "description": "Redesign of major corridors to prioritize pedestrian and cyclist safety with wider sidewalks, protected bike lanes, improved crosswalks, and traffic calming measures.",
                "cost_range": (800000, 2000000),
                "moral_value": "Care/Harm"
            },
            {
                "name": "Smart Traffic Management System",
                "description": "Intelligent traffic signals and sensors to reduce congestion, improve emergency vehicle response times, and provide real-time traffic information to commuters.",
                "cost_range": (500000, 1200000),
                "moral_value": "Authority/Subversion"
            },
            {
                "name": "Electric Vehicle Charging Infrastructure",
                "description": "Network of 100 public electric vehicle charging stations strategically placed throughout the city to support the transition to clean transportation and reduce emissions.",
                "cost_range": (300000, 800000),
                "moral_value": "Sanctity/Degradation"
            },
            {
                "name": "Pedestrian Bridge Network",
                "description": "Safe crossing infrastructure connecting neighborhoods separated by major roads, including accessible ramps and lighting for 24/7 use.",
                "cost_range": (400000, 1000000),
                "moral_value": "Care/Harm"
            },
            {
                "name": "Public Transit Real-Time Tracking",
                "description": "GPS tracking and mobile app integration for all public transportation, providing real-time arrival information and route optimization.",
                "cost_range": (200000, 500000),
                "moral_value": "Fairness/Cheating"
            }
        ],
        
        "urban greenery": [
            {
                "name": "Urban Forest Restoration and Expansion",
                "description": "Large-scale tree planting initiative to restore degraded urban forests, create wildlife corridors, improve air quality, and provide natural cooling for heat-vulnerable neighborhoods.",
                "cost_range": (400000, 1000000),
                "moral_value": "Sanctity/Degradation"
            },
            {
                "name": "Community Orchard and Food Forest",
                "description": "Public fruit and nut tree groves providing free food for residents, educational opportunities about sustainable agriculture, and habitat for urban wildlife.",
                "cost_range": (150000, 400000),
                "moral_value": "Care/Harm"
            },
            {
                "name": "Green Roof and Living Wall Program",
                "description": "Installation of vegetation on building rooftops and walls to reduce urban heat island effect, improve building energy efficiency, and create urban biodiversity.",
                "cost_range": (200000, 600000),
                "moral_value": "Sanctity/Degradation"
            },
            {
                "name": "Urban Wetland Restoration",
                "description": "Rehabilitation of degraded urban wetlands to improve water quality, provide flood control, and create habitat for native wildlife and migratory birds.",
                "cost_range": (300000, 800000),
                "moral_value": "Sanctity/Degradation"
            },
            {
                "name": "Community Garden Network",
                "description": "Coordinated system of neighborhood gardens with shared resources, educational programs, and produce distribution to address food insecurity.",
                "cost_range": (100000, 300000),
                "moral_value": "Care/Harm"
            }
        ],
        
        "sport": [
            {
                "name": "Adaptive Sports Complex for People with Disabilities",
                "description": "Specialized sports facility with equipment and programs designed for athletes with physical and cognitive disabilities, promoting inclusion and physical activity for all.",
                "cost_range": (800000, 2000000),
                "moral_value": "Fairness/Cheating"
            },
            {
                "name": "Youth Sports Scholarship and Equipment Program",
                "description": "Financial assistance and equipment lending program for low-income families to ensure all children can participate in organized sports regardless of economic circumstances.",
                "cost_range": (100000, 300000),
                "moral_value": "Fairness/Cheating"
            },
            {
                "name": "Community Sports League and Tournament System",
                "description": "Organized recreational sports leagues for all ages and skill levels, promoting community building, physical fitness, and friendly competition across neighborhoods.",
                "cost_range": (75000, 200000),
                "moral_value": "Loyalty/Betrayal"
            },
            {
                "name": "Outdoor Adventure Sports Park",
                "description": "Rock climbing walls, zip lines, obstacle courses, and adventure trails for thrill-seeking activities and team building exercises.",
                "cost_range": (500000, 1200000),
                "moral_value": "Liberty/Oppression"
            },
            {
                "name": "Senior Sports and Wellness Center",
                "description": "Specialized facility for older adults featuring low-impact sports, balance training, social activities, and health monitoring programs.",
                "cost_range": (400000, 1000000),
                "moral_value": "Care/Harm"
            }
        ],
        
        "public space": [
            {
                "name": "Public Wi-Fi Network in Parks and Plazas",
                "description": "Free internet access in all public spaces to bridge the digital divide, support remote work and learning, and enhance the usability of public areas.",
                "cost_range": (200000, 500000),
                "moral_value": "Fairness/Cheating"
            },
            {
                "name": "Community Gathering Spaces and Plazas",
                "description": "Design and construction of welcoming public squares with seating, shade, and programming to encourage community interaction and civic engagement.",
                "cost_range": (300000, 800000),
                "moral_value": "Loyalty/Betrayal"
            },
            {
                "name": "Public Art and Cultural Installations",
                "description": "Temporary and permanent art installations in public spaces celebrating local culture, history, and creativity while making public areas more engaging and beautiful.",
                "cost_range": (50000, 200000),
                "moral_value": "Sanctity/Degradation"
            },
            {
                "name": "Outdoor Performance Venue",
                "description": "Amphitheater and stage facilities for concerts, theater performances, movie screenings, and community events in public parks.",
                "cost_range": (400000, 1000000),
                "moral_value": "Loyalty/Betrayal"
            },
            {
                "name": "Public Market and Food Court",
                "description": "Open-air market space for local vendors, food trucks, and community events, promoting local business and social interaction.",
                "cost_range": (300000, 700000),
                "moral_value": "Loyalty/Betrayal"
            }
        ],
        
        "public transit and roads": [
            {
                "name": "Bicycle Infrastructure Master Plan Implementation",
                "description": "Comprehensive network of protected bike lanes, bike parking facilities, and cyclist amenities to promote sustainable transportation and improve safety for all road users.",
                "cost_range": (1000000, 3000000),
                "moral_value": "Sanctity/Degradation"
            },
            {
                "name": "Accessible Public Transit Improvements",
                "description": "Upgrades to bus stops, train stations, and vehicles to ensure full accessibility for people with disabilities, including ramps, audio announcements, and priority seating.",
                "cost_range": (500000, 1500000),
                "moral_value": "Fairness/Cheating"
            },
            {
                "name": "Smart Parking and Traffic Flow System",
                "description": "Intelligent parking management system with real-time availability, dynamic pricing, and traffic flow optimization to reduce congestion and improve urban mobility.",
                "cost_range": (400000, 1000000),
                "moral_value": "Authority/Subversion"
            },
            {
                "name": "Micro-Mobility Hub Network",
                "description": "Centralized locations for bike sharing, electric scooters, and other micro-mobility options with charging stations and maintenance facilities.",
                "cost_range": (200000, 500000),
                "moral_value": "Sanctity/Degradation"
            },
            {
                "name": "Emergency Vehicle Priority System",
                "description": "Smart traffic signal technology that automatically gives priority to emergency vehicles, reducing response times and improving public safety.",
                "cost_range": (300000, 700000),
                "moral_value": "Care/Harm"
            }
        ],
        
        "welfare": [
            {
                "name": "Mental Health Crisis Intervention Team",
                "description": "Specialized response unit for mental health emergencies, providing immediate support, de-escalation, and connection to appropriate services instead of law enforcement.",
                "cost_range": (300000, 800000),
                "moral_value": "Care/Harm"
            },
            {
                "name": "Housing First Program for Homeless Residents",
                "description": "Immediate housing placement for homeless individuals and families, followed by comprehensive support services to address underlying causes of homelessness.",
                "cost_range": (1000000, 3000000),
                "moral_value": "Care/Harm"
            },
            {
                "name": "Community Food Security Initiative",
                "description": "Comprehensive program including food banks, community gardens, nutrition education, and food assistance to ensure all residents have access to healthy, affordable food.",
                "cost_range": (200000, 600000),
                "moral_value": "Care/Harm"
            },
            {
                "name": "Job Training and Placement Center",
                "description": "Comprehensive employment services including skills assessment, training programs, job placement assistance, and ongoing support for career advancement.",
                "cost_range": (400000, 1000000),
                "moral_value": "Fairness/Cheating"
            },
            {
                "name": "Family Support and Childcare Network",
                "description": "Coordinated system of affordable childcare, family counseling, parenting education, and support services for families in crisis.",
                "cost_range": (300000, 800000),
                "moral_value": "Care/Harm"
            }
        ],
        
        "environmental protection": [
            {
                "name": "Urban Wildlife Habitat Conservation Program",
                "description": "Protection and enhancement of urban wildlife habitats, including bird nesting sites, bat roosts, and pollinator gardens to support biodiversity in the city.",
                "cost_range": (150000, 400000),
                "moral_value": "Sanctity/Degradation"
            },
            {
                "name": "Zero Waste Community Initiative",
                "description": "Comprehensive waste reduction program including composting facilities, recycling education, reusable item lending libraries, and community waste audits.",
                "cost_range": (250000, 600000),
                "moral_value": "Sanctity/Degradation"
            },
            {
                "name": "Climate Resilience and Adaptation Planning",
                "description": "Assessment and implementation of climate adaptation measures including flood protection, heat mitigation, and infrastructure upgrades for extreme weather events.",
                "cost_range": (500000, 1500000),
                "moral_value": "Care/Harm"
            },
            {
                "name": "Urban Bee and Pollinator Initiative",
                "description": "Installation of beehives, pollinator gardens, and educational programs to support declining pollinator populations and promote urban biodiversity.",
                "cost_range": (100000, 300000),
                "moral_value": "Sanctity/Degradation"
            },
            {
                "name": "Community Solar Energy Program",
                "description": "Shared solar installations on public buildings and community spaces, providing renewable energy access to residents who cannot install panels on their own properties.",
                "cost_range": (400000, 1000000),
                "moral_value": "Sanctity/Degradation"
            }
        ],
        
        "health": [
            {
                "name": "Community Health Worker Program",
                "description": "Trained community members providing health education, preventive care coordination, and cultural bridge services to improve health outcomes in underserved communities.",
                "cost_range": (200000, 500000),
                "moral_value": "Care/Harm"
            },
            {
                "name": "Mobile Health Clinic for Underserved Areas",
                "description": "Traveling medical facility providing basic healthcare, screenings, vaccinations, and health education to residents in areas with limited access to medical services.",
                "cost_range": (400000, 1000000),
                "moral_value": "Fairness/Cheating"
            },
            {
                "name": "Public Health Emergency Preparedness System",
                "description": "Comprehensive emergency response infrastructure including medical supplies, trained personnel, and coordination systems for public health crises and natural disasters.",
                "cost_range": (300000, 800000),
                "moral_value": "Care/Harm"
            },
            {
                "name": "Mental Health Awareness and Prevention",
                "description": "Community-wide mental health education, early intervention programs, and stigma reduction initiatives to improve overall community mental health.",
                "cost_range": (150000, 400000),
                "moral_value": "Care/Harm"
            },
            {
                "name": "Senior Health and Wellness Program",
                "description": "Comprehensive health services for elderly residents including preventive care, chronic disease management, and social support programs.",
                "cost_range": (250000, 600000),
                "moral_value": "Care/Harm"
            }
        ],
        
        "culture": [
            {
                "name": "Indigenous Cultural Heritage Preservation",
                "description": "Documentation, preservation, and celebration of indigenous cultural practices, languages, and traditions through educational programs, cultural events, and community partnerships.",
                "cost_range": (150000, 400000),
                "moral_value": "Sanctity/Degradation"
            },
            {
                "name": "Community Theater and Performance Arts Center",
                "description": "Multi-purpose performing arts facility hosting community theater productions, music performances, dance classes, and cultural events celebrating local talent and diversity.",
                "cost_range": (600000, 1500000),
                "moral_value": "Loyalty/Betrayal"
            },
            {
                "name": "Digital Storytelling and Media Lab",
                "description": "Community media center providing training and equipment for residents to create digital content, podcasts, videos, and other media sharing local stories and perspectives.",
                "cost_range": (200000, 500000),
                "moral_value": "Liberty/Oppression"
            },
            {
                "name": "Cultural Exchange and Language Program",
                "description": "Language learning initiatives, cultural exchange events, and international partnership programs to promote cross-cultural understanding and global citizenship.",
                "cost_range": (100000, 300000),
                "moral_value": "Loyalty/Betrayal"
            },
            {
                "name": "Community Music and Arts Education",
                "description": "Comprehensive arts education program providing music lessons, art classes, and creative workshops for residents of all ages and skill levels.",
                "cost_range": (200000, 500000),
                "moral_value": "Sanctity/Degradation"
            }
        ]
    }
    
    return project_templates

def generate_additional_unique_projects():
    """Generate additional unique projects that are significantly different from templates."""
    
    additional_projects = [
        {
            "name": "Community Time Bank System",
            "description": "Skill and service exchange platform where residents can trade expertise, time, and services without money, building community connections and mutual support networks.",
            "cost_range": (50000, 150000),
            "moral_value": "Loyalty/Betrayal",
            "category": "Culture & community"
        },
        {
            "name": "Urban Astronomy and Science Center",
            "description": "Public observatory with telescopes, planetarium shows, and science exhibits to promote scientific literacy and wonder in urban environments.",
            "cost_range": (300000, 800000),
            "moral_value": "Sanctity/Degradation",
            "category": "Education"
        },
        {
            "name": "Community Emergency Food Forest",
            "description": "Perennial food-producing landscape designed for resilience and abundance, providing emergency food sources and teaching sustainable agriculture practices.",
            "cost_range": (100000, 300000),
            "moral_value": "Care/Harm",
            "category": "urban greenery"
        },
        {
            "name": "Intergenerational Technology Mentorship",
            "description": "Program pairing tech-savvy youth with seniors for mutual learning, where young people teach digital skills while seniors share life wisdom and experience.",
            "cost_range": (75000, 200000),
            "moral_value": "Loyalty/Betrayal",
            "category": "Education"
        },
        {
            "name": "Community Currency and Local Economy Initiative",
            "description": "Local complementary currency system to support neighborhood businesses, encourage local spending, and strengthen community economic resilience.",
            "cost_range": (100000, 250000),
            "moral_value": "Loyalty/Betrayal",
            "category": "Culture & community"
        },
        {
            "name": "Urban Wildlife Rehabilitation Center",
            "description": "Facility for treating injured urban wildlife, educating the public about coexistence, and releasing rehabilitated animals back into their natural habitats.",
            "cost_range": (200000, 500000),
            "moral_value": "Care/Harm",
            "category": "environmental protection"
        },
        {
            "name": "Community Energy Democracy Project",
            "description": "Cooperative renewable energy initiative allowing residents to collectively own and benefit from local solar and wind installations.",
            "cost_range": (500000, 1200000),
            "moral_value": "Liberty/Oppression",
            "category": "environmental protection"
        },
        {
            "name": "Neighborhood Safety and Watch Program",
            "description": "Organized community safety initiative with training, communication systems, and neighborhood patrols to reduce crime and increase community cohesion.",
            "cost_range": (50000, 150000),
            "moral_value": "Loyalty/Betrayal",
            "category": "public space"
        },
        {
            "name": "Community Legal Aid and Advocacy Center",
            "description": "Free legal services, tenant rights education, and advocacy programs to ensure equal access to justice for all community members.",
            "cost_range": (300000, 700000),
            "moral_value": "Fairness/Cheating",
            "category": "welfare"
        },
        {
            "name": "Urban Foraging and Wild Food Education",
            "description": "Program teaching safe identification and harvesting of wild edible plants, mushrooms, and other urban food sources while promoting sustainable foraging practices.",
            "cost_range": (75000, 200000),
            "moral_value": "Sanctity/Degradation",
            "category": "environmental protection"
        }
    ]
    
    return additional_projects

def generate_synthetic_projects(num_projects: int = 1000) -> List[Dict]:
    """Generate diverse synthetic projects with detailed descriptions."""
    
    project_templates = generate_specific_project_templates()
    additional_projects = generate_additional_unique_projects()
    projects = []
    
    # First, add all template projects (one of each type)
    for category, templates in project_templates.items():
        for template in templates:
            # Generate realistic coordinates (US-like coordinates)
            latitude = random.uniform(30.0, 50.0)
            longitude = random.uniform(-120.0, -70.0)
            
            # Generate realistic vote counts
            votes = random.randint(1000, 50000)
            
            # Random selection status
            selected = random.choice([0, 1])
            
            # Generate target demographics (1-3 targets)
            num_targets = random.randint(1, 3)
            targets = random.sample(TARGET_DEMOGRAPHICS, num_targets)
            target_str = ",".join(targets)
            
            project = {
                'project_id': f"synthetic_{category.replace(' ', '_')}_{template['name'].replace(' ', '_').replace('&', 'and')}",
                'category': category,
                'cost': random.randint(template['cost_range'][0], template['cost_range'][1]),
                'latitude': round(latitude, 6),
                'longitude': round(longitude, 6),
                'name': template['name'],
                'description': template['description'],
                'selected': selected,
                'target': target_str,
                'votes': votes,
                'moral_value': template['moral_value']
            }
            
            projects.append(project)
    
    # Add additional unique projects
    for i, template in enumerate(additional_projects):
        latitude = random.uniform(30.0, 50.0)
        longitude = random.uniform(-120.0, -70.0)
        votes = random.randint(1000, 50000)
        selected = random.choice([0, 1])
        
        num_targets = random.randint(1, 3)
        targets = random.sample(TARGET_DEMOGRAPHICS, num_targets)
        target_str = ",".join(targets)
        
        project = {
            'project_id': f"synthetic_{template['category'].replace(' ', '_')}_additional_{i}",
            'category': template['category'],
            'cost': random.randint(template['cost_range'][0], template['cost_range'][1]),
            'latitude': round(latitude, 6),
            'longitude': round(longitude, 6),
            'name': template['name'],
            'description': template['description'],
            'selected': selected,
            'target': target_str,
            'votes': votes,
            'moral_value': template['moral_value']
        }
        
        projects.append(project)
    
    # If we still need more projects, create variations of existing ones with different focuses
    remaining = num_projects - len(projects)
    if remaining > 0:
        # Create variations focusing on different aspects or target groups
        for i in range(remaining):
            # Pick a random existing project to vary
            base_project = random.choice(projects)
            
            # Create variation by focusing on different aspects
            variations = [
                "This initiative specifically targets underserved communities.",
                "The program emphasizes environmental sustainability and green practices.",
                "This project focuses on building intergenerational connections.",
                "The initiative prioritizes accessibility and universal design.",
                "This program emphasizes community engagement and participation.",
                "The project focuses on long-term impact and legacy building.",
                "This initiative emphasizes innovation and cutting-edge approaches.",
                "The program focuses on cultural diversity and inclusion."
            ]
            
            variation = random.choice(variations)
            new_description = base_project['description'] + " " + variation
            
            # Slightly modify the name to indicate it's a variation
            variation_suffixes = ["Enhanced", "Expanded", "Focused", "Specialized", "Community-Driven"]
            new_name = f"{base_project['name']} - {random.choice(variation_suffixes)}"
            
            # Generate new coordinates
            latitude = random.uniform(30.0, 50.0)
            longitude = random.uniform(-120.0, -70.0)
            
            project = {
                'project_id': f"synthetic_{base_project['category'].replace(' ', '_')}_variation_{i}",
                'category': base_project['category'],
                'cost': int(base_project['cost'] * random.uniform(0.8, 1.2)),  # Vary cost by ±20%
                'latitude': round(latitude, 6),
                'longitude': round(longitude, 6),
                'name': new_name,
                'description': new_description,
                'selected': random.choice([0, 1]),
                'target': base_project['target'],
                'votes': random.randint(1000, 50000),
                'moral_value': base_project['moral_value']
            }
            
            projects.append(project)
    
    return projects

# Voting data generation functions
def generate_realistic_moral_value_distribution():
    """Generate realistic moral value distributions based on research."""
    
    # Base distributions (approximated from research)
    base_distributions = {
        "Care/Harm": 0.35,        # Most common
        "Fairness/Cheating": 0.25, # Second most common
        "Loyalty/Betrayal": 0.15,  # Moderate
        "Authority/Subversion": 0.10, # Less common
        "Sanctity/Degradation": 0.10, # Less common
        "Liberty/Oppression": 0.05   # Least common
    }
    
    return base_distributions

def generate_voter_demographics():
    """Generate realistic voter demographics."""
    
    # Age distribution (approximated from voting data)
    age_weights = [0.15, 0.20, 0.18, 0.16, 0.15, 0.10, 0.06]
    age_group = random.choices(AGE_GROUPS, weights=age_weights)[0]
    age = random.randint(age_group[0], age_group[1])
    
    # Gender (slightly more female voters)
    gender = random.choices(GENDERS, weights=[0.45, 0.55])[0]
    
    # Education (higher education correlates with voting)
    education_weights = [0.20, 0.25, 0.35, 0.20]
    education = random.choices(EDUCATION_LEVELS, weights=education_weights)[0]
    
    # Income (middle class overrepresented in voting)
    income_weights = [0.15, 0.25, 0.30, 0.20, 0.10]
    income = random.choices(INCOME_LEVELS, weights=income_weights)[0]
    
    # Political orientation (bell curve around moderate)
    political_weights = [0.10, 0.25, 0.30, 0.25, 0.10]
    political = random.choices(POLITICAL_ORIENTATIONS, weights=political_weights)[0]
    
    return {
        'age': age,
        'sex': gender,
        'education': education,
        'income': income,
        'political_orientation': political
    }

def generate_moral_value_preferences():
    """Generate moral value preferences with realistic correlations."""
    
    base_dist = generate_realistic_moral_value_distribution()
    
    # Generate primary moral value
    primary_moral = random.choices(list(base_dist.keys()), weights=list(base_dist.values()))[0]
    
    # Generate secondary moral values (correlated with primary)
    moral_correlations = {
        "Care/Harm": ["Fairness/Cheating", "Sanctity/Degradation"],
        "Fairness/Cheating": ["Care/Harm", "Liberty/Oppression"],
        "Loyalty/Betrayal": ["Authority/Subversion", "Sanctity/Degradation"],
        "Authority/Subversion": ["Loyalty/Betrayal", "Sanctity/Degradation"],
        "Sanctity/Degradation": ["Care/Harm", "Loyalty/Betrayal"],
        "Liberty/Oppression": ["Fairness/Cheating", "Care/Harm"]
    }
    
    secondary_candidates = moral_correlations.get(primary_moral, MORAL_FOUNDATIONS)
    secondary_moral = random.choice([mv for mv in secondary_candidates if mv != primary_moral])
    
    # Generate tertiary moral value
    remaining_morals = [mv for mv in MORAL_FOUNDATIONS if mv not in [primary_moral, secondary_moral]]
    tertiary_moral = random.choice(remaining_morals)
    
    return {
        'top_moral_value': primary_moral,
        'secondary_moral_value': secondary_moral,
        'tertiary_moral_value': tertiary_moral
    }

def generate_category_preferences():
    """Generate category preferences based on moral values and demographics."""
    
    # Define category-moral value associations
    category_moral_mapping = {
        "Care/Harm": ["welfare", "health", "education", "public space"],
        "Fairness/Cheating": ["education", "welfare", "public space", "sport"],
        "Loyalty/Betrayal": ["culture", "public space", "sport", "urban greenery"],
        "Authority/Subversion": ["public transit and roads", "public space", "environmental protection"],
        "Sanctity/Degradation": ["environmental protection", "urban greenery", "health", "culture"],
        "Liberty/Oppression": ["public space", "sport", "culture", "education"]
    }
    
    # Get moral values
    moral_values = generate_moral_value_preferences()
    primary_moral = moral_values['top_moral_value']
    
    # Get preferred categories
    preferred_categories = category_moral_mapping.get(primary_moral, ["public space", "culture"])
    
    # Select top 3 categories
    top_categories = random.sample(preferred_categories, min(3, len(preferred_categories)))
    
    # Fill remaining slots with random categories
    all_categories = [
        "Culture & community", "Education", "Environment, public health & safety",
        "Facilities, parks & recreation", "Streets, Sidewalks & Transit",
        "urban greenery", "sport", "public space", "public transit and roads",
        "welfare", "environmental protection", "health", "culture"
    ]
    
    remaining_categories = [cat for cat in all_categories if cat not in top_categories]
    additional_categories = random.sample(remaining_categories, 3 - len(top_categories))
    
    final_categories = top_categories + additional_categories
    
    return {
        'top_category_1': final_categories[0] if len(final_categories) > 0 else "public space",
        'top_category_2': final_categories[1] if len(final_categories) > 1 else "culture",
        'top_category_3': final_categories[2] if len(final_categories) > 2 else "education"
    }

def generate_project_votes(num_projects: int = 5):
    """Generate realistic project voting patterns."""
    
    # Generate 3-8 project votes per voter
    num_votes = random.randint(3, 8)
    
    # Select random project IDs (assuming projects exist)
    project_ids = random.sample(range(1, num_projects + 1), num_votes)
    
    # Convert to comma-separated string
    vote_string = ",".join(map(str, project_ids))
    
    return vote_string

def generate_synthetic_voters(num_voters: int = 2000) -> List[Dict]:
    """Generate diverse synthetic voters with realistic characteristics."""
    
    voters = []
    
    for i in range(num_voters):
        # Generate demographics
        demographics = generate_voter_demographics()
        
        # Generate moral values
        moral_values = generate_moral_value_preferences()
        
        # Generate category preferences
        categories = generate_category_preferences()
        
        # Generate project votes
        votes = generate_project_votes(50)  # Assume 50 projects available
        
        # Create voter record
        voter = {
            'voter_id': f"synthetic_voter_{i:04d}",
            'age': demographics['age'],
            'sex': demographics['sex'],
            'education': demographics['education'],
            'income': demographics['income'],
            'political_orientation': demographics['political_orientation'],
            'top_moral_value': moral_values['top_moral_value'],
            'secondary_moral_value': moral_values['secondary_moral_value'],
            'tertiary_moral_value': moral_values['tertiary_moral_value'],
            'top_category_1': categories['top_category_1'],
            'top_category_2': categories['top_category_2'],
            'top_category_3': categories['top_category_3'],
            'vote': votes,
            'source': 'synthetic_data'
        }
        
        voters.append(voter)
    
    return voters

def main():
    """Generate and save comprehensive synthetic data."""
    print("🚀 Generating comprehensive synthetic data...")
    
    # Generate projects
    print("Generating synthetic projects...")
    projects = generate_synthetic_projects(1000)
    
    # Convert to DataFrame
    projects_df = pd.DataFrame(projects)
    
    # Add some additional columns for compatibility
    projects_df['source_files'] = 'synthetic_data'
    projects_df['moral_score_Care/Harm'] = projects_df['moral_value'].apply(lambda x: 0.9 if x == 'Care/Harm' else 0.1)
    projects_df['moral_score_Fairness/Cheating'] = projects_df['moral_value'].apply(lambda x: 0.9 if x == 'Fairness/Cheating' else 0.1)
    projects_df['moral_score_Loyalty/Betrayal'] = projects_df['moral_value'].apply(lambda x: 0.9 if x == 'Loyalty/Betrayal' else 0.1)
    projects_df['moral_score_Authority/Subversion'] = projects_df['moral_value'].apply(lambda x: 0.9 if x == 'Authority/Subversion' else 0.1)
    projects_df['moral_score_Sanctity/Degradation'] = projects_df['moral_value'].apply(lambda x: 0.9 if x == 'Sanctity/Degradation' else 0.1)
    projects_df['moral_score_Liberty/Oppression'] = projects_df['moral_value'].apply(lambda x: 0.9 if x == 'Liberty/Oppression' else 0.1)
    
    # Save projects
    projects_file = "../data/synthetic_projects.csv"
    save_csv_data(projects_df, projects_file)
    
    # Generate voters
    print("Generating synthetic voters...")
    voters = generate_synthetic_voters(2000)
    
    # Convert to DataFrame
    voters_df = pd.DataFrame(voters)
    
    # Save voters
    voters_file = "../data/synthetic_votes.csv"
    save_csv_data(voters_df, voters_file)
    
    # Print summary
    print_separator("SYNTHETIC DATA GENERATION COMPLETE")
    print(f"Generated {len(projects)} diverse synthetic projects")
    print(f"Generated {len(voters)} diverse synthetic voters")
    print(f"Categories represented: {projects_df['category'].nunique()}")
    print(f"Projects saved to: {projects_file}")
    print(f"Voters saved to: {voters_file}")
    
    # Show moral value distribution
    print("\n🧠 Moral values distribution:")
    print(projects_df['moral_value'].value_counts())
    
    # Show cost range
    print(f"\n💰 Cost range: ${projects_df['cost'].min():,} - ${projects_df['cost'].max():,}")
    
    # Show sample projects
    print(f"\n📋 Sample projects:")
    for i, (_, project) in enumerate(projects_df.head(5).iterrows(), 1):
        print(f"\n{i}. {project['name']}")
        print(f"   Category: {project['category']}")
        print(f"   Cost: ${project['cost']:,}")
        print(f"   Moral Value: {project['moral_value']}")
        print(f"   Description: {project['description'][:100]}...")

if __name__ == "__main__":
    main()
