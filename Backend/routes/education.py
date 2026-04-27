from flask import Blueprint, jsonify

education_bp = Blueprint('education', __name__, url_prefix='/api/education')


@education_bp.route('/soil-types', methods=['GET'])
def get_soil_types():
    content = {
        "loamy": {
            "title": "Loamy Soil",
            "description": "The gold standard for agriculture. Balanced mix of sand, silt, and clay.",
            "composition": {"Sand": 40, "Silt": 40, "Clay": 20},
            "best_crops": ["Wheat", "Corn", "Tomatoes", "Peppers", "Most vegetables"],
            "characteristics": ["Dark brown", "Crumbly texture", "Holds moisture", "Rich in nutrients", "Easy to work"]
        },
        "sandy": {
            "title": "Sandy Soil",
            "description": "Large particles, drains quickly, low nutrient retention.",
            "composition": {"Sand": 70, "Silt": 15, "Clay": 15},
            "best_crops": ["Carrots", "Potatoes", "Lettuce", "Strawberries", "Herbs"],
            "characteristics": ["Light brown", "Gritty texture", "Fast draining", "Warms quickly", "Low nutrients"]
        },
        "clay": {
            "title": "Clay Soil",
            "description": "Fine particles, retains water and nutrients but poor drainage.",
            "composition": {"Sand": 20, "Silt": 20, "Clay": 60},
            "best_crops": ["Rice", "Wheat", "Broccoli", "Cabbage", "Beans"],
            "characteristics": ["Red/brown/grey", "Sticky when wet", "Hard when dry", "High nutrients", "Poor drainage"]
        },
        "silty": {
            "title": "Silty Soil",
            "description": "Medium particles, fertile but prone to compaction.",
            "composition": {"Sand": 20, "Silt": 60, "Clay": 20},
            "best_crops": ["Most vegetables", "Grasses", "Shrubs"],
            "characteristics": ["Dark brown", "Silky texture", "Holds moisture", "Fertile", "Erosion prone"]
        },
        "peaty": {
            "title": "Peaty Soil",
            "description": "Rich in organic matter, very acidic, found in marshy areas.",
            "composition": {"Organic Matter": 70, "Mineral": 30},
            "best_crops": ["Blueberries", "Potatoes", "Heather"],
            "characteristics": ["Very dark/black", "Spongy", "Highly acidic", "Waterlogged", "Organic rich"]
        },
        "chalky": {
            "title": "Chalky Soil",
            "description": "Alkaline, stony, overlies limestone bedrock.",
            "composition": {"Calcium Carbonate": 40, "Sand": 30, "Clay": 30},
            "best_crops": ["Lavender", "Spinach", "Beets", "Sweet corn"],
            "characteristics": ["Pale/white", "Stony", "Very alkaline", "Free-draining", "Causes yellowing"]
        }
    }
    return jsonify(content)


@education_bp.route('/fertilizer-guide', methods=['GET'])
def get_fertilizer_guide():
    guide = {
        "npk_explained": {
            "title": "Understanding N-P-K Ratios",
            "content": "Three numbers on fertilizer bags represent Nitrogen, Phosphorus, and Potassium.",
            "details": {
                "nitrogen": {"symbol": "N", "role": "Leaf and stem growth", "deficiency_signs": "Yellow leaves", "excess_signs": "Too much foliage"},
                "phosphorus": {"symbol": "P", "role": "Root and flower development", "deficiency_signs": "Purple leaves", "excess_signs": "Blocks zinc/iron"},
                "potassium": {"symbol": "K", "role": "Overall health and disease resistance", "deficiency_signs": "Brown leaf edges", "excess_signs": "Blocks calcium"}
            }
        },
        "application_methods": {
            "title": "How to Apply Fertilizers",
            "methods": [
                {"name": "Broadcasting", "description": "Spread evenly across field", "best_for": "Pre-planting"},
                {"name": "Banding", "description": "Place near seed rows", "best_for": "Row crops"},
                {"name": "Side-dressing", "description": "Apply alongside plants", "best_for": "Mid-season boost"},
                {"name": "Foliar Spray", "description": "Spray on leaves", "best_for": "Quick fix"}
            ]
        },
        "organic_vs_synthetic": {
            "title": "Organic vs Synthetic",
            "organic": {"pros": ["Improves soil", "Slow release", "Eco-friendly"], "cons": ["Slower results", "Variable nutrients"]},
            "synthetic": {"pros": ["Fast acting", "Precise nutrients", "Affordable"], "cons": ["Can burn plants", "Degrades soil"]}
        }
    }
    return jsonify(guide)
