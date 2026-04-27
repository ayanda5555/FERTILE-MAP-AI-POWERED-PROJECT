FERTILIZER_DATABASE = {
    "loamy": {
        "status": "Excellent soil — ideal for most crops",
        "general_fertilizers": [
            {
                "name": "Balanced NPK 10-10-10",
                "purpose": "General maintenance",
                "application": "200-300 kg/hectare",
                "frequency": "Once per growing season"
            },
            {
                "name": "Compost / Organic Matter",
                "purpose": "Maintain soil structure",
                "application": "2-3 tonnes/hectare",
                "frequency": "Annually"
            }
        ],
        "crop_specific": {
            "wheat": [
                {"name": "Urea (46-0-0)", "dose": "130 kg/ha", "timing": "Split: sowing + 30 days"},
                {"name": "DAP (18-46-0)", "dose": "100 kg/ha", "timing": "At sowing"}
            ],
            "rice": [
                {"name": "Urea", "dose": "150 kg/ha", "timing": "3 splits"},
                {"name": "MOP (0-0-60)", "dose": "50 kg/ha", "timing": "At transplanting"}
            ],
            "vegetables": [
                {"name": "NPK 19-19-19", "dose": "150 kg/ha", "timing": "At transplanting"},
                {"name": "Vermicompost", "dose": "5 tonnes/ha", "timing": "Before planting"}
            ],
            "corn": [
                {"name": "Urea", "dose": "200 kg/ha", "timing": "Split application"},
                {"name": "SSP", "dose": "150 kg/ha", "timing": "At sowing"}
            ]
        },
        "organic_alternatives": [
            "Cow manure compost",
            "Green manure (legume cover crops)",
            "Bone meal for phosphorus",
            "Wood ash for potassium"
        ],
        "improvement_tips": [
            "Maintain organic matter with annual composting",
            "Practice crop rotation",
            "Avoid over-tilling to preserve structure",
            "Test soil pH annually"
        ]
    },
    "sandy": {
        "status": "Needs improvement — nutrients leach quickly",
        "general_fertilizers": [
            {
                "name": "NPK 20-10-10 (Nitrogen-heavy)",
                "purpose": "Compensate for nutrient leaching",
                "application": "250-350 kg/hectare in splits",
                "frequency": "Multiple small doses"
            },
            {
                "name": "Slow-release fertilizer",
                "purpose": "Sustained nutrient supply",
                "application": "As per product label",
                "frequency": "Every 2-3 months"
            }
        ],
        "crop_specific": {
            "wheat": [
                {"name": "Urea", "dose": "150 kg/ha in 3 splits", "timing": "Every 3 weeks"},
                {"name": "MOP", "dose": "80 kg/ha", "timing": "At sowing"}
            ],
            "vegetables": [
                {"name": "Liquid fertilizer", "dose": "Weekly application", "timing": "Throughout growth"},
                {"name": "NPK 15-15-15", "dose": "200 kg/ha", "timing": "Frequent small doses"}
            ],
            "corn": [
                {"name": "CAN (27-0-0)", "dose": "200 kg/ha", "timing": "Split 4 times"},
                {"name": "DAP", "dose": "100 kg/ha", "timing": "At sowing"}
            ]
        },
        "organic_alternatives": [
            "Heavy composting (5+ tonnes/ha)",
            "Mulching to retain moisture",
            "Peat moss or coconut coir",
            "Clay amendment to improve retention"
        ],
        "improvement_tips": [
            "Add organic matter heavily to improve water retention",
            "Use mulching to reduce evaporation",
            "Apply fertilizers in small, frequent doses",
            "Consider drip irrigation",
            "Plant cover crops to build soil structure"
        ]
    },
    "clay": {
        "status": "Nutrient-rich but needs structural improvement",
        "general_fertilizers": [
            {
                "name": "Gypsum (calcium sulfate)",
                "purpose": "Break up clay, improve drainage",
                "application": "500-1000 kg/hectare",
                "frequency": "Annually"
            },
            {
                "name": "NPK 10-20-10",
                "purpose": "Balanced nutrition",
                "application": "200 kg/hectare",
                "frequency": "Once per season"
            }
        ],
        "crop_specific": {
            "wheat": [
                {"name": "DAP", "dose": "100 kg/ha", "timing": "At sowing"},
                {"name": "Potash", "dose": "60 kg/ha", "timing": "At sowing"}
            ],
            "rice": [
                {"name": "Urea", "dose": "120 kg/ha", "timing": "Split 2-3 times"},
                {"name": "Zinc sulphate", "dose": "25 kg/ha", "timing": "At transplanting"}
            ],
            "vegetables": [
                {"name": "NPK 15-15-15", "dose": "150 kg/ha", "timing": "At planting"},
                {"name": "Compost", "dose": "5 tonnes/ha", "timing": "Before planting"}
            ]
        },
        "organic_alternatives": [
            "Compost to improve aeration",
            "Straw mulch",
            "Cover crops (deep-rooted)",
            "Green manure"
        ],
        "improvement_tips": [
            "Add gypsum or sand to improve drainage",
            "Avoid working soil when wet",
            "Use raised beds for vegetables",
            "Add organic matter to improve structure",
            "Consider deep-rooted cover crops"
        ]
    },
    "silty": {
        "status": "Fertile but prone to compaction",
        "general_fertilizers": [
            {
                "name": "NPK 10-10-10",
                "purpose": "Balanced nutrition",
                "application": "200 kg/hectare",
                "frequency": "Once per season"
            }
        ],
        "crop_specific": {
            "wheat": [
                {"name": "Urea", "dose": "130 kg/ha", "timing": "Split application"},
                {"name": "DAP", "dose": "100 kg/ha", "timing": "At sowing"}
            ],
            "vegetables": [
                {"name": "NPK 19-19-19", "dose": "150 kg/ha", "timing": "At planting"},
                {"name": "Compost", "dose": "4 tonnes/ha", "timing": "Before planting"}
            ]
        },
        "organic_alternatives": [
            "Compost", "Mulching", "Cover crops", "Green manure"
        ],
        "improvement_tips": [
            "Avoid compaction — minimize foot traffic",
            "Add organic matter regularly",
            "Use mulch to prevent erosion",
            "Practice no-till or minimal tillage"
        ]
    },
    "peaty": {
        "status": "Very acidic — needs pH correction",
        "general_fertilizers": [
            {
                "name": "Agricultural lime",
                "purpose": "Raise pH from acidic levels",
                "application": "2-4 tonnes/hectare",
                "frequency": "As needed based on pH test"
            },
            {
                "name": "NPK 5-10-10",
                "purpose": "Phosphorus and potassium boost",
                "application": "200 kg/hectare",
                "frequency": "Once per season"
            }
        ],
        "crop_specific": {
            "vegetables": [
                {"name": "Lime + NPK", "dose": "As per pH test", "timing": "Before planting"},
                {"name": "Bone meal", "dose": "200 kg/ha", "timing": "At planting"}
            ]
        },
        "organic_alternatives": [
            "Limestone to raise pH",
            "Wood ash",
            "Bone meal",
            "Rock phosphate"
        ],
        "improvement_tips": [
            "Test and correct pH — aim for 6.0-6.5",
            "Improve drainage with sand/gravel channels",
            "Great for blueberries and acid-loving plants as-is",
            "Add lime gradually over seasons"
        ]
    },
    "chalky": {
        "status": "Alkaline — may cause nutrient lockout",
        "general_fertilizers": [
            {
                "name": "Sulfur / Iron sulfate",
                "purpose": "Lower pH slightly",
                "application": "100-200 kg/hectare",
                "frequency": "Annually"
            },
            {
                "name": "NPK with micronutrients",
                "purpose": "Prevent iron/manganese deficiency",
                "application": "200 kg/hectare",
                "frequency": "Once per season"
            }
        ],
        "crop_specific": {
            "vegetables": [
                {"name": "Chelated iron", "dose": "As per label", "timing": "When deficiency appears"},
                {"name": "Acidifying fertilizer", "dose": "As per label", "timing": "Regular application"}
            ]
        },
        "organic_alternatives": [
            "Pine needle mulch (acidifying)",
            "Composted oak leaves",
            "Sulfur chips",
            "Acidic compost"
        ],
        "improvement_tips": [
            "Add organic matter to buffer alkalinity",
            "Use chelated micronutrients",
            "Choose alkaline-tolerant crops",
            "Avoid phosphorus excess — causes lockout",
            "Grow lavender, spinach, beets (thrive in alkaline)"
        ]
    }
}


def get_recommendations(soil_type, crop_type="general"):
    soil_data = FERTILIZER_DATABASE.get(soil_type, {})

    result = {
        "status": soil_data.get("status", "Unknown"),
        "general_fertilizers": soil_data.get("general_fertilizers", []),
        "organic_alternatives": soil_data.get("organic_alternatives", []),
        "improvement_tips": soil_data.get("improvement_tips", []),
    }

    # Add crop-specific if requested
    if crop_type != "general":
        crop_data = soil_data.get("crop_specific", {}).get(crop_type, [])
        result["crop_specific_fertilizers"] = crop_data
        result["selected_crop"] = crop_type
    else:
        result["available_crops"] = list(soil_data.get("crop_specific", {}).keys())

    return result
