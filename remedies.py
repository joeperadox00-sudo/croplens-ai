"""
CROPLENS AI - Agronomic Disease & Pest Management Remedy Database
Pure English reference database (Multilingual translations handled via Voice Assistant).
Includes 8 diagnostic classes: 5 foliar diseases/stresses + 3 major pest infestations.
"""

from typing import Dict, Any

DISEASE_DATABASE: Dict[str, Dict[str, Any]] = {
    "Healthy Leaf": {
        "scientific_name": "Optimal Physiological Status",
        "category": "Healthy Foliage",
        "stress_level": "None (0-10%)",
        "severity_color": "#22c55e",
        "etl_threshold": "Sub-threshold: Maintain standard prophylactic monitoring.",
        "symptoms": [
            "Uniform vibrant green pigmentation across leaf lamina.",
            "Normal turgidity with no wilting or chlorotic discoloration.",
            "Intact cuticle layer with zero visible fungal lesions, bites, or necrotic spots."
        ],
        "causes": [
            "Balanced nitrogen-phosphorus-potassium (NPK) nutrition.",
            "Adequate soil moisture and proper root aeration.",
            "Absence of bacterial, fungal, or insect pest pressure."
        ],
        "organic_treatment": [
            "Continue standard biological maintenance with seaweed extract or bio-stimulants.",
            "Apply prophylactic neem oil spray (2 ml/L) once every 14 days as preventative shield."
        ],
        "chemical_control": [
            "No chemical fungicides or insecticides needed for healthy foliage.",
            "Maintain routine soil nutrient testing schedule every cropping cycle."
        ],
        "preventive_tips": [
            "Maintain optimal crop spacing to ensure good canopy airflow and sunlight penetration.",
            "Use drip irrigation to avoid wet leaves and prevent fungal spore development."
        ]
    },
    "Early Blight": {
        "scientific_name": "Alternaria solani / Alternaria spp.",
        "category": "Fungal Blight Disease",
        "stress_level": "Moderate to High (45-75%)",
        "severity_color": "#f59e0b",
        "etl_threshold": "ETL Trigger: >= 5% leaf area affected on lower canopy or 1 lesion per 3 leaves.",
        "symptoms": [
            "Small brown-black necrotic spots developing on older foliage.",
            "Characteristic concentric rings creating a 'target board' appearance.",
            "Surrounding chlorotic yellow halos causing premature leaf drop."
        ],
        "causes": [
            "Warm temperatures (24-29°C) combined with high humidity or frequent rains.",
            "Splashing water transferring fungal spores from infected soil debris onto leaves."
        ],
        "organic_treatment": [
            "Spray Trichoderma viride or Bacillus subtilis bio-fungicide @ 5g/L.",
            "Apply 1% Bordeaux mixture or copper soap to inhibit fungal spore germination.",
            "Prune bottom 12 inches of infected foliage and destroy off-field."
        ],
        "chemical_control": [
            "Mancozeb 75% WP @ 2.5g/L or Chlorothalonil 75% WP @ 2g/L at first disease onset.",
            "Systemic action for severe spread: Difenoconazole 25% EC @ 1 ml/L or Azoxystrobin 23% SC @ 1 ml/L."
        ],
        "preventive_tips": [
            "Practice 3-year crop rotation with non-solanaceous crops like legumes or cereals.",
            "Apply straw or plastic mulch to prevent soil-borne spores splashing onto lower leaves."
        ]
    },
    "Common Rust": {
        "scientific_name": "Puccinia sorghi / Puccinia spp.",
        "category": "Fungal Rust Disease",
        "stress_level": "High (60-85%)",
        "severity_color": "#ea580c",
        "etl_threshold": "ETL Trigger: 3-5 pustules per leaf across 10% of scouted field plants.",
        "symptoms": [
            "Golden-brown to cinnamon-colored powdery pustules on leaf surfaces.",
            "Pustules rupture epidermal tissue, releasing powdery spores.",
            "Severe cases cause premature leaf drying and reduced photosynthesis."
        ],
        "causes": [
            "Windblown urediniospores travelling long distances in warm, humid weather.",
            "Extended leaf wetness (6-8 hours) at moderate temperatures (16-25°C)."
        ],
        "organic_treatment": [
            "Apply wettable sulfur powder (WP 80%) @ 3g/L during cool morning hours.",
            "Foliar spray with Ampelomyces quisqualis bio-control hyperparasitic fungus."
        ],
        "chemical_control": [
            "Propiconazole 25% EC @ 1 ml/L or Tebuconazole 250 EC @ 1 ml/L on active rust pustules.",
            "Mancozeb 75% WP @ 2.5g/L for broad protective coverage."
        ],
        "preventive_tips": [
            "Plant rust-resistant hybrid seed varieties suited for your region.",
            "Sow early in the cropping season to avoid peak air-borne spore flight periods."
        ]
    },
    "Leaf Spot / Late Blight": {
        "scientific_name": "Phytophthora infestans / Cercospora spp.",
        "category": "Oomycete / Severe Blight",
        "stress_level": "Severe (75-95%)",
        "severity_color": "#dc2626",
        "etl_threshold": "CRITICAL EMERGENCY: 1% foliar lesion area detected under wet forecast conditions.",
        "symptoms": [
            "Water-soaked irregular pale green/brown blotches rapidly expanding.",
            "White fungal mildew growth visible on undersides of leaves in humid mornings.",
            "Rapid collapse and rotting of entire leaf and petiole tissue within 48-72 hours."
        ],
        "causes": [
            "Cool, wet weather with relative humidity above 90% and temperatures 15-20°C.",
            "Prolonged leaf wetness exceeding 10 hours continuously."
        ],
        "organic_treatment": [
            "Spray Copper Hydroxide (Kocide 2000) @ 2g/L at earliest lesion visibility.",
            "Apply Potassium Phosphite foliar spray to activate systemic acquired plant resistance."
        ],
        "chemical_control": [
            "Metalaxyl 8% + Mancozeb 64% WP (Ridomil Gold) @ 2.5g/L.",
            "Dimethomorph 50% WP @ 1g/L combined with Mancozeb for aggressive outbreaks."
        ],
        "preventive_tips": [
            "Use only certified pathogen-free seeds and disease-free nursery stock.",
            "Destroy and deeply bury all volunteer plants and infected crop debris post-harvest."
        ]
    },
    "Nutrient Chlorosis / Water Stress": {
        "scientific_name": "Abiotic Physiological Stress (Chlorophyll Deficit / Moisture Shock)",
        "category": "Abiotic Stress",
        "stress_level": "Early Warning (30-55%)",
        "severity_color": "#eab308",
        "etl_threshold": "Physiological Alert: > 15% canopy leaves showing interveinal yellowing.",
        "symptoms": [
            "Interveinal or uniform leaf yellowing (chlorosis) due to chlorophyll loss.",
            "Leaf margins curling inward or upward with reduced lamina expansion.",
            "Loss of cell turgor and foliage wilting during peak midday sunlight."
        ],
        "causes": [
            "Root zone moisture deficit or high soil salinity impeding nutrient uptake.",
            "Deficiency in bioavailable Nitrogen (N), Iron (Fe), or Magnesium (Mg)."
        ],
        "organic_treatment": [
            "Foliar spray of Panchagavya (3%) or fermented vermicompost tea early in the morning.",
            "Apply chelated iron (Fe-EDTA 12%) @ 1g/L for rapid leaf greening against chlorosis."
        ],
        "chemical_control": [
            "Foliar spray of water-soluble balanced 19-19-19 NPK fertilizer @ 5g/L.",
            "Adjust soil pH using agricultural gypsum or lime based on soil analysis."
        ],
        "preventive_tips": [
            "Incorporate well-decomposed farmyard manure (FYM) to improve moisture retention.",
            "Regulate irrigation scheduling using tensiometers to avoid waterlogging and drought shock."
        ]
    },
    "Aphids / Whiteflies": {
        "scientific_name": "Aphis gossypii / Bemisia tabaci",
        "category": "Sucking Pest Infestation",
        "stress_level": "Moderate to High (40-70%)",
        "severity_color": "#06b6d4",
        "etl_threshold": "ETL Trigger: 5-10 nymphs/adults per leaf on 20% of sampled terminal shoots.",
        "symptoms": [
            "Leaf curling, crinkling, and stunted apical shoot growth from sap drainage.",
            "Shiny sticky honeydew deposits on upper leaf surfaces fostering black sooty mold.",
            "Yellow mosaic chlorotic flecking and transmission of geminivirus pathogens."
        ],
        "causes": [
            "Dry, warm microclimates accelerating rapid parthenogenetic reproduction cycles.",
            "Excessive nitrogenous fertilization producing soft, succulent leaf tissue attractive to pests."
        ],
        "organic_treatment": [
            "Install bright Yellow Sticky Traps @ 15-20 traps per acre at crop canopy height.",
            "Spray Neem Seed Kernel Extract (NSKE 5%) or cold-pressed Azadirachtin 10,000 ppm @ 2 ml/L.",
            "Release biological predators: Ladybird beetles (Coccinella septempunctata) or Chrysoperla carnea larvae @ 50,000/ha."
        ],
        "chemical_control": [
            "Imidacloprid 17.8% SL @ 0.5 ml/L or Thiamethoxam 25% WG @ 0.3g/L for systemic phloem protection.",
            "Acetamiprid 20% SP @ 0.5g/L during high whitefly swarm pressure."
        ],
        "preventive_tips": [
            "Erect yellow sticky cards early in season to detect first winged alate arrivals.",
            "Plant barrier border rows of maize, sorghum, or pearl millet around solanaceous fields."
        ]
    },
    "Leaf Miners / Armyworms": {
        "scientific_name": "Liriomyza trifolii / Spodoptera frugiperda",
        "category": "Chewing / Boring Pest Infestation",
        "stress_level": "High (55-80%)",
        "severity_color": "#a855f7",
        "etl_threshold": "ETL Trigger: 2-3 active serpentine mines per leaf or 1 caterpillar per plant.",
        "symptoms": [
            "Prominent white serpentine, winding tunnels (mines) across the leaf mesophyll.",
            "Irregular ragged holes and defoliation along leaf margins from chewing larvae.",
            "Dark frass (insect fecal pellets) visible inside leaf mines or leaf axils."
        ],
        "causes": [
            "Adult agromyzid flies or noctuid moths laying eggs inside tender young leaf tissue.",
            "Favorable warm temperatures (22-30°C) with dense foliage facilitating larval feeding."
        ],
        "organic_treatment": [
            "Spray Bacillus thuringiensis (Bt) kurstaki strain @ 2g/L targeting early instar larvae.",
            "Foliar spray of Beauveria bassiana entomopathogenic fungus @ 5g/L during late evening.",
            "Deploy sex pheromone traps @ 5 traps/acre for adult moth monitoring and mass trapping."
        ],
        "chemical_control": [
            "Emamectin Benzoate 5% SG @ 0.4g/L or Spinosad 45% SC @ 0.3 ml/L for translaminar ingestion action.",
            "Chlorantraniliprole 18.5% SC (Coragen) @ 0.4 ml/L for extended ovicidal and larvicidal protection."
        ],
        "preventive_tips": [
            "Manually collect and destroy heavily mined or egg-mass bearing leaves in early stages.",
            "Conduct deep summer plowing to expose pupae in soil to solar heat and predatory birds."
        ]
    },
    "Spider Mites / Thrips": {
        "scientific_name": "Tetranychus urticae / Frankliniella occidentalis",
        "category": "Acarid & Micro-Sucking Pest",
        "stress_level": "High to Severe (60-85%)",
        "severity_color": "#ec4899",
        "etl_threshold": "ETL Trigger: 5 mites or thrips per leaf underside or noticeable stippling on 10% canopy.",
        "symptoms": [
            "Dense silvery or chlorotic micro-stippling (pinhole speckling) across the upper leaf lamina.",
            "Fine silken web spinning on leaf undersides trapping dust and curling leaf edges.",
            "Leaf bronzing, silvering, and paper-thin brittle drying under heavy infestation."
        ],
        "causes": [
            "Hot, dry, and dusty microclimate conditions (> 30°C and < 50% relative humidity).",
            "Elimination of natural predatory mite populations from broad-spectrum pyrethroid sprays."
        ],
        "organic_treatment": [
            "Overhead sprinkler misting to raise localized humidity and disrupt fine webbing.",
            "Release phytoseiid predatory mites (Phytoseiulus persimilis or Neoseiulus californicus).",
            "Apply wettable sulfur 80% WP @ 3g/L or insecticidal potassium soap @ 5 ml/L."
        ],
        "chemical_control": [
            "Propargite 57% EC @ 2 ml/L or Fenpyroximate 5% EC @ 1 ml/L for contact miticidal knockdown.",
            "Spiromesifen 22.9% SC @ 1 ml/L for ovicidal and nymphal growth inhibition."
        ],
        "preventive_tips": [
            "Avoid water deficit stress as drought-stressed plants are preferred mite hosts.",
            "Maintain weed-free field boundaries to eliminate alternate wild host plants."
        ]
    }
}

CLASS_NAMES = list(DISEASE_DATABASE.keys())
