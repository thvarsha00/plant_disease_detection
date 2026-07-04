"""
Built-in fallback remedy content for every class the model can predict.

This module exists so the app is fully useful even when the Gemini API key
is missing, invalid, or the request fails at runtime. Each entry is shown
as-is when Gemini is unavailable, and is also handed to Gemini as grounding
context when it IS available, so Gemini's job becomes "personalize and
translate this" rather than "invent treatment advice from scratch."

Schema per entry:
{
    "is_healthy": bool,
    "severity": str,            # "None" for healthy, else Low/Medium/High
    "summary": str,             # 1-2 plain-language sentences on what it is
    "organic": [str, ...],      # organic / cultural treatment steps
    "chemical": [str, ...],     # chemical treatment options
    "prevention": [str, ...],   # prevention tips for future crops
}
"""

REMEDIES = {

    "Apple___Apple_scab": {
        "is_healthy": False,
        "severity": "Medium",
        "summary": "A fungal disease causing dark, scabby spots on leaves and fruit, worse in cool, wet spring weather.",
        "organic": [
            "Remove and destroy fallen leaves in autumn to reduce fungal spores overwintering in the orchard.",
            "Prune the canopy to improve airflow and sunlight penetration.",
            "Apply a sulfur or copper-based organic fungicide at bud break and repeat every 7-10 days in wet weather.",
        ],
        "chemical": [
            "Apply a protectant fungicide such as captan or myclobutanil starting at green tip stage.",
            "Repeat sprays every 7-14 days through the wet spring period per product label.",
        ],
        "prevention": [
            "Choose scab-resistant apple varieties for new plantings.",
            "Rake and dispose of fallen leaves every autumn.",
            "Avoid overhead irrigation that keeps leaves wet for long periods.",
        ],
    },

    "Apple___Black_rot": {
        "is_healthy": False,
        "severity": "Medium",
        "summary": "A fungal disease causing brown leaf spots, fruit rot, and sunken bark cankers on branches.",
        "organic": [
            "Prune out and burn or bury cankered branches and mummified fruit.",
            "Apply copper-based fungicide sprays during the growing season.",
            "Improve tree vigor with balanced compost feeding to reduce susceptibility.",
        ],
        "chemical": [
            "Apply captan or thiophanate-methyl fungicide sprays from petal fall through summer per label.",
        ],
        "prevention": [
            "Remove mummified fruit and dead wood every dormant season.",
            "Avoid tree stress from drought or nutrient deficiency.",
            "Space trees to allow good air circulation.",
        ],
    },

    "Apple___Cedar_apple_rust": {
        "is_healthy": False,
        "severity": "Low",
        "summary": "A fungal disease needing both apple and cedar/juniper trees to complete its cycle, causing orange leaf spots.",
        "organic": [
            "Remove nearby juniper or cedar trees within a few hundred meters if practical.",
            "Apply sulfur spray at bud break through early summer.",
        ],
        "chemical": [
            "Apply myclobutanil or propiconazole fungicide starting at pink bud stage, repeating per label instructions.",
        ],
        "prevention": [
            "Plant rust-resistant apple varieties.",
            "Avoid planting apples near ornamental junipers/cedars.",
        ],
    },

    "Apple___healthy": {
        "is_healthy": True,
        "severity": "None",
        "summary": "The apple leaf shows no visible signs of disease and appears healthy.",
        "prevention": [
            "Continue regular watering at the base to keep foliage dry.",
            "Apply balanced fertilizer in early spring for steady growth.",
            "Inspect leaves every couple of weeks for early signs of pests or disease.",
        ],
    },

    "Blueberry___healthy": {
        "is_healthy": True,
        "severity": "None",
        "summary": "The blueberry plant appears healthy with no visible disease symptoms.",
        "prevention": [
            "Keep soil acidic (pH 4.5-5.5) with mulch such as pine bark.",
            "Water consistently, especially during fruit development.",
            "Prune out old, unproductive canes each dormant season.",
        ],
    },

    "Cherry_(including_sour)___Powdery_mildew": {
        "is_healthy": False,
        "severity": "Low",
        "summary": "A fungal disease that coats leaves and shoots with a white powdery growth, favored by warm, dry days and humid nights.",
        "organic": [
            "Spray a potassium bicarbonate or sulfur-based fungicide at first sign of white patches.",
            "Prune to open up the canopy and increase air circulation.",
            "Remove and destroy heavily infected shoots.",
        ],
        "chemical": [
            "Apply a triazole fungicide such as myclobutanil per label instructions.",
        ],
        "prevention": [
            "Avoid excess nitrogen fertilizer, which encourages soft, susceptible growth.",
            "Water at the base rather than overhead.",
        ],
    },

    "Cherry_(including_sour)___healthy": {
        "is_healthy": True,
        "severity": "None",
        "summary": "The cherry leaf shows no signs of disease and appears healthy.",
        "prevention": [
            "Prune annually to maintain open airflow through the canopy.",
            "Water deeply but infrequently to encourage strong roots.",
            "Monitor for early signs of pests such as aphids or fruit flies.",
        ],
    },

    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "is_healthy": False,
        "severity": "Medium",
        "summary": "A fungal disease producing rectangular gray-brown lesions on corn leaves, spreading fastest in humid, warm conditions.",
        "organic": [
            "Rotate crops away from corn for at least one season to break the disease cycle.",
            "Remove and compost or bury infected crop residue after harvest.",
        ],
        "chemical": [
            "Apply a strobilurin or triazole foliar fungicide at early symptom onset, especially on susceptible hybrids.",
        ],
        "prevention": [
            "Plant resistant hybrids where available.",
            "Practice residue management with tillage or crop rotation.",
            "Avoid dense planting that limits airflow.",
        ],
    },

    "Corn_(maize)___Common_rust_": {
        "is_healthy": False,
        "severity": "Low",
        "summary": "A fungal disease that produces small reddish-brown pustules on both leaf surfaces.",
        "organic": [
            "Remove heavily infected lower leaves if the outbreak is limited.",
            "Encourage good field airflow through proper plant spacing.",
        ],
        "chemical": [
            "Apply a triazole or strobilurin fungicide if rust appears before tasseling and conditions favor spread.",
        ],
        "prevention": [
            "Plant rust-resistant hybrids.",
            "Avoid late planting that exposes young plants to peak spore periods.",
        ],
    },

    "Corn_(maize)___Northern_Leaf_Blight": {
        "is_healthy": False,
        "severity": "Medium",
        "summary": "A fungal disease causing long, cigar-shaped gray-green lesions on corn leaves that reduce yield if severe.",
        "organic": [
            "Rotate with a non-host crop such as soybean for at least one year.",
            "Till or bury infected residue after harvest to speed decomposition.",
        ],
        "chemical": [
            "Apply a foliar fungicide (e.g. strobilurin-triazole mix) at early lesion onset, especially near tasseling.",
        ],
        "prevention": [
            "Choose resistant hybrids.",
            "Rotate crops and manage residue every season.",
        ],
    },

    "Corn_(maize)___healthy": {
        "is_healthy": True,
        "severity": "None",
        "summary": "The corn leaf shows no visible disease symptoms and appears healthy.",
        "prevention": [
            "Maintain balanced nitrogen, phosphorus, and potassium fertilization.",
            "Scout fields regularly, especially after humid weather.",
            "Rotate crops each season to reduce disease buildup.",
        ],
    },

    "Grape___Black_rot": {
        "is_healthy": False,
        "severity": "Medium",
        "summary": "A fungal disease causing circular brown leaf spots and shriveled, mummified berries.",
        "organic": [
            "Remove mummified berries and infected canes during dormant pruning.",
            "Apply copper or sulfur-based fungicide sprays from bud break through fruit set.",
        ],
        "chemical": [
            "Apply myclobutanil or mancozeb fungicide starting at bud break and continuing on a 10-14 day schedule through early summer.",
        ],
        "prevention": [
            "Prune for an open canopy to speed leaf drying.",
            "Clear fallen leaves and mummified fruit each season.",
        ],
    },

    "Grape___Esca_(Black_Measles)": {
        "is_healthy": False,
        "severity": "High",
        "summary": "A complex fungal trunk disease causing striped 'tiger stripe' leaf discoloration and internal wood decay.",
        "organic": [
            "Prune out and destroy infected wood well below visible symptoms, disinfecting tools between cuts.",
            "Avoid large pruning wounds during wet weather when spores are active.",
        ],
        "chemical": [
            "No fully reliable chemical cure exists; wound-protectant fungicidal pastes can be applied to fresh pruning cuts.",
        ],
        "prevention": [
            "Prune during dry weather and seal larger cuts with a wound protectant.",
            "Remove and destroy severely affected vines to limit spread.",
        ],
    },

    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "is_healthy": False,
        "severity": "Low",
        "summary": "A fungal disease causing angular brown spots on grape leaves, mainly a cosmetic and vigor issue.",
        "organic": [
            "Remove and destroy heavily spotted leaves.",
            "Apply a copper-based fungicide spray during humid periods.",
        ],
        "chemical": [
            "Apply a mancozeb or copper-based fungicide per label during the growing season.",
        ],
        "prevention": [
            "Improve canopy airflow through pruning and leaf thinning.",
            "Avoid overhead watering.",
        ],
    },

    "Grape___healthy": {
        "is_healthy": True,
        "severity": "None",
        "summary": "The grape leaf shows no visible signs of disease and appears healthy.",
        "prevention": [
            "Maintain an open canopy with regular pruning and leaf thinning.",
            "Water at the base to keep foliage dry.",
            "Monitor for early signs of mildew after humid weather.",
        ],
    },

    "Orange___Haunglongbing_(Citrus_greening)": {
        "is_healthy": False,
        "severity": "High",
        "summary": "A serious bacterial disease spread by psyllid insects, causing blotchy yellowing leaves and bitter, misshapen fruit; there is no cure.",
        "organic": [
            "Remove and destroy infected trees promptly to reduce the source of infection for nearby trees.",
            "Use reflective mulches or sticky traps to reduce psyllid populations.",
        ],
        "chemical": [
            "Apply approved insecticides to control the Asian citrus psyllid vector, following local agricultural extension guidance.",
        ],
        "prevention": [
            "Plant only certified disease-free nursery stock.",
            "Monitor and control psyllid populations proactively.",
            "Report suspected cases to local agricultural authorities, as this disease is often regulated.",
        ],
    },

    "Peach___Bacterial_spot": {
        "is_healthy": False,
        "severity": "Medium",
        "summary": "A bacterial disease causing small dark, angular spots on leaves and fruit, worsened by warm, wet weather.",
        "organic": [
            "Apply copper-based bactericide sprays during dormant season and early leaf-out.",
            "Avoid overhead irrigation that splashes bacteria onto leaves.",
        ],
        "chemical": [
            "Apply copper-based sprays combined with mancozeb in early season per label instructions.",
        ],
        "prevention": [
            "Choose bacterial-spot-resistant peach varieties.",
            "Avoid excessive nitrogen fertilization, which encourages susceptible new growth.",
        ],
    },

    "Peach___healthy": {
        "is_healthy": True,
        "severity": "None",
        "summary": "The peach leaf shows no visible disease symptoms and appears healthy.",
        "prevention": [
            "Prune annually for good light and airflow.",
            "Water deeply and consistently, especially during fruit development.",
            "Watch for early signs of leaf curl or bacterial spot in spring.",
        ],
    },

    "Pepper,_bell___Bacterial_spot": {
        "is_healthy": False,
        "severity": "Medium",
        "summary": "A bacterial disease causing small water-soaked spots on leaves and fruit that turn brown and scabby.",
        "organic": [
            "Remove and destroy infected leaves and plant debris.",
            "Apply copper-based bactericide sprays at first sign of spotting.",
            "Avoid working in the field when foliage is wet to prevent spreading bacteria.",
        ],
        "chemical": [
            "Apply a copper plus mancozeb combination spray on a 7-10 day schedule during humid weather.",
        ],
        "prevention": [
            "Use certified disease-free seed or transplants.",
            "Rotate away from peppers and tomatoes for at least a year.",
            "Avoid overhead irrigation.",
        ],
    },

    "Pepper,_bell___healthy": {
        "is_healthy": True,
        "severity": "None",
        "summary": "The bell pepper leaf shows no visible disease symptoms and appears healthy.",
        "prevention": [
            "Water at the base to keep leaves dry.",
            "Rotate crops each season to limit soil-borne disease buildup.",
            "Stake or cage plants to improve airflow.",
        ],
    },

    "Potato___Early_blight": {
        "is_healthy": False,
        "severity": "Medium",
        "summary": "A fungal disease producing dark, target-like concentric ring spots on lower, older leaves first.",
        "organic": [
            "Remove and destroy affected lower leaves as soon as they appear.",
            "Apply copper-based fungicide sprays every 7-10 days during humid weather.",
            "Ensure balanced fertilization; stressed plants are more susceptible.",
        ],
        "chemical": [
            "Apply chlorothalonil or mancozeb fungicide on a 7-10 day preventive schedule during the growing season.",
        ],
        "prevention": [
            "Rotate potatoes with non-host crops for at least two years.",
            "Space plants to allow good airflow.",
            "Water at the base rather than overhead.",
        ],
    },

    "Potato___Late_blight": {
        "is_healthy": False,
        "severity": "High",
        "summary": "A fast-spreading fungal-like disease (the same pathogen behind the Irish potato famine) causing dark, water-soaked lesions that can destroy a crop within days.",
        "organic": [
            "Remove and destroy infected plants immediately, ideally by burning or deep burial away from the field.",
            "Apply a copper-based fungicide preventively before symptoms appear in high-risk, cool wet weather.",
        ],
        "chemical": [
            "Apply a systemic fungicide such as one containing mefenoxam or chlorothalonil as soon as risk conditions appear, repeating per label.",
        ],
        "prevention": [
            "Use certified disease-free seed potatoes.",
            "Avoid overhead irrigation and improve field drainage.",
            "Destroy volunteer potato plants and cull piles that can harbor the pathogen.",
        ],
    },

    "Potato___healthy": {
        "is_healthy": True,
        "severity": "None",
        "summary": "The potato leaf shows no visible disease symptoms and appears healthy.",
        "prevention": [
            "Hill soil around stems to protect developing tubers.",
            "Water consistently and avoid wetting foliage late in the day.",
            "Scout regularly for early blight spots on lower leaves.",
        ],
    },

    "Raspberry___healthy": {
        "is_healthy": True,
        "severity": "None",
        "summary": "The raspberry plant appears healthy with no visible disease symptoms.",
        "prevention": [
            "Prune out old floricanes after fruiting to improve airflow.",
            "Space canes to reduce humidity within the row.",
            "Water at the base and avoid wetting foliage.",
        ],
    },

    "Soybean___healthy": {
        "is_healthy": True,
        "severity": "None",
        "summary": "The soybean leaf shows no visible disease symptoms and appears healthy.",
        "prevention": [
            "Rotate with non-legume crops to manage soil-borne disease.",
            "Monitor fields regularly during humid periods for early rust or spot symptoms.",
            "Maintain balanced soil fertility based on a recent soil test.",
        ],
    },

    "Squash___Powdery_mildew": {
        "is_healthy": False,
        "severity": "Medium",
        "summary": "A fungal disease coating leaves with white powdery patches, common in warm days and cool, humid nights.",
        "organic": [
            "Spray a potassium bicarbonate, neem oil, or diluted milk solution at first sign of white patches.",
            "Remove and destroy heavily infected leaves.",
            "Improve spacing between plants for better airflow.",
        ],
        "chemical": [
            "Apply a sulfur-based or triazole fungicide per label instructions once symptoms appear.",
        ],
        "prevention": [
            "Choose powdery-mildew-resistant squash varieties.",
            "Avoid overhead watering and water early in the day.",
        ],
    },

    "Strawberry___Leaf_scorch": {
        "is_healthy": False,
        "severity": "Low",
        "summary": "A fungal disease causing small purplish spots that merge into scorched-looking patches on strawberry leaves.",
        "organic": [
            "Remove and destroy old, infected leaves after harvest.",
            "Apply a copper-based fungicide during wet spring weather.",
        ],
        "chemical": [
            "Apply captan or a triazole fungicide during periods of high disease pressure per label instructions.",
        ],
        "prevention": [
            "Use disease-free transplants.",
            "Renovate beds after harvest by removing old foliage.",
            "Avoid overhead irrigation late in the day.",
        ],
    },

    "Strawberry___healthy": {
        "is_healthy": True,
        "severity": "None",
        "summary": "The strawberry leaf shows no visible disease symptoms and appears healthy.",
        "prevention": [
            "Mulch with straw to keep fruit and lower leaves dry.",
            "Renovate beds and remove old foliage after harvest each year.",
            "Water at the base in the morning.",
        ],
    },

    "Tomato___Bacterial_spot": {
        "is_healthy": False,
        "severity": "Medium",
        "summary": "A bacterial disease causing small dark, greasy-looking spots on leaves, stems, and fruit.",
        "organic": [
            "Remove and destroy infected leaves and avoid working with wet plants.",
            "Apply a copper-based bactericide spray at first sign of spotting.",
        ],
        "chemical": [
            "Apply copper plus mancozeb combination sprays on a 7-10 day schedule during humid weather.",
        ],
        "prevention": [
            "Use certified disease-free seed or transplants.",
            "Rotate crops away from tomatoes and peppers for at least a year.",
            "Avoid overhead irrigation.",
        ],
    },

    "Tomato___Early_blight": {
        "is_healthy": False,
        "severity": "Medium",
        "summary": "A fungal disease causing dark, target-like ring spots on older, lower leaves first, which can spread upward.",
        "organic": [
            "Remove and destroy affected lower leaves promptly.",
            "Mulch around the base to reduce soil splash onto leaves.",
            "Apply a copper-based fungicide every 7-10 days in humid weather.",
        ],
        "chemical": [
            "Apply chlorothalonil or mancozeb fungicide on a 7-10 day preventive schedule.",
        ],
        "prevention": [
            "Stake or cage plants to improve airflow.",
            "Rotate crops each season.",
            "Water at the base rather than overhead.",
        ],
    },

    "Tomato___Late_blight": {
        "is_healthy": False,
        "severity": "High",
        "summary": "A fast-moving disease causing large, water-soaked, dark lesions on leaves and fruit that can destroy a crop within days in cool, wet weather.",
        "organic": [
            "Remove and destroy infected plants immediately, away from healthy plants.",
            "Apply a copper-based fungicide preventively before symptoms appear during high-risk weather.",
        ],
        "chemical": [
            "Apply a systemic fungicide containing chlorothalonil or mefenoxam as soon as risk conditions appear, repeating per label.",
        ],
        "prevention": [
            "Space and stake plants for maximum airflow.",
            "Avoid overhead irrigation, especially in the evening.",
            "Monitor local blight forecasts during cool, wet seasons.",
        ],
    },

    "Tomato___Leaf_Mold": {
        "is_healthy": False,
        "severity": "Low",
        "summary": "A fungal disease common in humid greenhouses, causing pale spots on top of leaves and olive-green mold underneath.",
        "organic": [
            "Increase ventilation and reduce humidity around plants.",
            "Remove and destroy affected leaves.",
            "Apply a copper-based fungicide if humidity cannot be reduced quickly.",
        ],
        "chemical": [
            "Apply chlorothalonil or a triazole fungicide per label instructions.",
        ],
        "prevention": [
            "Improve greenhouse or garden ventilation.",
            "Avoid overhead watering and water early in the day.",
            "Space plants to reduce humidity buildup.",
        ],
    },

    "Tomato___Septoria_leaf_spot": {
        "is_healthy": False,
        "severity": "Medium",
        "summary": "A fungal disease producing many small, circular spots with dark borders and gray centers, starting on lower leaves.",
        "organic": [
            "Remove and destroy affected lower leaves as soon as spotted.",
            "Mulch around plants to reduce soil splash.",
            "Apply a copper-based fungicide every 7-10 days in wet weather.",
        ],
        "chemical": [
            "Apply chlorothalonil or mancozeb fungicide on a 7-10 day schedule.",
        ],
        "prevention": [
            "Rotate crops away from tomatoes for at least a year.",
            "Stake plants and remove lower leaves that touch the soil.",
        ],
    },

    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "is_healthy": False,
        "severity": "Medium",
        "summary": "A pest infestation (not a disease) causing fine yellow speckling and webbing on leaves, worse in hot, dry conditions.",
        "organic": [
            "Spray leaves (especially undersides) with a strong jet of water to dislodge mites.",
            "Apply insecticidal soap or neem oil, covering leaf undersides thoroughly.",
            "Introduce natural predators such as predatory mites or ladybugs if practical.",
        ],
        "chemical": [
            "Apply a labeled miticide if infestation is severe, rotating chemical classes to avoid resistance.",
        ],
        "prevention": [
            "Keep plants well-watered, since drought-stressed plants are more vulnerable.",
            "Monitor closely during hot, dry spells.",
        ],
    },

    "Tomato___Target_Spot": {
        "is_healthy": False,
        "severity": "Medium",
        "summary": "A fungal disease causing brown, target-like spots with concentric rings on leaves, stems, and fruit.",
        "organic": [
            "Remove and destroy infected leaves and plant debris.",
            "Apply a copper-based fungicide every 7-10 days during humid weather.",
        ],
        "chemical": [
            "Apply chlorothalonil or a strobilurin fungicide per label instructions.",
        ],
        "prevention": [
            "Stake plants and prune for good airflow.",
            "Rotate crops and remove crop debris after harvest.",
        ],
    },

    "Tomato___Tomato_mosaic_virus": {
        "is_healthy": False,
        "severity": "High",
        "summary": "A viral disease causing mottled light and dark green patterns on leaves, stunted growth, and reduced yield; there is no cure once infected.",
        "organic": [
            "Remove and destroy infected plants promptly to prevent spread.",
            "Wash hands and tools with soap after handling infected plants, since the virus spreads easily by contact.",
        ],
        "chemical": [
            "No chemical treatment cures a viral infection; focus on removing infected plants and controlling spread.",
        ],
        "prevention": [
            "Use certified virus-free seed and resistant varieties where available.",
            "Avoid handling healthy plants after touching infected ones, or tobacco products, without washing hands first.",
        ],
    },

    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "is_healthy": False,
        "severity": "High",
        "summary": "A viral disease spread by whiteflies, causing upward-curling, yellowing leaves and severely stunted plants.",
        "organic": [
            "Remove and destroy infected plants promptly.",
            "Use yellow sticky traps and reflective mulch to reduce whitefly populations.",
            "Introduce or protect natural whitefly predators where possible.",
        ],
        "chemical": [
            "Apply an approved insecticide targeting whiteflies to reduce virus transmission, following local guidance.",
        ],
        "prevention": [
            "Plant resistant tomato varieties where available.",
            "Use insect netting on seedlings and control whiteflies early in the season.",
        ],
    },

    "Tomato___healthy": {
        "is_healthy": True,
        "severity": "None",
        "summary": "The tomato leaf shows no visible disease symptoms and appears healthy.",
        "prevention": [
            "Stake or cage plants for good airflow and light.",
            "Water at the base in the morning to keep foliage dry.",
            "Rotate planting location each season to reduce soil-borne disease risk.",
        ],
    },
}


def get_remedy(class_name):
    """Returns the remedy dict for a raw CLASS_NAMES key, with a safe generic fallback."""
    return REMEDIES.get(class_name, {
        "is_healthy": False,
        "severity": "Unknown",
        "summary": "Specific reference information for this class is not yet available.",
        "organic": ["Isolate the affected plant and monitor closely.",
                     "Remove visibly damaged leaves to slow any potential spread."],
        "chemical": ["Consult a local agricultural extension office for a targeted treatment recommendation."],
        "prevention": ["Practice crop rotation and good field sanitation as general precautions."],
    })
