"""
CROPLENS AI - Agronomic Weather Feasibility & Spray Tank Calculation Engine
Features:
1. Daily Weather-Based Farming & Crop Sowing Feasibility Advisor
   (Tells farmers whether today's weather is suitable to sow, transplant, spray, or cultivate)
2. Knapsack Spray Tank & Formulation Dosage Calculator
100% Offline, Pure Python math, zero external APIs.
"""

from typing import Dict, Any, List


def evaluate_farming_feasibility(
    temperature_c: float,
    humidity_pct: float,
    weather_condition: str,
    wind_speed_kmh: float = 12.0,
    crop_category: str = "General Crops (Vegetables, Pulses, Cereals)"
) -> Dict[str, Any]:
    """
    Evaluates today's microclimate & weather parameters to determine whether
    farmers should proceed with sowing, transplanting, spraying, or field cultivation today.
    """
    score = 100  # Start from optimal 100 and deduct based on weather stresses
    reasons = []
    cautions = []

    # 1. Rainfall / Storm Analysis
    is_storm = "Heavy Downpour" in weather_condition or "Storm" in weather_condition
    is_drizzle = "Drizzle" in weather_condition or "Showers" in weather_condition

    if is_storm:
        score -= 60
        reasons.append("Heavy rain will wash away unestablished seeds, cause topsoil erosion, and lead to root waterlogging/damping-off.")
    elif is_drizzle:
        score -= 20
        cautions.append("Light precipitation provides good natural moisture for transplanted seedlings, but foliage is too wet for chemical sprays.")

    # 2. Temperature Stress Analysis
    if temperature_c >= 40.0:
        score -= 45
        reasons.append(f"Severe heatwave ({temperature_c}°C): Topsoil will dry rapidly, causing high seed mortality and fatal seedling desiccation.")
    elif temperature_c >= 35.0:
        score -= 25
        cautions.append(f"High ambient temperature ({temperature_c}°C): Midday heat will cause wilting. Confine sowing/transplanting to early morning or dusk.")
    elif temperature_c <= 14.0:
        score -= 30
        reasons.append(f"Cold ambient temperature ({temperature_c}°C): Slows soil biological activity and severely delays seed germination.")

    # 3. Relative Humidity Impact
    if humidity_pct >= 92.0 and not is_storm:
        score -= 15
        cautions.append(f"High humidity ({humidity_pct}%): Leaves remain wet for prolonged hours, elevating foliar fungal pathogen risk.")
    elif humidity_pct <= 25.0 and temperature_c >= 32.0:
        score -= 25
        reasons.append(f"Extremely dry air ({humidity_pct}% RH): Accelerates transpiration moisture deficit in newly sown plots.")

    # 4. Wind Speed Impact (Critical for Spraying & Soil Evaporation)
    if wind_speed_kmh >= 28.0:
        score -= 25
        reasons.append(f"Strong winds ({wind_speed_kmh} km/h): Severe risk of pesticide/fertilizer drift and mechanical lodging of tender young shoots.")
    elif wind_speed_kmh >= 18.0:
        score -= 10
        cautions.append(f"Moderate wind speed ({wind_speed_kmh} km/h): Avoid fine-droplet foliar misting to reduce off-target drift.")

    # Final Score Bounds
    final_score = max(5, min(100, score))

    # Determine Verdict & Categorization
    if is_storm or temperature_c >= 40.0 or final_score < 45:
        verdict = "NOT RECOMMENDED TODAY"
        verdict_badge = "🔴 ADVERSE WEATHER — POSTPONE SOWING & FIELDWORK"
        color = "#ef4444"
        summary = (
            "Today's weather poses high risk for farm operations. Seed sowing and pesticide spraying should be postponed "
            "to prevent seed loss, wash-off wastage, and soil compaction."
        )
        working_window = "Limit activities to clearing drainage channels and securing farm nursery covers."

        op_sowing = {
            "status": "🚫 Avoid Sowing",
            "color": "#ef4444",
            "detail": "Seeds risk being washed away by torrential water or desiccated by extreme surface heat."
        }
        op_transplant = {
            "status": "🚫 Avoid Transplanting",
            "color": "#ef4444",
            "detail": "Tender root systems cannot withstand soil saturation or severe thermal shock."
        }
        op_spraying = {
            "status": "🚫 Avoid Spraying",
            "color": "#ef4444",
            "detail": "Rain will cause 100% chemical wash-off, wasting input costs and polluting nearby water bodies."
        }
        op_tillage = {
            "status": "🚫 Avoid Heavy Tillage",
            "color": "#ef4444",
            "detail": "Working wet soils damages soil structure and causes severe plow-pan compaction."
        }

    elif final_score < 80:
        verdict = "PROCEED WITH CAUTION"
        verdict_badge = "🟡 MODERATE CONDITIONS — PROCEED WITH SPECIFIC SAFEGUARDS"
        color = "#f59e0b"
        summary = (
            "Today is moderately acceptable for farm work, but specific precautions are necessary. "
            "Adjust irrigation and avoid midday heat or windy hours."
        )
        working_window = "Optimal working window: Early Morning 6:00 AM – 9:30 AM or Late Afternoon 4:00 PM – 6:30 PM."

        op_sowing = {
            "status": "⚠️ Sowing with Caution",
            "color": "#f59e0b",
            "detail": "Ensure immediate light irrigation after seed placement to guarantee uniform germination moisture."
        }
        op_transplant = {
            "status": "✅ Favorable in Evening",
            "color": "#22c55e",
            "detail": "Transplant seedlings after 4:00 PM so roots establish overnight without harsh solar radiation."
        }
        op_spraying = {
            "status": "⚠️ Spray During Calm Hours",
            "color": "#f59e0b",
            "detail": "Spray early morning while wind is calm and leaves are dry. Do not spray if rain is impending."
        }
        op_tillage = {
            "status": "✅ Workable",
            "color": "#22c55e",
            "detail": "Soil moisture is acceptable for shallow harrowing, bed shaping, or weeding."
        }

    else:
        verdict = "HIGHLY RECOMMENDED"
        verdict_badge = "🟢 OPTIMAL WEATHER — EXCELLENT FOR SOWING & FARMING"
        color = "#22c55e"
        summary = (
            "Weather conditions today are optimal for agricultural operations. Temperature, moisture, and airflow "
            "favor rapid seed germination, root proliferation, and nutrient assimilation."
        )
        working_window = "Ideal field operating conditions throughout Morning (6:00 AM – 11:00 AM) and Afternoon (3:30 PM – 6:30 PM)."

        op_sowing = {
            "status": "✅ Prime Sowing Weather",
            "color": "#22c55e",
            "detail": "Optimal soil temperature and moisture will accelerate germination and produce vigorous seedlings."
        }
        op_transplant = {
            "status": "✅ Excellent Transplanting",
            "color": "#22c55e",
            "detail": "Mild temperature and balanced humidity minimize seedling transplant shock."
        }
        op_spraying = {
            "status": "✅ Optimal Spray Window",
            "color": "#22c55e",
            "detail": "Calm wind and dry leaves enable maximum foliar absorption and zero spray drift."
        }
        op_tillage = {
            "status": "✅ Ideal Tilth & Plowing",
            "color": "#22c55e",
            "detail": "Soil moisture level is at field capacity, producing optimal friable seedbeds."
        }

    return {
        "score": final_score,
        "verdict": verdict,
        "verdict_badge": verdict_badge,
        "color": color,
        "summary": summary,
        "working_window": working_window,
        "temperature_c": temperature_c,
        "humidity_pct": humidity_pct,
        "weather_condition": weather_condition,
        "wind_speed_kmh": wind_speed_kmh,
        "reasons": reasons,
        "cautions": cautions,
        "operations": {
            "Direct Seed Sowing": op_sowing,
            "Seedling Transplanting": op_transplant,
            "Foliar Spraying & Nutrition": op_spraying,
            "Tillage & Land Prep": op_tillage
        }
    }


def calculate_spray_dosage(
    field_size: float,
    unit: str = "Acres",
    sprayer_tank_capacity_l: float = 16.0,
    water_rate_per_acre_l: float = 200.0,
    dosage_rate_per_liter: float = 2.0,
    dosage_unit: str = "ml/L"
) -> Dict[str, Any]:
    """
    Calculates exact water volume, number of knapsack spray tanks, and formulation requirement.
    """
    acres = field_size if unit == "Acres" else field_size * 2.47105

    total_water_needed_l = round(acres * water_rate_per_acre_l, 1)
    num_tanks = round(total_water_needed_l / sprayer_tank_capacity_l, 1)
    dose_per_tank = round(dosage_rate_per_liter * sprayer_tank_capacity_l, 1)
    total_formulation_needed = round(dosage_rate_per_liter * total_water_needed_l, 1)

    unit_name = "ml" if "ml" in dosage_unit else "grams"

    return {
        "field_size_acres": round(acres, 2),
        "total_water_volume_l": total_water_needed_l,
        "tank_capacity_l": sprayer_tank_capacity_l,
        "estimated_tanks": num_tanks,
        "dosage_per_tank": f"{dose_per_tank} {unit_name}",
        "total_chemical_needed": f"{total_formulation_needed} {unit_name}",
        "recommendation": (
            f"Mix {dose_per_tank} {unit_name} per {sprayer_tank_capacity_l}L knapsack sprayer. "
            f"You will need approximately {int(num_tanks + 0.9)} full tanks to cover {round(acres, 2)} acres."
        )
    }
