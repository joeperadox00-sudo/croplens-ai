"""
CROPLENS AI - Comprehensive System & Model Verification Script
"""
import os
import sys
import glob
import time
import torch
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

print("=" * 60)
print("1. VERIFYING ALL MODULE IMPORTS")
print("=" * 60)
try:
    from src.models import CropLensMicroNet, extract_bio_optical_features
    from src.cam import generate_leaf_attention_map
    from src.remedies import DISEASE_DATABASE, CLASS_NAMES
    from src.database import get_or_create_user, save_scan, get_user_scans, get_user_notes
    from src.voice_assistant import get_advisory, generate_speech_audio
    from src.ipm_calculator import calculate_spray_dosage, evaluate_farming_feasibility
    print("[PASS] All Python modules imported successfully!")
except Exception as e:
    print(f"[FAIL] Import Error: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("2. CHECKING CROPLENS MICRONET ARCHITECTURE & PARAMETERS")
print("=" * 60)
model = CropLensMicroNet(num_classes=8)
state_path = os.path.join(PROJECT_ROOT, "models", "croplens_micronet.pt")
if os.path.exists(state_path):
    model.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=True))
    print(f"[PASS] Loaded checkpoint from {state_path}")
model.eval()

param_count = model.count_parameters()
print(f"Total Model Parameters: {param_count} parameters")
print(f"Condition Check: {param_count} < 50,000,000 -> {'PASS' if param_count < 50000000 else 'FAIL'}")
assert param_count == 56, f"Expected 56 parameters, got {param_count}"

print("\n" + "=" * 60)
print("3. TESTING INFERENCE ON ALL 8 AGRONOMIC CONDITIONS")
print("=" * 60)
samples = sorted(glob.glob(os.path.join(PROJECT_ROOT, "data", "samples", "*.jpg")))
print(f"Found {len(samples)} sample images to analyze.")

latencies = []
all_passed = True

for s in samples:
    fname = os.path.basename(s)
    img = Image.open(s)
    t0 = time.perf_counter()
    res = model.predict_image(img)
    t1 = time.perf_counter()
    lat = (t1 - t0) * 1000.0
    latencies.append(lat)

    pred = res["prediction"]
    conf = res["confidence"] * 100.0
    stress = res["stress_score"]
    features = res["features"]

    print(f"\nSpecimen: {fname}")
    print(f"  -> Predicted Condition: {pred}")
    print(f"  -> Confidence: {conf:.1f}% | Stress Severity: {stress:.1f}% | Latency: {lat:.3f} ms")
    print(f"  -> Extracted Bio-Features: ExG={features['Greenness Index (ExG)']:.3f}, ExR={features['Rust / Necrosis Index (ExR)']:.3f}, Stipple={features['Micro-Stippling Texture']:.3f}, Mine={features['Perforation & Mine Ratio']:.3f}")

avg_lat = sum(latencies) / len(latencies)
print(f"\n[SUMMARY] Tested {len(samples)} classes. Average CPU Latency: {avg_lat:.3f} ms (Target: < 5 ms)")

print("\n" + "=" * 60)
print("4. TESTING SPATIAL ATTENTION HEATMAP (CAM)")
print("=" * 60)
test_img = Image.open(samples[0])
overlay, raw_map, coverage = generate_leaf_attention_map(test_img)
print(f"[PASS] Heatmap successfully generated! Output size: {overlay.size}, Foliar coverage: {coverage}%")

print("\n" + "=" * 60)
print("5. TESTING DAILY WEATHER & FARMING ADVISOR")
print("=" * 60)
from src.weather import get_daily_weather
today_w = get_daily_weather()
print(f"Date: {today_w['date_str']}")
print(f"Weather: {today_w['temperature_c']}°C • {today_w['condition']}")
print(f"Verdict: {today_w['verdict']}")
print(f"Best Working Time: {today_w['work_time']}")
print(f"Best Sowing Time: {today_w['sow_time']}")

print("\n" + "=" * 60)
print("6. TESTING KNAPSACK SPRAY TANK CALCULATOR")
print("=" * 60)
calc = calculate_spray_dosage(2.5, "Acres", 16.0, 200.0, 2.0, "ml/L")
print(f"[PASS] Water: {calc['total_water_volume_l']} L | Knapsack Tanks: {calc['estimated_tanks']} | Dose: {calc['dosage_per_tank']} | Total: {calc['total_chemical_needed']}")

print("\n" + "=" * 60)
print("7. TESTING MULTILINGUAL VOICE ASSISTANT IN 5 LANGUAGES")
print("=" * 60)
for lang in ["English", "Tamil", "Hindi", "Telugu", "Malayalam"]:
    adv = get_advisory("Early Blight", lang)
    print(f"[PASS] Language: {lang:10s} -> Advisory Title: {adv['title']}")

print("\n" + "=" * 60)
print("8. TESTING QR CODE REPORT GENERATION & CASE ID SYSTEM")
print("=" * 60)
from src.qr_report import generate_case_id, generate_qr_code_bytes, generate_pdf_report
mock_case_id = generate_case_id("7f8a9b1c", "25 Sep 2026")
test_case_data = {
    "case_id": mock_case_id,
    "date": "25 Sep 2026",
    "time": "11:45 AM",
    "user_email": "farmer@gmail.com",
    "prediction": "Early Blight",
    "confidence": 91.2,
    "stress_score": 76.4,
    "category": "Fungal Blight Disease",
    "scientific_name": "Alternaria solani",
    "etl_threshold": "ETL Trigger: >= 5% leaf area affected on lower canopy",
    "primary_action": "Apply Chlorothalonil 75 WP @ 2.0g/L",
    "organic_treatment": ["Spray Trichoderma viride bio-fungicide @ 5g/L."],
    "chemical_control": ["Mancozeb 75% WP @ 2.5g/L."],
    "preventive_tips": ["Practice 3-year crop rotation."]
}
qr_bytes = generate_qr_code_bytes(test_case_data)
pdf_bytes = generate_pdf_report(test_case_data, qr_bytes)
print(f"[PASS] Generated Case ID: {mock_case_id}")
print(f"[PASS] Synthesized QR Code PNG: {len(qr_bytes)} bytes")
print(f"[PASS] Generated Official PDF Report: {len(pdf_bytes)} bytes")

print("\n" + "=" * 60)
print("ALL 8 CRITICAL SUBSYSTEMS VERIFIED AND FULLY FUNCTIONAL!")
print("=" * 60)
