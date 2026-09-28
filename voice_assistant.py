"""
CROPLENS AI - Multilingual Voice Assistant & Translation Module
Provides native spoken audio and localized text in 5 Indic languages:
- English
- Hindi (हिन्दी)
- Telugu (తెలుగు)
- Malayalam (മലയാളം)
Covers all 8 diagnostic conditions (5 foliar diseases/stresses + 3 pest infestations).
"""

import os
import hashlib
import asyncio
from typing import Dict, Any

AUDIO_CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "audio_cache"))
os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)

# Edge-TTS Neural Voice Models for Indian Languages
VOICE_MAPPING = {
    "Tamil": {"code": "ta", "voice": "ta-IN-PallaviNeural"},
    "English": {"code": "en", "voice": "en-IN-NeerjaNeural"},
    "Hindi": {"code": "hi", "voice": "hi-IN-SwaraNeural"},
    "Telugu": {"code": "te", "voice": "te-IN-ShrutiNeural"},
    "Malayalam": {"code": "ml", "voice": "ml-IN-SobhanaNeural"}
}

# Multilingual Translations for All 8 Diagnoses
TRANSLATIONS: Dict[str, Dict[str, Dict[str, str]]] = {
    "Healthy Leaf": {
        "Tamil": {
            "title": "ஆரோக்கியமான பயிர் இலை",
            "summary": "உங்கள் பயிர் முழுமையான ஆரோக்கியத்துடன் உள்ளது. நோய் தொற்று அல்லது பூச்சி தாக்குதல் எதுவும் இல்லை.",
            "action": "வழக்கமான ஊட்டச்சத்து பராமரிப்பை தொடரவும். முன்னெச்சரிக்கை நடவடிக்கையாக 14 நாட்களுக்கு ஒருமுறை ஒரு லிட்டர் தண்ணீரில் 2 மில்லி வேப்பெண்ணெய் கலந்து தெளிக்கவும்.",
            "spoken_text": "மகிழ்ச்சியான செய்தி! உங்கள் பயிர் இலைகள் மிக ஆரோக்கியமாக உள்ளன. வழக்கமான நீர்ப்பாசனம் செய்து, பாதுகாப்புக்காக வேப்பெண்ணெய் தெளிக்கவும்."
        },
        "English": {
            "title": "Healthy Crop Foliage",
            "summary": "Your crop is in optimal physiological health. No disease pathogens or pest symptoms detected.",
            "action": "Continue regular nutrient maintenance. Apply prophylactic neem oil spray (2 ml per liter) every 14 days as a preventative shield.",
            "spoken_text": "Good news! Your crop is healthy with optimal leaf vigor. Continue standard irrigation and apply neem oil spray as a preventative shield."
        },
        "Hindi": {
            "title": "स्वस्थ फसल",
            "summary": "आपकी फसल पूरी तरह स्वस्थ है। किसी भी प्रकार के रोग या कीट का प्रकोप नहीं पाया गया है।",
            "action": "नियमित पोषण प्रबंधन जारी रखें। रोकथाम के लिए हर 14 दिन में 2 मिली नीम का तेल प्रति लीटर पानी में मिलाकर छिड़काव करें।",
            "spoken_text": "शुभ समाचार! आपकी फसल पूरी तरह स्वस्थ है। सामान्य सिंचाई जारी रखें और सुरक्षा के लिए नीम के तेल का छिड़काव करें।"
        },
        "Telugu": {
            "title": "ఆరోగ్యకరమైన పంట",
            "summary": "మీ పంట సంపూర్ణ ఆరోగ్యంతో ఉంది. ఎటువంటి తెగులు లేదా పురుగుల లక్షణాలు లేవు.",
            "action": "సాధారణ పోషక యాజమాన్యాన్ని కొనసాగించండి. నివారణ చర్యగా ప్రతి 14 రోజులకు ఒకసారి లీటరు నీటికి 2 మిల్లీలీటర్ల వేపనూనె కలిపి పిచికారీ చేయండి.",
            "spoken_text": "శుభవార్త! మీ పంట ఆకులు ఆరోగ్యంగా ఉన్నాయి. సరైన నీటి యాజమాన్యం చేపట్టి, నివారణ కోసం వేపనూనె పిచికారీ చేయండి."
        },
        "Malayalam": {
            "title": "ആരോഗ്യമുള്ള വിള",
            "summary": "വിള പൂർണ്ണ ആരോഗ്യത്തോടെയാണ് വളരുന്നത്. രോഗലക്ഷണങ്ങളോ കീടബാധയോ കാണാനില്ല.",
            "action": "പതിവ് വളപ്രയോഗം തുടരുക. രോഗപ്രതിരോധത്തിനായി 14 ദിവസത്തിലൊരിക്കൽ 2 മില്ലി വേപ്പെണ്ണ ഒരു ലിറ്റർ വെള്ളത്തിൽ കലക്കി തളിക്കുക.",
            "spoken_text": "നല്ല വാർത്ത! നിങ്ങളുടെ വിള ആരോഗ്യത്തോടെ ഇരിക്കുന്നു. പതിവ് ജലസേചനം തുടരുക, വേപ്പെണ്ണ സ്പ്രേ ചെയ്യുക."
        }
    },
    "Early Blight": {
        "Tamil": {
            "title": "முன் பருவ இலை கருகல் நோய் (ஆல்டர்னேரியா)",
            "summary": "இலைகளில் கருகல் நோய் மற்றும் வளைய வடிவ கருப்பு புள்ளிகள் கண்டறியப்பட்டுள்ளன.",
            "action": "இயற்கை முறை: சூடோமோனாஸ் அல்லது ட்ரைக்கோடெர்மா விரிடி 5 கிராம்/லிட்டர் தெளிக்கவும். இரசாயன முறை: மேன்கோசெப் 2.5 கிராம் அல்லது டைபனோகோனசோல் 1 மில்லி ஒரு லிட்டர் தண்ணீரில் கலந்து தெளிக்கவும்.",
            "spoken_text": "எச்சரிக்கை! உங்கள் பயிரில் முன் பருவ இலை கருகல் நோய் தாக்கியுள்ளது. பாதிக்கப்பட்ட கீழ் இலைகளை வெட்டி அகற்றி, உடனடியாக மேன்கோசெப் அல்லது ட்ரைக்கோடெர்மா மருந்தை தெளிக்கவும்."
        },
        "English": {
            "title": "Early Blight Infection (Alternaria)",
            "summary": "Fungal early blight detected with necrotic target-board lesions and yellow chlorotic halos.",
            "action": "Biological: Spray Trichoderma viride or Bacillus subtilis at 5 grams per liter. Chemical: Spray Mancozeb 75 WP at 2.5 grams per liter, or Difenoconazole at 1 ml per liter.",
            "spoken_text": "Warning: Early blight fungal infection detected. Prune infected bottom leaves immediately. Spray Trichoderma viride or Mancozeb fungicide to prevent spread."
        },
        "Hindi": {
            "title": "अगेती झुलसा रोग (अल्टरनेरिया)",
            "summary": "पत्तियों पर अगेती झुलसा फंगल रोग और छल्लेदार धब्बे पाए गए हैं।",
            "action": "जैविक उपचार: ट्राइकोडर्मा विरिडी 5 ग्राम प्रति लीटर छिड़कें। रासायनिक उपचार: मैंकोजेब 2.5 ग्राम या डाइफेनोकोनाजोल 1 मिली प्रति लीटर पानी में मिलाकर छिड़काव करें।",
            "spoken_text": "चेतावनी! आपकी फसल में अगेती झुलसा रोग का प्रकोप है। प्रभावित निचली पत्तियों को काट दें और तुरंत मैंकोजेब या ट्राइकोडर्मा का छिड़काव करें।"
        },
        "Telugu": {
            "title": "ముందస్తు తెగులు (ఆల్టర్నేరియా)",
            "summary": "ఆకులపై ముందస్తు ఎండు తెగులు మరియు నల్లని వలయాల మచ్చలు గుర్తించబడ్డాయి.",
            "action": "సేంద్రీయ పద్ధతి: లీటరు నీటికి 5 గ్రాముల ట్రైకోడెర్మా విరిడి పిచికారీ చేయండి. రసాయన పద్ధతి: లీటరు నీటికి 2.5 గ్రాముల మాంకోజెబ్ లేదా 1 మిల్లీలీటర్ డైఫెనోకోనజోల్ పిచికారీ చేయండి.",
            "spoken_text": "హెచ్చరిక! పంటలో ముందస్తు ఎండు తెగులు సోకింది. తెగులు సోకిన ఆకులను తొలగించి, వెంటనే మాంకోజెబ్ లేదా ట్రైకోడెర్మా మందును పిచికారీ చేయండి."
        },
        "Malayalam": {
            "title": "നേരത്തെയുള്ള കരിച്ചിൽ രോഗം (ആൾട്ടർനേറിയ)",
            "summary": "ഇലകളിൽ കരിച്ചിൽ രോഗത്തിന്റെ ലക്ഷണങ്ങളും കറുത്ത പുള്ളിക്കുത്തുകളും കണ്ടെത്തിയിട്ടുണ്ട്.",
            "action": "ജൈവ നിയന്ത്രണം: ട്രൈക്കോഡെർമ വിരിഡി 5 ഗ്രാം ഒരു ലിറ്റർ വെള്ളത്തിൽ തളിക്കുക. രാസ നിയന്ത്രണം: മാങ്കോസെബ് 2.5 ഗ്രാം അല്ലെങ്കിൽ ഡൈഫെനൊകൊണസോൾ 1 മില്ലി തളിക്കുക.",
            "spoken_text": "മുന്നറിയിപ്പ്! വിളയിൽ കരിച്ചിൽ രോഗം ബാധിച്ചിരിക്കുന്നു. ബാധിച്ച ഇലകൾ നശിപ്പിച്ച് ഉടൻ തന്നെ മാങ്കോസെബ് അല്ലെങ്കിൽ ട്രൈക്കോഡെർമ തളിക്കുക."
        }
    },
    "Common Rust": {
        "Tamil": {
            "title": "துரு நோய் (பக்சீனியா)",
            "summary": "இலைகளின் மேல் பகுதியில் செம்பழுப்பு நிற துரு போன்ற துகள்கள் பரவியுள்ளன.",
            "action": "இயற்கை முறை: நனையும் கந்தகம் 3 கிராம்/லிட்டர் தெளிக்கவும். இரசாயன முறை: புரோபிகோனசோல் 1 மில்லி ஒரு லிட்டர் தண்ணீரில் கலந்து அதிகாலை வேளையில் தெளிக்கவும்.",
            "spoken_text": "பயிரில் துரு நோய் அறிகுறி தென்படுகிறது. அதிகாலை வேளையில் நனையும் கந்தகம் அல்லது புரோபிகோனசோல் பூஞ்சாணக் கொல்லியை ஒரு லிட்டருக்கு ஒரு மில்லி கலந்து தெளிக்கவும்."
        },
        "English": {
            "title": "Common Rust (Puccinia)",
            "summary": "Fungal rust infection with reddish-cinnamon powdery pustules breaking through leaf tissue.",
            "action": "Biological: Spray wettable sulfur 80 WP at 3 grams per liter. Chemical: Spray Propiconazole 25 EC at 1 ml per liter or Tebuconazole at 1 ml per liter.",
            "spoken_text": "Rust fungal disease detected. Spray wettable sulfur powder or Propiconazole at 1 milliliter per liter during cool morning hours."
        },
        "Hindi": {
            "title": "गेरूई / रतुआ रोग (पक्सीनिया)",
            "summary": "पत्तियों पर लाल-भूरे रंग के फफूंद फफोले पाए गए हैं जो पत्तियों को कमजोर करते हैं।",
            "action": "जैविक उपचार: घुलनशील गंधक 3 ग्राम प्रति लीटर छिड़कें। रासायनिक उपचार: प्रोपिकोनाजोल 1 मिली प्रति लीटर पानी में मिलाकर छिड़काव करें।",
            "spoken_text": "फसल में रतुआ या गेरूई रोग के लक्षण मिले हैं। सुबह के समय घुलनशील गंधक या प्रोपिकोनाजोल फफूंदनाशक का छिड़काव करें।"
        },
        "Telugu": {
            "title": "కుంకుమ తెగులు / తుప్పు తెగులు (పక్సీనియా)",
            "summary": "ఆకులపై ఎరుపు-గోధుమ రంగు తుప్పు వంటి బుడిపెలు వ్యాపించాయి.",
            "action": "సేంద్రీయ పద్ధతి: లీటరు నీటికి 3 గ్రాముల గంధకం పొడి కలపండి. రసాయన పద్ధతి: లీటరు నీటికి 1 మిల్లీలీటర్ ప్రొపికోనజోల్ పిచికారీ చేయండి.",
            "spoken_text": "పంటలో తుప్పు తెగులు గమనించబడింది. ఉదయం వేళలో గంధకం పొడి లేదా ప్రొపికోనజోల్ మందును పిచికారీ చేయండి."
        },
        "Malayalam": {
            "title": "തുരുമ്പ് രോഗം (പക്സീനിയ)",
            "summary": "ഇലകളിൽ ചുവപ്പുകലർന്ന തവിട്ടുനിറത്തിലുള്ള തുരുമ്പ് പാടുകൾ പടർന്നുപിടിക്കുന്നു.",
            "action": "ജൈവ നിയന്ത്രണം: നനയ്ക്കാവുന്ന ഗന്ധകം 3 ഗ്രാം തളിക്കുക. രാസ നിയന്ത്രണം: പ്രൊപികൊണാസോൾ 1 മില്ലി ഒരു ലിറ്റർ വെള്ളത്തിൽ തളിക്കുക.",
            "spoken_text": "വിളയിൽ തുരുമ്പ് രോഗം ബാധിച്ചിരിക്കുന്നു. രാവിലെ തന്നെ ഗന്ധകപ്പൊടിയോ പ്രൊപികൊണാസോളോ തളിക്കുക."
        }
    },
    "Leaf Spot / Late Blight": {
        "Tamil": {
            "title": "பின் பருவ இலை கருகல் நோய் / இலை புள்ளி",
            "summary": "இலைகளில் நீர் ஊறிய பழுப்பு நிற புள்ளிகள் மற்றும் தீவிர இலை கருகல் கண்டறியப்பட்டுள்ளது.",
            "action": "இயற்கை முறை: போர்டோ கலவை 1% தெளிக்கவும். இரசாயன முறை: மெட்டலாக்சில் + மேன்கோசெப் (ரிடோமில் கோல்ட்) 2.5 கிராம்/லிட்டர் உடனடியாக தெளிக்கவும்.",
            "spoken_text": "மிக அவசர எச்சரிக்கை! பயிரில் பின் பருவ இலை கருகல் நோய் தீவிரம் அடைந்துள்ளது. உடனடியாக ரிடோமில் கோல்ட் அல்லது மெட்டலாக்சில் மருந்தை தெளித்து பயிரை பாதுகாக்கவும்."
        },
        "English": {
            "title": "Severe Leaf Spot & Late Blight",
            "summary": "Critical oomycete infection rapidly rotting foliar tissue with water-soaked dark lesions.",
            "action": "Critical Emergency: Spray Metalaxyl 8% + Mancozeb 64% WP (Ridomil Gold) at 2.5 grams per liter, or Dimethomorph 50% WP at 1 gram per liter immediately.",
            "spoken_text": "Emergency! Late blight pathogen detected. Water-soaked lesions can destroy foliage in 48 hours. Spray Ridomil Gold or Dimethomorph immediately."
        },
        "Hindi": {
            "title": "पछेती झुलसा और पत्ती धब्बा रोग",
            "summary": "गंभीर फंगल संक्रमण जो तेजी से पत्तियों को गला देता है।",
            "action": "आपातकालीन उपचार: तुरंत मेटालेक्सिल + मैंकोजेब (रिडोमिल गोल्ड) 2.5 ग्राम प्रति लीटर या डाइमेथोमॉर्फ 1 ग्राम प्रति लीटर पानी में मिलाकर छिड़कें।",
            "spoken_text": "आपातकालीन चेतावनी! पछेती झुलसा रोग बहुत तेजी से फैलता है। फसल को बचाने के लिए तुरंत रिडोमिल गोल्ड दवा का छिड़काव करें।"
        },
        "Telugu": {
            "title": "ఆలస్యపు ఎండు తెగులు / ఆకుమచ్చ తెగులు",
            "summary": "ఆకులు వేగంగా కుళ్ళిపోయే అత్యంత తీవ్రమైన శిలీంధ్ర వ్యాధి గుర్తించబడింది.",
            "action": "తక్షణ అత్యవసర చర్య: లీటరు నీటికి 2.5 గ్రాముల రిడోమిల్ గోల్డ్ లేదా 1 గ్రాము డైమెథోమార్ఫ్ కలిపి వెంటనే పిచికారీ చేయండి.",
            "spoken_text": "తీవ్ర హెచ్చరిక! ఆలస్యపు ఎండు తెగులు ఆశించింది. ఇది వేగంగా పంటను నాశనం చేస్తుంది. వెంటనే రిడోమిల్ గోల్డ్ మందును పిచికారీ చేయండి."
        },
        "Malayalam": {
            "title": "ഗുരുതരമായ ഇലപ്പുള്ളി, മുരടിപ്പ് രോഗം",
            "summary": "ഇലകൾ വേഗത്തിൽ അഴുകിപ്പോകുന്ന അതീവ ഗുരുതരമായ കുമിൾ രോഗം കണ്ടെത്തി.",
            "action": "അടിയന്തര നടപടി: റിഡോമിൽ ഗോൾഡ് 2.5 ഗ്രാം അല്ലെങ്കിൽ ഡൈമെത്തോമോർഫ് 1 ഗ്രാം ഒരു ലിറ്റർ വെള്ളത്തിൽ കലക്കി ഉടൻ തളിക്കുക.",
            "spoken_text": "അടിയന്തിര മുന്നറിയിപ്പ്! അതിവേഗം പടരുന്ന കുമിൾ രോഗം ബാധിച്ചിരിക്കുന്നു. ഉടൻ തന്നെ റിഡോമിൽ ഗോൾഡ് സ്പ്രേ ചെയ്യുക."
        }
    },
    "Nutrient Chlorosis / Water Stress": {
        "Tamil": {
            "title": "ஊட்டச்சத்து குறைபாடு / நீர் பற்றாக்குறை",
            "summary": "பச்சைய இழப்பு காரணமாக இலைகள் மஞ்சள் நிறமாக மாறியுள்ளன அல்லது வாடல் ஏற்பட்டுள்ளது.",
            "action": "இயற்கை முறை: பஞ்சகவ்யா 3% அல்லது மண்புழு உரக் கரைசல் தெளிக்கவும். இரசாயன முறை: 19-19-19 நீரில் கரையும் உரம் 5 கிராம் மற்றும் ஃபெரஸ் சல்பேட் 2 கிராம்/லிட்டர் தெளிக்கவும்.",
            "spoken_text": "பயிரில் இரும்புச்சத்து அல்லது தழைச்சத்து பற்றாக்குறை ஏற்பட்டு இலைகள் மஞ்சள் நிறமாகியுள்ளன. பஞ்சகவ்யா அல்லது 19-19-19 உரக் கரைசலை இலைகளில் தெளிக்கவும்."
        },
        "English": {
            "title": "Nutrient Deficiency & Moisture Stress",
            "summary": "Physiological chlorophyll loss causing yellowing and turgor reduction without active pathogen infection.",
            "action": "Foliar spray of 19-19-19 water-soluble NPK fertilizer at 5 grams per liter plus Chelated Iron (Fe-EDTA) at 1 gram per liter. Regulate irrigation.",
            "spoken_text": "Crop shows yellowing due to nutrient deficiency or moisture stress. Spray 19-19-19 balanced fertilizer and chelated iron, and adjust irrigation."
        },
        "Hindi": {
            "title": "पोषक तत्वों की कमी और पानी का तनाव",
            "summary": "यह कोई बीमारी नहीं है, बल्कि नाइट्रोजन या आयरन की कमी से पत्तियां पीली पड़ रही हैं।",
            "action": "19-19-19 एनपीके उर्वरक 5 ग्राम और चिलेटेड आयरन 1 ग्राम प्रति लीटर पानी में मिलाकर पत्तियों पर छिड़कें। सिंचाई संतुलित रखें।",
            "spoken_text": "फसल में कोई बीमारी नहीं है, केवल पोषक तत्वों की कमी से पीलापन है। 19-19-19 खाद और आयरन का छिड़काव करें तथा सिंचाई सुधारें।"
        },
        "Telugu": {
            "title": "పోషకాల లోపం మరియు తేమ లోపం",
            "summary": "ఎటువంటి వ్యాధి లేదు, కేవలం నత్రజని లేదా ఇనుము లోపం వల్ల ఆకులు పసుపు రంగులోకి మారాయి.",
            "action": "చర్య: లీటరు నీటికి 1 గ్రాము చీలేటెడ్ ఐరన్ మరియు 5 గ్రాముల 19-19-19 ఎరువు కలిపి ఆకులపై పిచికారీ చేయండి. సమతుల్యంగా నీరందించండి.",
            "spoken_text": "పోషకాల లోపం వల్ల ఆకులు పసుపు రంగుకు మారాయి. 19-19-19 ఎరువు మరియు ఐరన్ పిచికారీ చేసి సక్రమంగా నీటిని అందించండి."
        },
        "Malayalam": {
            "title": "പോഷകക്കുറവ് / ജലക്ഷാമം",
            "summary": "ഇരുമ്പിന്റെയോ നൈട്രജന്റെയോ കുറവ് മൂലം ഇലകൾ മഞ്ഞനിറമാകുന്നു. രോഗബാധയില്ല.",
            "action": "പരിഹാരം: ചീലേറ്റഡ് ഇരുമ്പ് 1 ഗ്രാം, 19-19-19 വളം 5 ഗ്രാം എന്നിവ ഒരു ലിറ്റർ വെള്ളത്തിൽ കലക്കി ഇലകളിൽ തളിക്കുക. കൃത്യമായി നനയ്ക്കുക.",
            "spoken_text": "പോഷകക്കുറവ് മൂലം ഇലകൾ മഞ്ഞളിക്കുന്നു. 19-19-19 വളവും ഇരുമ്പ് സത്തും തളിക്കുക, കൃത്യമായി നനയ്ക്കുക."
        }
    },
    "Aphids / Whiteflies": {
        "Tamil": {
            "title": "அசுவினி / வெள்ளை ஈக்கள் தாக்குதல்",
            "summary": "இலைகளில் சாறு உறிஞ்சும் பூச்சிகள், தேன் போன்ற பிசுபிசுப்பு மற்றும் இலை சுருட்டல் காணப்படுகிறது.",
            "action": "இயற்கை முறை: ஏக்கருக்கு 15 மஞ்சள் நிற ஒட்டும் பொறிகள் வைக்கவும். வேப்பங்கொட்டை சாறு 5% தெளிக்கவும். இரசாயன முறை: இமிடாக்ளோப்ரிட் 17.8 SL 0.5 மில்லி அல்லது அசிடமிப்ரிட் 0.5 கிராம்/லிட்டர் தெளிக்கவும்.",
            "spoken_text": "பயிரில் அசுவினி அல்லது வெள்ளை ஈக்கள் தாக்கியுள்ளன. மஞ்சள் ஒட்டும் பொறிகளை வைத்து, இமிடாக்ளோப்ரிட் அல்லது வேப்பெண்ணெய் கரைசலை இலையின் அடிப்பகுதியில் தெளிக்கவும்."
        },
        "English": {
            "title": "Aphids & Whiteflies Infestation",
            "summary": "Sucking insect pests detected. Sap drainage is causing leaf curling, honeydew excretion, and sooty mold risk.",
            "action": "Deploy yellow sticky traps (15-20 traps/acre). Spray neem oil (NSKE 5%) or Azadirachtin @ 2 ml/L. For severe infestations, spray Imidacloprid 17.8% SL @ 0.5 ml/L.",
            "spoken_text": "Pest alert! Aphids and whiteflies infestation detected on crop leaves. Install yellow sticky traps and spray neem oil or Imidacloprid to protect leaves from sap drainage."
        },
        "Hindi": {
            "title": "माहू और सफेद मक्खी का प्रकोप",
            "summary": "रस चूसक कीटों का प्रकोप पाया गया है। पत्तियां मुड़ रही हैं और चिपचिपा रस काला फफूंद पैदा कर सकता है।",
            "action": "प्रति एकड़ 15-20 पीले चिपचिपे ट्रैप लगाएं। नीम का तेल (2 मिली प्रति लीटर) या इमिडाक्लोप्रिड 0.5 मिली प्रति लीटर पानी में मिलाकर छिड़काव करें।",
            "spoken_text": "कीट चेतावनी! फसल पर माहू और सफेद मक्खी का प्रकोप देखा गया है। पीले चिपचिपे कार्ड लगाएं और नीम तेल या इमिडाक्लोप्रिड का छिड़काव करें।"
        },
        "Telugu": {
            "title": "పేనుబంక మరియు తెల్లదోమ ఉధృతి",
            "summary": "రసం పీల్చే పురుగుల ఉధృతి గుర్తించబడింది. ఆకులు ముడుచుకుపోవడం మరియు మసి తెగులు వచ్చే ప్రమాదం ఉంది.",
            "action": "ఎకరాకు 15-20 పసుపు రంగు జిగురు అట్టలను అమర్చండి. వేపనూనె లేదా ఇమిడాక్లోప్రిడ్ 0.5 మిల్లీలీటర్ లీటరు నీటికి కలిపి పిచికారీ చేయండి.",
            "spoken_text": "పురుగుల హెచ్చరిక! పంటపై పేనుబంక మరియు తెల్లదోమ ఆశించాయి. పసుపు జిగురు అట్టలను అమర్చి, వెంటనే వేపనూనె లేదా ఇమిడాక్లోప్రిడ్ పిచికారీ చేయండి."
        },
        "Malayalam": {
            "title": "ഇലപ്പേൻ, വെള്ളീച്ച ബാധ",
            "summary": "നീരൂറ്റിക്കുടിക്കുന്ന പ്രാണികളുടെ സാന്നിധ്യം കണ്ടെത്തി. ഇലകൾ ചുരുളാനും കരിമ്പൻ രോഗം വരാനും സാധ്യതയുണ്ട്.",
            "action": "മഞ്ഞക്കെണികൾ (ഏക്കറിന് 15-20) സ്ഥാപിക്കുക. വേപ്പെണ്ണ മിശ്രിതം അല്ലെങ്കിൽ ഇമിഡാക്ലോപ്രിഡ് 0.5 മില്ലി ഒരു ലിറ്റർ വെള്ളത്തിൽ തളിക്കുക.",
            "spoken_text": "കീടബാധ മുന്നറിയിപ്പ്! വിളയിൽ ഇലപ്പേനും വെള്ളീച്ചയും കണ്ടെത്തിയിരിക്കുന്നു. മഞ്ഞക്കെണികൾ സ്ഥാപിക്കുകയും വേപ്പെണ്ണയോ ഇമിഡാക്ലോപ്രിഡോ തളിക്കുകയും ചെയ്യുക."
        }
    },
    "Leaf Miners / Armyworms": {
        "Tamil": {
            "title": "இலை சுரங்க புழு / படைப்புழு",
            "summary": "இலைகளில் பாம்பு போன்ற வெள்ளை கோடுகள் மற்றும் புழுக்கள் கடித்த துளைகள் உள்ளன.",
            "action": "இயற்கை முறை: பி.டி (Bacillus thuringiensis) 2 கிராம்/லிட்டர் அல்லது வேப்ப எண்ணெய் தெளிக்கவும். இரசாயன முறை: எமாமெக்டின் பென்சோயேட் 0.4 கிராம் அல்லது கோரஜின் 0.4 மில்லி/லிட்டர் தெளிக்கவும்.",
            "spoken_text": "இலை சுரங்க புழுக்கள் அல்லது படைப்புழுக்கள் பயிரை சேதப்படுத்தியுள்ளன. சேதமடைந்த இலைகளை அகற்றி, எமாமெக்டின் பென்சோயேட் அல்லது பேசிலஸ் துரிஞ்சியென்சிஸ் தெளிக்கவும்."
        },
        "English": {
            "title": "Leaf Miners & Armyworm Infestation",
            "summary": "Chewing pests detected. Winding serpentine white trails and irregular margin defoliation reduce photosynthesis.",
            "action": "Handpick heavily mined leaves. Spray Bacillus thuringiensis (Bt) @ 2g/L or Emamectin Benzoate 5% SG @ 0.4g/L in late evening.",
            "spoken_text": "Alert: Leaf miner and caterpillar damage detected with white serpentine trails. Spray Bacillus thuringiensis or Emamectin Benzoate during evening hours."
        },
        "Hindi": {
            "title": "लीफ माइनर और सैनिक कीट का प्रकोप",
            "summary": "पत्तियों पर सफेद टेढ़ी-मेढ़ी सुरंगें और कीड़ों द्वारा पत्तियों को काटने के निशान पाए गए हैं।",
            "action": "प्रभावित पत्तियों को नष्ट करें। शाम के समय बेसिलस थुरिंजिएंसिस (2 ग्राम/लीटर) या एमामेक्टिन बेंजोएट (0.4 ग्राम/लीटर) का छिड़काव करें।",
            "spoken_text": "चेतावनी! लीफ माइनर कीट पत्तियों में सुरंग बनाकर नुकसान पहुंचा रहे हैं। शाम के समय एमामेक्टिन बेंजोएट या जैविक कीटनाशक का छिड़काव करें।"
        },
        "Telugu": {
            "title": "ఆకు తొలుచు పురుగు మరియు లద్దె పురుగు",
            "summary": "ఆకులపై తెల్లని వంకరటింకర సొరంగాలు మరియు అంచులు కొరికిన నష్టం కనిపించింది.",
            "action": "తీవ్రంగా దెబ్బతిన్న ఆకులను తీసివేయండి. బాసిల్లస్ తురింజియెన్సిస్ 2 గ్రాములు లేదా ఎమామెక్టిన్ బెంజోయేట్ 0.4 గ్రాములు లీటరు నీటికి పిచికారీ చేయండి.",
            "spoken_text": "హెచ్చరిక! ఆకు తొలుచు పురుగులు ఆకులపై సొరంగాలు చేస్తున్నాయి. సాయంత్రం వేళలో ఎమామెక్టిన్ బెంజోయేట్ మందును పిచికారీ చేయండి."
        },
        "Malayalam": {
            "title": "ഇല തുരപ്പൻ പുഴു ബാധ",
            "summary": "ഇലകളിൽ വളഞ്ഞുപുളഞ്ഞ വെളുത്ത വരകളും പുഴുക്കൾ കരണ്ടുതിന്ന പാടുകളും കണ്ടെത്തി.",
            "action": "ബാധിച്ച ഇലകൾ നശിപ്പിക്കുക. ബാസിലസ് തുറിഞ്ചിയെൻസിസ് 2 ഗ്രാം അല്ലെങ്കിൽ എമാമെക്റ്റിൻ ബെൻസോയേറ്റ് 0.4 ഗ്രാം ഒരു ലിറ്റർ വെള്ളത്തിൽ കലക്കി തളിക്കുക.",
            "spoken_text": "മുന്നറിയിപ്പ്! ഇല തുരപ്പൻ പുഴുക്കളുടെ ആക്രമണം കണ്ടെത്തിയിരിക്കുന്നു. വൈകുന്നേരങ്ങളിൽ എമാമെക്റ്റിൻ ബെൻസോയേറ്റ് തളിക്കുക."
        }
    },
    "Spider Mites / Thrips": {
        "Tamil": {
            "title": "செம்பேன் / இலைப்பேன் தாக்குதல்",
            "summary": "இலைகளின் அடிப்பகுதியில் சிலந்தி வலைகள், வெளிறிய நுண்ணிய புள்ளிகள் மற்றும் இலை சுருக்கம் உள்ளன.",
            "action": "இயற்கை முறை: நீரை வேகமாக தெளித்து வலைகளை அழிக்கவும். நனையும் கந்தகம் 3 கிராம்/லிட்டர் தெளிக்கவும். இரசாயன முறை: புரோபார்கைட் 2 மில்லி அல்லது ஸ்பைரோமெசிஃபென் 1 மில்லி/லிட்டர் தெளிக்கவும்.",
            "spoken_text": "பயிரில் செம்பேன் அல்லது இலைப்பேன் பரவியுள்ளது. நனையும் கந்தகம் அல்லது புரோபார்கைட் மருந்தை இலைகளின் அடிப்பாகத்தில் நன்கு நனையும்படி தெளிக்கவும்."
        },
        "English": {
            "title": "Spider Mites & Thrips Infestation",
            "summary": "Micro-puncture stippling and web spinning detected on foliage, leading to bronzing and desiccated leaves.",
            "action": "Perform overhead misting to increase canopy moisture. Apply wettable sulfur 80% WP @ 3g/L or Spiromesifen 22.9% SC @ 1 ml/L.",
            "spoken_text": "Pest alert! Spider mites and thrips detected with micro-stippling and webbing. Apply overhead water misting and spray wettable sulfur or Spiromesifen."
        },
        "Hindi": {
            "title": "मकड़ी कीट (माइट्स) और थ्रिप्स का प्रकोप",
            "summary": "पत्तियों पर बारीक धब्बे, जाले और पत्तियों का तांबे जैसा रंग होना देखा गया है।",
            "action": "नमी बढ़ाने के लिए पानी का छिड़काव करें। घुलनशील गंधक 3 ग्राम प्रति लीटर या स्पाइरोमेसिफेन 1 मिली प्रति लीटर का छिड़काव करें।",
            "spoken_text": "कीट चेतावनी! फसल पर लाल मकड़ी कीट और थ्रिप्स का हमला हुआ है। घुलनशील गंधक या स्पाइरोमेसिफेन का तुरंत छिड़काव करें।"
        },
        "Telugu": {
            "title": "నల్లి పురుగులు మరియు తామర పురుగులు",
            "summary": "ఆకులపై సూక్ష్మమైన చుక్కలు, సాలె గూళ్ళు మరియు ఆకులు కాంస్య రంగులోకి మారడం గుర్తించబడింది.",
            "action": "తేమను పెంచడానికి నీటిని తుంపర్లుగా చల్లండి. గంధకం పొడి 3 గ్రాములు లేదా స్పైరోమెసిఫెన్ 1 మిల్లీలీటర్ లీటరు నీటికి పిచికారీ చేయండి.",
            "spoken_text": "హెచ్చరిక! ఆకులపై నల్లి పురుగులు మరియు సాలె గూళ్ళు ఆశించాయి. వెంటనే గంధకం లేదా స్పైరోమెసిఫెన్ మందును పిచికారీ చేయండి."
        },
        "Malayalam": {
            "title": "മൈറ്റുകൾ (ചിലന്തിപ്പേൻ), ഇലപ്പേൻ ബാധ",
            "summary": "ഇലകളിൽ സൂക്ഷ്മ പുള്ളികളും ചിലന്തി വലകളും തവിട്ടുനിറവും കാണപ്പെടുന്നു.",
            "action": "ഇലകളിൽ വെള്ളം തളിച്ച് ഈർപ്പം നിലനിർത്തുക. നനയ്ക്കാവുന്ന ഗന്ധകം 3 ഗ്രാം അല്ലെങ്കിൽ സ്പൈറോമെസിഫെൻ 1 മില്ലി തളിക്കുക.",
            "spoken_text": "മുന്നറിയിപ്പ്! ചിലന്തിപ്പേൻ ബാധ മൂലം ഇലകൾ നശിക്കുന്നു. ഉടൻ തന്നെ ഗന്ധകപ്പൊടിയോ സ്പൈറോമെസിഫെനോ തളിക്കുക."
        }
    }
}


def get_advisory(condition: str, language: str) -> Dict[str, str]:
    """Retrieves localized text advisory in the chosen language."""
    cond_entry = TRANSLATIONS.get(condition, TRANSLATIONS["Healthy Leaf"])
    return cond_entry.get(language, cond_entry["English"])


async def _synthesize_edge_tts(text: str, voice: str, output_path: str):
    import edge_tts
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)


def generate_speech_audio(text: str, language: str) -> bytes:
    """
    Generates neural spoken audio MP3 in the selected language.
    Returns audio bytes for direct in-memory playback.
    """
    voice_info = VOICE_MAPPING.get(language, VOICE_MAPPING["English"])
    voice_name = voice_info["voice"]

    # Hash text + voice to reuse cached audio
    text_hash = hashlib.md5(f"{text}_{voice_name}".encode("utf-8")).hexdigest()
    output_filename = f"voice_{text_hash}.mp3"
    output_path = os.path.join(AUDIO_CACHE_DIR, output_filename)

    if os.path.exists(output_path) and os.path.getsize(output_path) > 500:
        with open(output_path, "rb") as f:
            return f.read()

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_synthesize_edge_tts(text, voice_name, output_path))
        loop.close()

        if os.path.exists(output_path) and os.path.getsize(output_path) > 500:
            with open(output_path, "rb") as f:
                return f.read()
    except Exception as e:
        print(f"Edge-TTS failed: {e}. Attempting offline pyttsx3 fallback...")
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.save_to_file(text, output_path)
            engine.runAndWait()
            if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
                with open(output_path, "rb") as f:
                    return f.read()
        except Exception as pe:
            print(f"pyttsx3 also failed: {pe}")

    return b""
