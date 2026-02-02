from flask import Flask, request, render_template, jsonify
import pickle
import os

app = Flask(__name__)

# Pre-load models on startup
model = None
ferti = None

def load_models():
    global model, ferti
    if model is not None and ferti is not None:
        return model, ferti
    
    try:
        # Check if files exist
        if not os.path.exists('classifier.pkl'):
            print("ERROR: classifier.pkl not found!")
            print("Please run 'python retrain.py' to generate the model files.")
            return None, None
        if not os.path.exists('fertilizer.pkl'):
            print("ERROR: fertilizer.pkl not found!")
            print("Please run 'python retrain.py' to generate the model files.")
            return None, None
            
        with open('classifier.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('fertilizer.pkl', 'rb') as f:
            ferti = pickle.load(f)
        print("✓ Models loaded successfully!")
        return model, ferti
    except FileNotFoundError as e:
        print(f"ERROR: Model file not found - {e}")
        print("Please run 'python retrain.py' to generate the model files.")
        return None, None
    except Exception as e:
        print(f"ERROR loading models: {e}")
        print("The model files may be corrupted or incompatible.")
        print("Please run 'python retrain.py' to regenerate the model files.")
        return None, None

# Load models when app starts
print("Loading models...")
load_models()

SOIL_MAP = {
    0: "Black",
    1: "Clayey",
    2: "Loamy",
    3: "Red",
    4: "Sandy",
}

CROP_MAP = {
    0: "Barley",
    1: "Cotton",
    2: "Ground Nuts",
    3: "Maize",
    4: "Millets",
    5: "Oil Seeds",
    6: "Paddy",
    7: "Pulses",
    8: "Sugarcane",
    9: "Tobacco",
    10: "Wheat",
    11: "coffee",
    12: "kidneybeans",
    13: "orange",
    14: "pomegranate",
    15: "rice",
    16: "watermelon",
}

def _level_from_value(v: float, low: float, high: float) -> str:
    if v < low:
        return "Low"
    if v > high:
        return "High"
    return "Medium"

def build_decision_support(
    fertilizer_name: str,
    *,
    temp: float,
    humi: float,
    mois: float,
    soil_id: float,
    crop_id: float,
    nitro: float,
    pota: float,
    phosp: float,
    stage: str | None,
) -> dict:
    fert = (fertilizer_name or "").strip()
    fert_u = fert.upper()

    # Soil health labels from numeric inputs (simple interpretation)
    soil_health = {
        "nitrogen": _level_from_value(nitro, low=30, high=80),
        "phosphorus": _level_from_value(phosp, low=20, high=60),
        "potassium": _level_from_value(pota, low=30, high=80),
    }

    # Dosage + method heuristics (fallback-safe)
    quantity = None
    application_method = "Split application (2 doses) and irrigate lightly after."
    best_time = "Early morning or evening (avoid strong sun)."

    if "UREA" in fert_u:
        quantity = 90
        application_method = "Split into 2–3 doses (broadcast + incorporate)."
    elif "DAP" in fert_u:
        quantity = 80
        application_method = "Basal application near root zone (band placement)."
    elif "POTASH" in fert_u or "MOP" in fert_u or "KCL" in fert_u:
        quantity = 60
        application_method = "Side placement along rows; avoid direct seed contact."
    elif "NPK" in fert_u:
        quantity = 100
        application_method = "Broadcast and incorporate; split if stage is later."
    else:
        quantity = 75

    # Stage adjustments (simple)
    stage_norm = (stage or "Vegetative").strip() or "Vegetative"
    if stage_norm.lower() in {"flowering", "fruiting"} and quantity is not None:
        quantity = int(round(quantity * 0.8))

    # Weather warnings (basic)
    weather_warning = None
    if mois >= 75:
        weather_warning = "Soil moisture is high — avoid heavy fertilizer today (risk of leaching)."
    elif humi >= 85 and temp >= 30:
        weather_warning = "Hot + very humid — apply in early morning; avoid foliar spray at noon."
    elif temp >= 38:
        weather_warning = "Very high temperature — delay application or irrigate before applying."

    # Cost + yield estimate (very rough, rule-based)
    # Using a simple INR/kg estimate based on common fertilizers; fallback for unknown.
    price_per_kg = 32
    if "UREA" in fert_u:
        price_per_kg = 12
    elif "DAP" in fert_u:
        price_per_kg = 28
    elif "POTASH" in fert_u or "MOP" in fert_u:
        price_per_kg = 22
    elif "NPK" in fert_u:
        price_per_kg = 30
    cost_inr = int(round((quantity or 0) * price_per_kg))

    # Yield improvement based on deficiency severity (more deficiency => more potential gain)
    deficiency_score = sum(1 for v in soil_health.values() if v == "Low")
    expected_yield_increase_pct = 8 + deficiency_score * 6  # 8–26%

    soil_name = SOIL_MAP.get(int(soil_id), "Unknown")
    crop_name = CROP_MAP.get(int(crop_id), "Unknown")

    return {
        "quantity_kg_per_ha": quantity,
        "application_method": application_method,
        "best_time": best_time,
        "weather_warning": weather_warning,
        "soil_health": soil_health,
        "cost_inr": cost_inr if cost_inr > 0 else None,
        "expected_yield_increase_pct": expected_yield_increase_pct,
        "soil_name": soil_name,
        "crop_name": crop_name,
        "stage": stage_norm,
    }

@app.route('/')
def home():
    return render_template('plantindex.html')

@app.route('/Model1')
def Model1():
    return render_template('Model1.html')

@app.route('/Detail')
def Detail():
    return render_template('Detail.html')

def build_water_plan(
    *,
    temp: float,
    humi: float,
    soil_id: float,
    crop_id: float,
    stage: str | None,
    fertilizer_type: str | None,
) -> dict:
    """Simple rule-based irrigation guidance."""
    soil_name = SOIL_MAP.get(int(soil_id), "Unknown")
    crop_name = CROP_MAP.get(int(crop_id), "Unknown")
    stage_norm = (stage or "Vegetative").strip() or "Vegetative"
    fert_type = (fertilizer_type or "Balanced NPK").strip() or "Balanced NPK"

    # Climate class
    if temp <= 20:
        climate = "cool"
    elif temp >= 32:
        climate = "hot"
    else:
        climate = "moderate"

    # Base interval in days
    if climate == "cool":
        interval = 8
    elif climate == "hot":
        interval = 4
    else:
        interval = 6

    # Soil adjustment
    soil_factor_note = ""
    if int(soil_id) == 4:  # Sandy
        interval -= 2
        soil_factor_note = "Sandy soils drain quickly, so irrigate more frequently with smaller doses."
    elif int(soil_id) == 1:  # Clayey
        interval += 2
        soil_factor_note = "Clayey soils hold water longer; avoid waterlogging and give deeper but less frequent irrigations."
    elif int(soil_id) == 2:  # Loamy
        soil_factor_note = "Loamy soils are balanced – follow the base schedule."

    # Humidity tweak
    if humi >= 80:
        interval += 1
    elif humi <= 35:
        interval -= 1

    interval = max(2, min(10, interval))

    # Depth per irrigation (mm)
    if int(soil_id) == 4:      # Sandy
        depth = 30
    elif int(soil_id) == 1:    # Clayey
        depth = 50
    else:                      # Others / Loamy
        depth = 40

    if climate == "hot":
        depth += 5
    elif climate == "cool":
        depth -= 5

    # Stage adjustments
    if stage_norm.lower() in {"sowing"}:
        depth = max(25, depth - 5)
    elif stage_norm.lower() in {"flowering", "fruiting"}:
        depth += 5

    # Fertilizer-specific note
    fert_note = ""
    fert_upper = fert_type.upper()
    if any(key in fert_upper for key in ["UREA", "DAP", "NPK"]):
        fert_note = "After applying chemical fertilizers like Urea/DAP/NPK, give a light irrigation within 24 hours to avoid burning and improve uptake."
    elif "ORGANIC" in fert_upper or "COMPOST" in fert_upper:
        fert_note = "With organic fertilizers/compost, keep the soil uniformly moist to help microbes release nutrients."

    notes = []
    if soil_factor_note:
        notes.append(soil_factor_note)
    if fert_note:
        notes.append(fert_note)

    climate_text = f"{climate.capitalize()} conditions with relative humidity around {int(humi)}%."

    return {
        "soil_name": soil_name,
        "crop_name": crop_name,
        "stage": stage_norm,
        "fertilizer_type": fert_type,
        "interval_days": interval,
        "depth_mm": depth,
        "climate_text": climate_text,
        "notes": notes,
    }

@app.route('/Water')
def Water():
    return render_template('Water.html')

@app.route('/water_plan', methods=['POST'])
def water_plan():
    try:
        temp = float(request.form.get('temp'))
        humi = float(request.form.get('humid'))
        soil = float(request.form.get('soil'))
        crop = float(request.form.get('crop'))
        stage = request.form.get('stage', 'Vegetative')
        fert_type = request.form.get('fert_type', 'Balanced NPK')
    except (TypeError, ValueError):
        return render_template('Water.html', error="Please provide numeric values for temperature and humidity.", plan=None)

    plan = build_water_plan(
        temp=temp,
        humi=humi,
        soil_id=soil,
        crop_id=crop,
        stage=stage,
        fertilizer_type=fert_type,
    )
    return render_template('Water.html', plan=plan, error=None)

@app.route('/get_weather', methods=['GET'])
def get_weather():
    """Fetch weather data automatically"""
    import requests
    
    try:
        # Get location from request (latitude and longitude)
        lat = request.args.get('lat', None)
        lon = request.args.get('lon', None)
        
        if not lat or not lon:
            # Try to get location from IP (fallback)
            try:
                ip_response = requests.get('http://ip-api.com/json/', timeout=3)
                ip_data = ip_response.json()
                lat = ip_data.get('lat')
                lon = ip_data.get('lon')
            except:
                # Default location (can be changed)
                lat = 20.5937  # Default to India center
                lon = 78.9629
        
        # Use OpenWeatherMap API (free tier - no key needed for basic usage)
        # Alternative: Using a free weather API that doesn't require key
        try:
            # Using open-meteo.com (free, no API key required)
            weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m&hourly=soil_moisture_0_to_7cm"
            response = requests.get(weather_url, timeout=5)
            data = response.json()
            
            if 'current' in data:
                temperature = round(data['current'].get('temperature_2m', 25))
                humidity = round(data['current'].get('relative_humidity_2m', 60))
                
                # Estimate moisture based on humidity and recent weather
                # Higher humidity = higher moisture, with some variation
                if humidity > 70:
                    moisture = round(humidity * 0.85 + 10)
                elif humidity > 50:
                    moisture = round(humidity * 0.9)
                else:
                    moisture = round(humidity * 0.95 - 5)
                
                # Ensure moisture is within reasonable range (20-80)
                moisture = max(20, min(80, moisture))
                
                return jsonify({
                    'success': True,
                    'temperature': temperature,
                    'humidity': humidity,
                    'moisture': moisture
                })
        except Exception as e:
            print(f"Weather API error: {e}")
        
        # Fallback: Use estimated values based on location
        # Estimate temperature based on latitude
        temp_estimate = max(15, min(40, 30 - (abs(float(lat)) - 20) * 0.5))
        humidity_estimate = 60 + (abs(float(lat)) - 20) * 1.5
        moisture_estimate = round(humidity_estimate * 0.85)
        
        return jsonify({
            'success': True,
            'temperature': round(temp_estimate),
            'humidity': round(humidity_estimate),
            'moisture': moisture_estimate
        })
        
    except Exception as e:
        print(f"Error fetching weather: {e}")
        # Return default values
        return jsonify({
            'success': True,
            'temperature': 25,
            'humidity': 60,
            'moisture': 50
        })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form data
        temp = float(request.form.get('temp'))
        humi = float(request.form.get('humid'))
        mois = float(request.form.get('mois'))
        soil = float(request.form.get('soil'))
        crop = float(request.form.get('crop'))
        stage = request.form.get('stage', 'Vegetative')
        nitro = float(request.form.get('nitro'))
        pota = float(request.form.get('pota'))
        phosp = float(request.form.get('phos'))
        input_values = [temp, humi, mois, soil, crop, nitro, pota, phosp]
    except ValueError:
        return render_template('Model1.html', x='Invalid input. Please provide numeric values for all fields.')

    # Load models
    model, ferti = load_models()
    if model is None or ferti is None:
        error_msg = "Error: Models not found or cannot be loaded. Please run 'python retrain.py' to generate the model files, then restart the application."
        print(error_msg)
        return render_template('Model1.html', x=error_msg)

    try:
        # Predict
        prediction = model.predict([input_values])[0]  # Get predicted class index
        if hasattr(ferti, "classes_"):
            res = ferti.classes_[prediction]  # Map the class index to the fertilizer name
        else:
            res = f"Unknown class index: {prediction}"  # In case the classes_ attribute is missing
        details = build_decision_support(
            str(res),
            temp=temp,
            humi=humi,
            mois=mois,
            soil_id=soil,
            crop_id=crop,
            nitro=nitro,
            pota=pota,
            phosp=phosp,
            stage=stage,
        )
        return render_template('Model1.html', x=res, recommendation=str(res), details=details)
    except Exception as e:
        return render_template('Model1.html', x=f"Error during prediction: {e}")

if __name__ == "__main__":
    app.run(debug=True)
