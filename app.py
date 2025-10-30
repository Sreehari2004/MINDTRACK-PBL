from flask import Flask, request, jsonify, render_template, send_file
import numpy as np
import joblib
import os
import json
from datetime import datetime
from openai import OpenAI # Ensure OpenAI library is installed: pip install openai
from flask_cors import CORS
from reportlab.pdfgen import canvas # Using basic canvas API as per the provided code
from io import BytesIO
import traceback # For error details

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app, resources={r"/*": {"origins": "*"}})

# --- Configuration ---u
# !!! --- SECURITY WARNING --- !!!
# Hardcoding API keys is insecure. Use Environment Variables for production.
# Replace placeholder only if you understand the risks.
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY" # <<< REPLACE WITH YOUR KEY
# !!! --- END SECURITY WARNING --- !!!

CONTACT_SUBMISSIONS_DIR = 'contact_submissions'
MODEL_PATH = 'svm_model.pkl'
SCALER_PATH = 'scaler.pkl'
RFE_PATH = 'rfe.pkl'

# --- Initialize OpenAI Client ---
client = None
if OPENAI_API_KEY and OPENAI_API_KEY != "YOUR_OPENAI_API_KEY" and not OPENAI_API_KEY.startswith("sk-"): # Basic check
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        print("OpenAI client initialized.")
    except Exception as e:
        print(f"Failed to initialize OpenAI client: {e}")
else:
     print("Warning: OpenAI API Key missing or placeholder. Chatbot disabled.")

# --- Load ML model, scaler, and RFE ---
def try_load(path):
    try:
        if os.path.exists(path):
            obj = joblib.load(path)
            print(f"{path} loaded successfully.")
            return obj
        else:
            print(f"Warning: File not found at {path}")
            return None
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

model = try_load(MODEL_PATH)
scaler = try_load(SCALER_PATH)
rfe = try_load(RFE_PATH)

if not all([model, scaler, rfe]):
    print("Warning: ML model/scaler/rfe not loaded. Prediction will use basic calculation.")

# --- Ensure submissions directory exists ---
os.makedirs(CONTACT_SUBMISSIONS_DIR, exist_ok=True)

# --- Routes for HTML pages ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/assessment')
def assessment():
    return render_template('assessment.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

# --- Chatbot endpoint ---
@app.route('/chatbot', methods=['POST'])
def chatbot():
    if not client:
        return jsonify({'error': 'Chatbot service unavailable due to configuration.'}), 503
    try:
        data = request.get_json()
        user_message = data.get('message')
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        print(f"Chatbot received: {user_message}")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful mental health counselor named MindTrack. Provide supportive, empathetic and brief responses."},
                {"role": "user", "content": user_message}
            ],
            max_tokens=150,
            temperature=0.7
        )
        reply = response.choices[0].message.content.strip()
        print(f"Chatbot reply: {reply}")
        return jsonify({'reply': reply})
    except Exception as e:
        print(f"!!! Chatbot error: {str(e)} !!!")
        traceback.print_exc()
        return jsonify({'error': f'Chatbot interaction failed: {str(e)}'}), 500

# --- Stress prediction endpoint ---
@app.route('/api/predict', methods=['POST'])
def predict():
    print("\n--- Received request for /api/predict ---")
    try:
        data = request.json
        print(f"Received data: {data}")
        responses = data.get('responses', [])
        print(f"Extracted responses: {responses}")

        # Basic Input Validation
        if not responses or not isinstance(responses, list) or len(responses) != 20:
             print(f"Error: Invalid responses data. Length: {len(responses) if isinstance(responses, list) else 'N/A'}")
             return jsonify({'error': 'Invalid or incomplete responses data'}), 400

        try:
            features_numeric = np.array(responses).astype(float)
            features = features_numeric.reshape(1, -1)
            print(f"Features prepared: shape={features.shape}")
        except ValueError as ve:
            print(f"Error converting responses to numeric array: {ve}")
            traceback.print_exc()
            return jsonify({'error': 'Invalid data in responses (must be numeric)'}), 400

        score = 0.0 # Default score

        # ML Model Prediction
        if model and scaler and rfe:
            print("Attempting ML prediction...")
            try:
                features_scaled = scaler.transform(features)
                features_selected = rfe.transform(features_scaled)
                # --- Adjust Model Output Handling As Needed ---
                # Assuming model.predict gives a value directly usable or mappable to 0-2 score
                stress_level = model.predict(features_selected)[0]
                print(f"Raw model output: {stress_level}")
                # Convert raw output to score (needs specific logic based on your model)
                # Example: if output IS the score
                score = float(stress_level)
                # Example: if output is class label [0, 1, 2, 3]
                # label_to_score_map = { 0: 0.25, 1: 0.75, 2: 1.25, 3: 1.75 }
                # score = label_to_score_map.get(stress_level, 1.0) # Default if unknown label
                score = max(0, min(2, score)) # Clamp to 0-2
                print(f"ML Prediction Score: {score:.2f}")
            except Exception as model_err:
                print(f"!!! ML model prediction step failed: {model_err} !!!")
                traceback.print_exc()
                print("Falling back to basic score calculation.")
                score = calculate_basic_score(numeric_responses) # Use numeric responses
        else:
            print("ML components not loaded. Using basic score calculation.")
            score = calculate_basic_score(numeric_responses) # Use numeric responses

        # Get Recommendations with updated detailed text
        recommendations_data = get_recommendations(score)
        print(f"Generated Recommendations for score {score:.2f}: Level={recommendations_data.get('level')}")

        print("--- Prediction request processed successfully ---")
        return jsonify({
            'stress_score': score,
            'recommendations': recommendations_data
        })

    except Exception as e:
        print(f"!!! Unhandled exception during prediction request: {e} !!!")
        traceback.print_exc()
        return jsonify({'error': f'Prediction failed due to an internal server error.'}), 500

# --- calculate_basic_score Function ---
def calculate_basic_score(numeric_responses):
    # Needs numeric list as input
    try:
        # Simple scaling - adjust max_avg_response based on your questionnaire
        max_avg_response_estimate = 15 # Example value, tune this
        avg_response = sum(numeric_responses) / len(numeric_responses)
        score = (avg_response / max_avg_response_estimate) * 2
        print(f"Basic score calculation: avg={avg_response:.2f}, score={score:.2f}")
        return max(0, min(2, score))
    except Exception as e:
        print(f"Error in basic score calculation: {e}")
        return 0.5 # Default fallback score

# --- get_recommendations Function with DETAILED TEXT ---
def get_recommendations(score):
    """ Provides structured recommendations with DETAILED text descriptions. """
    level = "Not Determined"
    color = "grey" # Simple color name for JSON
    detailed_text = "Your stress assessment results are available. Please review the recommendations below."
    action_plan = ["Review your results with a healthcare professional if you have concerns."]

    if score < 0.5:
        level = "Low"
        color = "green"
        detailed_text = ("**Low Stress:** Your assessment indicates a low level of stress. "
                         "This is a positive result! Maintaining well-being involves "
                         "consistent healthy habits. Focus on preserving this balance.")
        action_plan = [ # Kept the original extensive list for low stress
            "Stay physically active (walking, yoga, gym).", "Ensure adequate sleep (7-9 hours).",
            "Nurture social connections.", "Practice mindfulness or brief meditation.",
            "Eat a balanced diet.", "Stay hydrated.", "Take regular breaks.", "Practice gratitude.",
            "Listen to calming music.", "Avoid overworking.", "Spend time in nature.", "Set realistic goals.",
            "Engage in hobbies.", "Laugh often.", "Limit non-essential screen time.",
            "Maintain support systems.", "Stay organized.", "Moderate caffeine and alcohol.",
            "Practice positive self-talk.", "Use simple breathing exercises.", "Schedule time to unplug."
            ]
    elif score < 1.0:
        level = "Moderate"
        color = "yellow"
        detailed_text = ("**Moderate Stress:** Your assessment suggests a moderate level of stress. "
                         "While likely manageable day-to-day, this level indicates that stress might be noticeable. "
                         "It's beneficial to be proactive and incorporate more stress-reducing strategies into your routine "
                         "to prevent escalation.")
        action_plan = [ # Kept original list
            "Practice Meditation or Mindfulness daily (5-10 mins).", "Try Journaling thoughts and feelings.",
            "Establish consistent Healthy Habits (sleep, diet).", "Engage in regular Moderate Exercise.",
            "Review and improve Time Management skills.", "Prioritize Socializing with supportive people.",
            "Make time for enjoyable Hobbies.", "Use Deep Breathing techniques during tense moments.",
             "Be Mindful of stress triggers throughout your day."
            ]
    elif score < 1.5:
        level = "Moderate to High"
        color = "orange"
        detailed_text = ("**Moderate to High Stress:** Your assessment indicates a moderate-to-high level of stress. "
                         "At this level, stress is likely impacting your well-being, mood, or daily functioning. "
                         "Implementing targeted stress management techniques is strongly recommended to regain balance.")
        action_plan = [ # Kept original list
            "Talk to Someone trusted (friend, family, mentor).", "Break down large tasks into smaller steps.",
            "Increase regular Exercise (consult doctor if needed).", "Practice guided Meditation.",
            "Use Journaling to process difficult emotions.", "Focus on core Healthy Habits (sleep, nutrition).",
            "Improve Time Management and prioritization.", "Seek positive Socializing opportunities.",
            "Dedicate time to relaxing Hobbies.", "Practice Deep Breathing frequently.",
            "Cultivate Mindfulness in daily activities.", "Engage in Reading for pleasure/escape.",
            "Listen to calming Music or nature sounds.", "Explore Aromatherapy (lavender, chamomile).",
            "Take regular Nature Walks.", "Try Progressive Muscle Relaxation (PMR).",
            "Use Positive Affirmations.", "Practice Visualization techniques.",
            "Consider Limiting Caffeine intake.", "Schedule regular short Taking Breaks.",
            "Learn and practice Setting Boundaries.", "Actively start Seeking Support networks."
            ]
    else: # score >= 1.5
        level = "High"
        color = "red"
        detailed_text = ("**High Stress:** Your assessment suggests a high level of stress. "
                         "This level can significantly impact physical and mental health, relationships, and overall quality of life. "
                         "It is important to take immediate action, prioritize your well-being, and seek support.")
        action_plan = [ # Kept original list
            "Seriously consider Seeking professional Help (therapist, counselor, doctor).",
            "Identify and actively Reduce exposure to major Stressors.",
            "Implement immediate grounding/calming techniques.",
            "Prioritize basic self-care: sleep, nutrition, rest.",
            "Reach out to your support system NOW.",
            "Avoid unhealthy coping mechanisms (alcohol, etc.).",
            "If in crisis or feeling overwhelmed, call a helpline immediately (e.g., 91-9820466726)."
            ]

    return {
        "level": level,
        "color": color, # Simple color name for JSON
        "text": detailed_text, # The detailed description
        "action_plan": action_plan
    }


# --- Contact form submission ---
@app.route('/api/submit-contact', methods=['POST'])
def submit_contact():
    # ... (submit_contact logic remains the same) ...
    try:
        data = request.json
        if not all(k in data for k in ['name', 'email', 'message']):
            return jsonify({'error': 'Missing required fields'}), 400
        data['timestamp'] = datetime.now().isoformat()
        filename = f"{CONTACT_SUBMISSIONS_DIR}/contact_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{data['name'].replace(' ', '_')}.json"
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Contact form submission saved to {filename}")
        return jsonify({'success': True})
    except Exception as e:
        print(f"Contact form error: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Contact form submission failed'}), 500

# --- PDF report generation endpoint (Using basic Canvas) ---
@app.route('/download-report', methods=['POST'])
def download_report():
    # This PDF generation part still uses the basic canvas API as per your code
    # It will use the 'level', 'score', and 'action_plan' from the request data
    print("\n--- Received request for /download-report ---")
    try:
        data = request.json
        print(f"Received data for PDF: {data}")
        score = data.get('stress_score')
        # Use recommendations data passed from frontend (which got it from /api/predict)
        recommendations_data = data.get('recommendations', {})
        action_plan = recommendations_data.get('action_plan', [])
        level = recommendations_data.get('level')
        # Note: The detailed 'text' from get_recommendations isn't used by this simple PDF generator

        if score is None or level is None:
             print("Error: Missing score or level in PDF request data.")
             return jsonify({'error': 'Missing score or level for report generation'}), 400

        buffer = BytesIO()
        p = canvas.Canvas(buffer) # Using reportlab.pdfgen.canvas
        p.setFont("Helvetica", 12)

        # Basic PDF Structure (You might want to enhance this later with Platypus)
        p.drawString(100, 800, "🧠 MindTrack Wellness Report")
        p.drawString(100, 780, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        p.drawString(100, 760, f"Assessed Stress Level: {level} (Score: {score:.2f}/2.0)") # Show score
        p.line(100, 755, 500, 755) # Simple separator

        p.drawString(100, 735, "Suggested Practices:")
        y_pos = 715
        line_height = 18 # Reduced spacing slightly
        max_y = 100 # Stop before bottom margin
        for tip in action_plan:
            if y_pos < max_y:
                p.showPage() # Add new page if needed
                p.setFont("Helvetica", 12)
                y_pos = 780 # Reset Y position
                p.drawString(100, y_pos, "(Continued...)")
                y_pos -= line_height * 2

            # Simple text wrapping approximation (might break long words)
            max_chars_per_line = 65
            lines = [tip[i:i+max_chars_per_line] for i in range(0, len(tip), max_chars_per_line)]
            p.drawString(120, y_pos, f"• ") # Bullet point
            first_line = True
            for line in lines:
                p.drawString(130 if first_line else 125, y_pos, line) # Indent text slightly
                y_pos -= line_height * 0.8 # Line spacing within item
                first_line = False
            y_pos -= line_height * 0.4 # Space after item

        # Move helpline lower if space allows, otherwise might be on new page
        if y_pos < 150:
             p.showPage()
             p.setFont("Helvetica", 12)
             y_pos = 780
        p.drawString(100, y_pos - line_height * 2, "📞 Helpline: Call 91-9820466726 for support")

        p.showPage()
        p.save()
        buffer.seek(0)

        print("Basic PDF generation successful.")
        return send_file(buffer, as_attachment=True, download_name="wellness_report.pdf", mimetype='application/pdf')

    except Exception as e:
        print(f"!!! PDF generation error: {e} !!!")
        traceback.print_exc()
        return jsonify({'error': f'Failed to generate report: {str(e)}'}), 500

# --- Health check ---
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'})

# --- Run app ---
if __name__ == '__main__':
    print("Starting Flask development server...")
    app.run(host='0.0.0.0', port=5000, debug=False) # Keep debug=True for development


    