from flask import Flask, render_template, request, jsonify
from predict import predict_flood

app = Flask(__name__)

# ----------------------------
# Routes for frontend pages
# ----------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/floodPrediction.html')
def flood_prediction_page():
    return render_template('floodPrediction.html')

@app.route('/index2.html')
def case_study():
    return render_template('index2.html')

# ----------------------------
# API endpoint for predictions
# ----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    village = data.get('village')
    try:
        result = predict_flood(village)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ----------------------------
# Run server
# ----------------------------
if __name__ == '__main__':
    app.run(debug=True)
