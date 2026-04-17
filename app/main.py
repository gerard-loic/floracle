from flask import Flask, render_template, request, jsonify
from prediction import Prediction

app = Flask("Prévision de floraison de la vigne")

prediction = Prediction()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/getprediction', methods=['POST'])
def getprediction():
    data = request.json
    year = int(data["year"])
    latitude = float(data["latitude"])
    longitude = float(data["longitude"])
    altitude = float(data["altitude"])

    result = prediction.makePrediction(year, latitude, longitude, altitude)

    return jsonify({
        'predicted_date': result["date_moyenne"],
        'jour_prevu': result["jour_prevu"],
        'graph': result["graph"]
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
