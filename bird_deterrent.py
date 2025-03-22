from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/distanceApi', methods=['POST'])
def receive_data():
    data = request.json
    distance = data.get("distance", 100)  # Default to 100 cm if missing
    print(f"Received Distance: {distance} cm")

    # If distance < 50 cm, send "attack" response
    if distance < 50:
        response = {"action_type": "attack"}
    else:
        response = {"action_type": "safe"}

    return jsonify(response)

if __name__ == '__main__':
    app.run()  # Ensure 0.0.0.0