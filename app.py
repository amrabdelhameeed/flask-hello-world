from flask import Flask, request, jsonify

app = Flask(__name__)

target_prices = {
    "COMI": 73.57,
    "AAPL": 150.00,
    "TSLA": 700.00,
}

stop_loss_prices = {
    "COMI": 73.55,
    "AAPL": 130.00,
    "TSLA": 650.00,
}

@app.route('/update-prices', methods=['POST'])
def update_prices():
    global target_prices, stop_loss_prices
    data = request.get_json()
    target_prices = data.get('target_prices', target_prices)
    stop_loss_prices = data.get('stop_loss_prices', stop_loss_prices)
    return jsonify({
        "message": "Prices updated successfully",
        "target_prices": target_prices,
        "stop_loss_prices": stop_loss_prices
    })

# if __name__ == "__main__":
#     app.run(host='0.0.0.0', port=8000)
