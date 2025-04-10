from flask import Flask, render_template, request, jsonify
from predict import ChatbotPredictor
import os

app = Flask(__name__)
chatbot = ChatbotPredictor()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400

        # Get response from chatbot
        response = chatbot.predict(user_message)
        
        # Extract the text response
        bot_response = response.get('text', 'I apologize, but I am having trouble understanding.')
        
        # Check if there are product details
        if 'product_details' in response:
            product_info = response['product_details']
            bot_response += f"\n\nProduct Details:\nPrice: {product_info['price']}\n"
            if 'image_url' in product_info:
                bot_response += f"\nImage: {product_info['image_url']}"

        return jsonify({'response': bot_response})

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=True)
