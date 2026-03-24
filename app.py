from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model
model = pickle.load(open('house_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    area = float(request.form['area'])
    bedrooms = int(request.form['bedrooms'])

    prediction = model.predict([[area, bedrooms]])

    return render_template('index.html',
                           prediction_text=f"Estimated Price: ₹{int(prediction[0])}")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)