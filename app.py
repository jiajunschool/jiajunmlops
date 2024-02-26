from flask import Flask, request, render_template
import pandas as pd  # Make sure pandas is imported
import os
from pycaret.classification import load_model, predict_model


app = Flask(__name__)

# Load the model using pandas' read_pickle
model_path = "model/final_model_mushroom"
model = load_model(model_path)


@app.route('/', methods=['GET', 'POST'])
def classify():
    if request.method == 'POST':
        try:
            # Collect user input
            user_input = {
                'cap-shape': request.form['cap-shape'],
                'cap-surface': request.form['cap-surface'],
                'cap-color': request.form['cap-color'],
                'bruises': request.form['bruises'],
                'odor': request.form['odor']
            }

            # Convert user input into a DataFrame
            user_input_df = pd.DataFrame([user_input])

            # Make prediction using PyCaret's predict_model (assuming the model is compatible)
            predictions = predict_model(model, data=user_input_df)

            # Extract prediction and score (adjust as per your model's output)
            prediction = predictions['prediction_label'].iloc[0]
            score = predictions['prediction_score'].iloc[0]  # This line might need adjustment

            return render_template('jiajun_classification.html', prediction=prediction, score=score)
        except Exception as e:
            return render_template('jiajun_classification.html', error=str(e))

    return render_template('jiajun_classification.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

