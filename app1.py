from flask import Flask, request, render_template
import pandas as pd
import joblib


# Declare a Flask app
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def main():
    
    # If a form is submitted
    if request.method == "POST":
        
        # Unpickle classifier
        pred = joblib.load("predictionfinal.pkl")
        
        # Get values through input bars
        cp = request.form.get("Chest pain")
        bp = request.form.get("BP")
        cholestrol = request.form.get("Cholestrol")
        
        # Put inputs to dataframe
        X = pd.DataFrame([[cp,bp,cholestrol]], columns = ["Chest pain","BP","Cholestrol"])
        
        # Get prediction
        prediction = pred.predict(X)[0]
        
    else:
        prediction = ""
        
    return render_template("website.html", output = prediction)

# Running the app
if __name__ == '__main__':
    app.run(debug = True)