from flask import Flask, render_template, request
import pickle
import numpy as np

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load vectorizer
with open("count_vector.pkl", "rb") as f:
    vectorizer = pickle.load(f)

app = Flask(__name__)

valid_userid = ['1234','Kajal','Hemant','Suman','Mahadev','Jack','Susan']

@app.route('/')
def view():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend_top5():
    user_name = request.form['User Name']
    X = vectorizer.transform([user_name]) 

    if user_name in valid_userid and request.method == 'POST':
        # Get probability scores for all classes
        proba = model.predict_proba(X)[0]

        # Pick top 5 product indices
        top5_idx = np.argsort(proba)[-5:][::-1]

        # Map indices back to product names (model.classes_ stores labels)
        get_top5 = [model.classes_[i] for i in top5_idx]

        return render_template(
            'index.html',
            row_data=get_top5,
            text='Recommended Products for ' + user_name
        )
    elif user_name not in valid_userid:
        return render_template('index.html', text='No Recommendation found for the user')
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
