import pandas as pd
import json
from flask import Flask, render_template, request

app = Flask(__name__)

with open('../Results/dev_with_results.json', 'r') as f:
    val_data = pd.DataFrame(json.load(f))

# @app.route('/')
# def index():
#     # pass the DataFrame to the template
#     return render_template('index.html', data=val_data.to_dict(orient='records'))

@app.route("/")
def index():
    return render_template("index.html", data=val_data.to_dict('records'))


# Define a Flask route to handle form submission
@app.route("/", methods=["POST"])
def submit():
    selected_entry = request.form.get("entry")
    return render_template("index.html", data=val_data, selected_entry=selected_entry)


if __name__ == '__main__':
    app.run()