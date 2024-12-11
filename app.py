from flask import Flask, render_template, request, jsonify
from flask_mysqldb import MySQL
import matplotlib.pyplot as plt
import io
import base64
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
app.config.from_pyfile('config.py')
mysql = MySQL(app)

# Homepage
@app.route('/')
def index():
    return render_template('index.html')

# Add Expense
@app.route('/add_expense', methods=['POST'])
def add_expense():
    date = request.form['date']
    category = request.form['category']
    amount = float(request.form['amount'])

    # Check if amount exceeds 1k limit
    if amount > 1000:
        return "Error: Amount exceeds ₹1000 limit!", 400

    cursor = mysql.connection.cursor()
    cursor.execute("INSERT INTO expenses (date, category, amount) VALUES (%s, %s, %s)", (date, category, amount))
    mysql.connection.commit()
    cursor.close()
    return "Expense added successfully!"

# View Report
@app.route('/report')
def report():
    cursor = mysql.connection.cursor()
    cursor.execute("SELECT date, category, amount FROM expenses ORDER BY date")
    data = cursor.fetchall()
    cursor.close()

    if not data:
        return "No expenses recorded yet!"

    df = pd.DataFrame(data, columns=['date', 'category', 'amount'])
    df['date'] = pd.to_datetime(df['date'])

    # Individual Category Graphs
    category_plots = {}
    for category in df['category'].unique():
        category_data = df[df['category'] == category]
        plt.figure(figsize=(10, 6))
        plt.plot(category_data['date'], category_data['amount'], marker='o', label=category)
        plt.xticks(rotation=30)
        plt.ylim(0, 1000)
        plt.xlabel('Date')
        plt.ylabel('Amount')
        plt.title(f'{category} Expense Trend')
        plt.legend()

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        category_plots[category] = base64.b64encode(img.getvalue()).decode()
        plt.close()

    # Combined Graph
    plt.figure(figsize=(10, 6))
    for category in df['category'].unique():
        category_data = df[df['category'] == category]
        plt.plot(category_data['date'], category_data['amount'], marker='o', label=category)
    plt.xticks(rotation=30)
    plt.ylim(0, 1000)
    plt.xlabel('Date')
    plt.ylabel('Amount')
    plt.title('Combined Expense Trend')
    plt.legend()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    combined_plot = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return render_template('report.html', category_plots=category_plots, combined_plot=combined_plot)

# Predict Future Expenses
@app.route('/predict', methods=['GET'])
def predict():
    cursor = mysql.connection.cursor()
    cursor.execute("SELECT date, category, amount FROM expenses")
    data = cursor.fetchall()
    cursor.close()

    if not data:
        return jsonify({'error': 'No data available for predictions.'})

    df = pd.DataFrame(data, columns=['date', 'category', 'amount'])
    df['date'] = pd.to_datetime(df['date'])
    df['days'] = (df['date'] - df['date'].min()).dt.days

    predictions = {}
    for category in df['category'].unique():
        category_data = df[df['category'] == category]
        X = category_data[['days']]
        y = category_data['amount']

        if len(X) < 2:
            predictions[category] = "Not enough data to predict."
            continue

        model = LinearRegression()
        model.fit(X, y)

        future_days = np.array([[category_data['days'].max() + i] for i in range(1, 31)])
        future_dates = [category_data['date'].max() + pd.Timedelta(days=i) for i in range(1, 31)]
        future_amounts = model.predict(future_days)

        plt.figure(figsize=(10, 6))
        plt.plot(category_data['date'], category_data['amount'], marker='o', label='Historical')
        plt.plot(future_dates, future_amounts, linestyle='--', color='red', label='Predicted')
        plt.xticks(rotation=30)
        plt.ylim(0, 1000)
        plt.xlabel('Date')
        plt.ylabel('Amount')
        plt.title(f'{category} Predicted Expenses')
        plt.legend()

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        predictions[category] = base64.b64encode(img.getvalue()).decode()
        plt.close()

    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)
