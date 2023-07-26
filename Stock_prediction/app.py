import base64
from io import BytesIO
from flask import Flask, render_template, request
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.svm import SVR
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        stock_code = request.form['stock_code']
        days = int(request.form['days'])
        
        # Fetch stock information using yfinance
        stock = yf.Ticker(stock_code)
        info = stock.info
        logo_url = info.get('logo_url', '')
        registered_name = info.get('longName', '')
        description = info.get('longBusinessSummary', '')
        print(info)
        
        # Generate stock trend graph for the past 1 month
        today = datetime.now().date()
        start_date_1_month = today - timedelta(days=30)
        history_1_month = stock.history(start=start_date_1_month, end=today)
        
        # Generate stock trend graph for the past 6 months
        start_date_6_months = today - timedelta(days=30*6)
        history_6_months = stock.history(start=start_date_6_months, end=today)
        
        # Generate stock trend graph for the past 12 months
        start_date_12_months = today - timedelta(days=30*12)
        history_12_months = stock.history(start=start_date_12_months, end=today)
        
        # Generate plot for the past 1 month
        plt.figure(figsize=(10, 6))
        plt.plot(history_1_month.index, history_1_month['Close'])
        plt.xticks(rotation=45)
        plt.xticks(fontsize=8)
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.title(f'Stock Price Trend (Past 1 Month) of {registered_name}')
        plt.grid(True)
        plt.savefig('plot_1_month.png')
        
        # Generate plot for the past 6 months
        plt.figure(figsize=(10, 6))
        plt.plot(history_6_months.index, history_6_months['Close'])
        plt.xticks(rotation=45)
        plt.xticks(fontsize=8)
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.title(f'Stock Price Trend (Past 6 Months) of {registered_name}')
        plt.grid(True)
        plt.savefig('plot_6_months.png')
        
        # Generate plot for the past 12 months
        plt.figure(figsize=(10, 6))
        plt.plot(history_12_months.index, history_12_months['Close'])
        plt.xticks(rotation=45)
        plt.xticks(fontsize=8)
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.title(f'Stock Price Trend (Past 12 Months) of {registered_name}')
        plt.grid(True)
        plt.savefig('plot_12_months.png')
        
        # Perform forecasting for the specified number of days
        future_dates = [today + timedelta(days=i) for i in range(1, days+1)]
        X = np.array(range(len(history_6_months))).reshape(-1, 1)
        y = history_6_months['Close'].values
        svr = SVR(kernel='rbf')
        svr.fit(X, y)
        future_prices = svr.predict(np.array(range(len(history_6_months), len(history_6_months)+days)).reshape(-1, 1))
        
        plt.figure(figsize=(10, 6))
        plt.plot(future_dates, future_prices, color='red', linestyle='dashed', label='Predicted Price')
        plt.legend()
        plt.xticks(rotation=45)
        plt.xticks(fontsize=8)
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.title('Stock Price Forecast')
        plt.grid(True)
        plt.savefig('plot_forecast.png')
        
        return render_template('index.html', logo_url=logo_url, registered_name=registered_name,
                               description=description, plot_image_1_month=plot_to_base64('plot_1_month.png'),
                               plot_image_6_months=plot_to_base64('plot_6_months.png'),
                               plot_image_12_months=plot_to_base64('plot_12_months.png'),
                               plot_image_forecast=plot_to_base64('plot_forecast.png'),
                               future_dates=future_dates, future_prices=future_prices)
    
    return render_template('index.html')

def plot_to_base64(image_path):
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    return plot_data

if __name__ == '__main__':
    app.run()
