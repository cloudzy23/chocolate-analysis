import pandas as pd
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load dataset
file_path = 'C:/Users/govib/Downloads/Chocolate Sales.csv'
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    tk.messagebox.showerror("Error", "File not found")
    exit(1)

# Data Preprocessing
df['Amount'] = df['Amount'].replace({'\$': '', ',': ''}, regex=True).astype(float)
df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%y', errors='coerce')
df['Year'] = pd.to_datetime(df['Date']).dt.year
df['Month'] = pd.to_datetime(df['Date']).dt.month

# Column mappings
product_col, country_col, employee_col = 'Product', 'Country', 'Sales Person'
quantity_col = 'Boxes Shipped' if 'Boxes Shipped' in df.columns else None
salary_col = 'Amount' if 'Amount' in df.columns else None

# Create main app window
root = tk.Tk()
root.title("NimbusAI by Sarvesh")
root.geometry("900x800")

# Dropdown menu options
options = [
    "Most Popular Chocolate Bar",
    "Most Popular Chocolate Bar by Country",
    "Highest Selling Chocolate Bar",
    "Highest Selling Chocolate Bar by Country",
    "Predict Future Sales",
    "Advanced Regression Model",
    "Time Series Forecasting for Sales"
]

selected_option = tk.StringVar()
selected_option.set(options[0])

# Function to handle user selection
def analyze_data():
    choice = selected_option.get()

    # Clear previous result in the Text widget
    result_text.delete(1.0, tk.END)
    
    # Clear previous plot
    for widget in plot_frame.winfo_children():
        widget.destroy()

    if choice == "Most Popular Chocolate Bar":
        result = df[product_col].value_counts().idxmax()
        result_text.insert(tk.END, f"Most popular chocolate bar: {result}")

        # Create a bar plot for most popular chocolate bar
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x=product_col)
        plt.title("Most Popular Chocolate Bar")
        plt.xticks(rotation=45)
        plot_canvas = FigureCanvasTkAgg(plt.gcf(), plot_frame)
        plot_canvas.get_tk_widget().pack()
        plot_canvas.draw()

    elif choice == "Most Popular Chocolate Bar by Country":
        result = df.groupby(country_col)[product_col].agg(lambda x: x.value_counts().idxmax())
        result_text.insert(tk.END, f"Most popular chocolate bars by country:\n{result.to_string()}")

        # Create a bar plot for most popular chocolate bar by country
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x=product_col, hue=country_col)
        plt.title("Most Popular Chocolate Bar by Country")
        plt.xticks(rotation=45)
        plot_canvas = FigureCanvasTkAgg(plt.gcf(), plot_frame)
        plot_canvas.get_tk_widget().pack()
        plot_canvas.draw()

    elif choice == "Predict Future Sales":
        # Linear Regression for future sales prediction
        df['Year'] = pd.to_datetime(df['Date']).dt.year
        X = df[['Year']]
        y = df['Amount']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)

        future_sales = model.predict([[2026]])
        result_text.insert(tk.END, f"Predicted Sales in 2026: {future_sales[0]:.2f}")

        # Create a plot for predicted sales over time
        plt.figure(figsize=(10, 6))
        sns.lineplot(x='Year', y='Amount', data=df, marker='o', label='Actual Sales')
        plt.plot([2026], [future_sales[0]], 'ro', label='Predicted Sales')
        plt.title("Sales Over Time with Prediction for 2026")
        plt.xlabel("Year")
        plt.ylabel("Sales Amount")
        plt.legend()
        plot_canvas = FigureCanvasTkAgg(plt.gcf(), plot_frame)
        plot_canvas.get_tk_widget().pack()
        plot_canvas.draw()

    elif choice == "Advanced Regression Model":
        # Using Random Forest Regressor for better prediction
        df['Month'] = df['Date'].dt.month
        X = df[['Year', 'Month', 'Boxes Shipped']].dropna()
        y = df['Amount'].dropna()

        # Scaling features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)

        y_pred = rf_model.predict(X_test)
        result_text.insert(tk.END, f"Random Forest Prediction for test set: {y_pred[:5]}")

        # Plotting Feature Importance
        plt.figure(figsize=(10, 6))
        feature_importances = rf_model.feature_importances_
        sns.barplot(x=X.columns, y=feature_importances)
        plt.title("Feature Importance in Predicting Sales")
        plot_canvas = FigureCanvasTkAgg(plt.gcf(), plot_frame)
        plot_canvas.get_tk_widget().pack()
        plot_canvas.draw()

    elif choice == "Time Series Forecasting for Sales":
        # Time Series Forecasting using Holt-Winters Exponential Smoothing
        df_monthly = df.groupby(['Year', 'Month'])['Amount'].sum().reset_index()
        df_monthly['Date'] = pd.to_datetime(df_monthly[['Year', 'Month']].assign(DAY=1))
        model = ExponentialSmoothing(df_monthly['Amount'], trend='add', seasonal='add', seasonal_periods=12)
        model_fit = model.fit()

        forecast = model_fit.forecast(12)
        result_text.insert(tk.END, f"Forecasted Sales for Next Year: {forecast.values}")

        # Plotting the forecasted data
        plt.figure(figsize=(10, 6))
        plt.plot(df_monthly['Date'], df_monthly['Amount'], label='Actual Sales')
        plt.plot(pd.date_range(df_monthly['Date'].max(), periods=13, freq='M')[1:], forecast, label='Forecasted Sales', color='red')
        plt.title("Sales Forecast Using Holt-Winters Exponential Smoothing")
        plt.xlabel("Date")
        plt.ylabel("Sales Amount")
        plt.legend()
        plot_canvas = FigureCanvasTkAgg(plt.gcf(), plot_frame)
        plot_canvas.get_tk_widget().pack()
        plot_canvas.draw()

# Create GUI components
frame = tk.Frame(root)
frame.pack(pady=20)

label = tk.Label(frame, text="Select an option:", font=("Arial", 12))
label.pack(side="left", padx=10)

dropdown = ttk.Combobox(frame, textvariable=selected_option, values=options, state="readonly", width=40)
dropdown.pack(side="left", padx=10)

analyze_button = tk.Button(frame, text="Analyze", command=analyze_data, font=("Arial", 12), bg="lightblue")
analyze_button.pack(pady=10)

# Text widget to display results
result_text = tk.Text(root, height=10, width=100)
result_text.pack(padx=20, pady=10)
result_text.insert(tk.END, "Select an option and click Analyze to see results...\n")

# Frame for displaying graphs
plot_frame = tk.Frame(root)
plot_frame.pack(pady=10)

# Run the application
root.mainloop()
