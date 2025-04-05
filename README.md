# 💹 MudraVridhiAI

**MudraVridhiAI** is an AI-powered tool that helps analyze and identify **profitable stocks** using real-time financial data, technical indicators, and machine learning. Built with Python and Streamlit, it’s designed to be lightweight, scalable, and beginner-friendly — perfect for finance tech experiments or a future fintech product.

---

## 🚀 What It Does

- Visualizes stock trends using RSI, MACD, and moving averages  
- Predicts stock movement using machine learning models  
- (Optional) Adds sentiment analysis via news/NLP  
- Runs as an interactive web app using Streamlit  
- Ready to deploy on Streamlit Cloud or other platforms

---

### 🔧 Tech Stack

- Python
- Scikit-learn, Pandas, NumPy
- Streamlit
- yFinance / Alpha Vantage API
- Matplotlib / Plotly

---

### ---

### 🔧 Tech Stack

- Python
- Scikit-learn, Pandas, NumPy
- Streamlit
- yFinance / Alpha Vantage API
- Matplotlib / Plotly

---

### 🗂️ Project Structure

MudraVridhiAI/
├── app/
│   ├── main.py              # Streamlit main app
│   └── components/          # Reusable UI components (charts, widgets)
│
├── data/
│   └── raw/                 # Raw downloaded stock data (CSV/JSON)
│   └── processed/           # Cleaned and transformed data
│
├── models/
│   ├── stock_predictor.pkl  # Saved ML models
│   └── training/            # Model training scripts
│
├── notebooks/
│   └── exploration.ipynb    # Jupyter notebooks for EDA/experiments
│
├── utils/
│   ├── indicators.py        # RSI, MACD, etc.
│   └── fetch_data.py        # Stock data fetching and preprocessing
│
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
└── .gitignore               # Files to exclude from version control
