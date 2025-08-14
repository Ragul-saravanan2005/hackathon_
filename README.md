# 🚀 SurveyX - Multi-Modal Survey Data Management System

A comprehensive survey data management and analysis system with AI-powered occupation search capabilities, featuring both web interface and REST API access.

## 🌟 Features

- **📊 Interactive Web Interface**: Built with Gradio for easy data exploration
- **🔍 AI-Powered Occupation Search**: Multilingual job title matching using Sentence Transformers
- **🚀 REST API**: FastAPI-based API with secure authentication
- **🗄️ Oracle Database Integration**: Enterprise-grade data storage
- **📈 Data Analytics**: Search, filter, and export capabilities
- **🌐 Multilingual Support**: Supports multiple languages for occupation search

## 🛠️ Technology Stack

- **Backend**: FastAPI, Oracle Database
- **Frontend**: Gradio
- **AI/ML**: Sentence Transformers, RapidFuzz
- **Data Processing**: Pandas
- **Database**: Oracle 11g with Instant Client

## 📋 Prerequisites

- Python 3.8+
- Oracle Instant Client
- Oracle Database (11g or higher)

## 🚀 Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd project
```

2. Install required packages:
```bash
pip install fastapi gradio pandas sentence-transformers rapidfuzz oracledb uvicorn
```

3. Set up Oracle Instant Client and update the path in the scripts.

4. Configure your Oracle database connection in the scripts.

## 💻 Usage

### Web Interface (Gradio)
```bash
python app.py
```
Access the web interface at the provided URL.

### REST API Server
```bash
python main.py
```
Or use uvicorn:
```bash
uvicorn main:app --reload
```

### Database Connection Test
```bash
python oracle.py
```

## 🔐 API Authentication

The REST API uses API key authentication. Include the key in your requests:
- Header: `X-API-Key: your-api-key`
- Query parameter: `?api_key=your-api-key`

## 📊 API Endpoints

- `GET /` - Get total row count
- `GET /data?limit=10` - Fetch limited records
- `GET /download` - Download complete dataset as CSV
- `GET /search?state=Tamil Nadu&gender=Male` - Filter data
- `GET /predict` - ML prediction (placeholder)

## 🗂️ Data Structure

The system manages survey data with the following fields:
- ID, Occupation, State, District, Gender, Income, Year, NCO Code

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 🙋‍♂️ Support

For questions and support, please open an issue in the GitHub repository.
