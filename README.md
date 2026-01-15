# AI Music Genre Classification

A modern, high-performance web application that classifies music genres using Machine Learning. This project features a React-based "Premium" interface, a Node.js backend, and a Python-powered ML engine.

## 🚀 Overview

This project uses a **Hybrid Architecture** to leverage the best tools for each job:

1.  **Frontend (React + Vite)**:
    - Provides a modern, responsive, and polished user interface.
    - Employs **Glassmorphism** design principles for a premium feel.
    - Communicates with the backend via a REST API.

2.  **Backend (Node.js + Express)**:
    - Serves as the orchestrator.
    - Manages file uploads and user authentication.
    - Handles data persistence via a file-based "Virtual Database".
    - Executes Python ML scripts as subprocesses for real-time classification.

3.  **Machine Learning (Python)**:
    - Pure logic layer focused on signal processing.
    - Uses `Librosa` for advanced audio feature extraction (MFCCs, Spectral features).
    - Utilizes a `Random Forest` classifier trained on the GTZAN dataset to predict genres.

### 🗄️ Virtual Database (Persistence)
Instead of requiring a heavy database setup like MongoDB or PostgreSQL, we implemented a **Virtual Database** using a file-based approach (`server/database.json`). This ensures history is preserved across sessions without any installation overhead.

---

## 📁 Project Structure

- `client/`: React/Vite source code (Frontend).
- `server/`: Node.js Express API and DB logic (Backend).
- `ml/`: Python scripts for model training (`train_model.py`) and prediction (`predict.py`).
- `Data/`: Processed dataset used for training.
- `research/`: Archived original notebooks and legacy files.

---

## 🛠️ Installation & Setup

### Prerequisites
- **Node.js**: [Download](https://nodejs.org/en/) (v16 or higher)
- **Python**: [Download](https://www.python.org/downloads/) (v3.8 or higher)
- **pip**: Python package manager.

### First-Time Setup

1. **Install Dependencies**:
   Open your terminal in the project root and run:
   ```bash
   npm install && pip install -r ml/requirements.txt
   ```

2. **Train the AI Model**:
   This step is crucial to generate the `model.pkl` file needed for predictions:
   ```bash
   npm run setup
   ```

---

## 🚦 Running the Application

To start both the Frontend and Backend simultaneously, run:

```bash
npm start
```

- **Backend**: Starts on port `3000`.
- **Frontend**: Starts on port `5173`.
- **Access**: Open your browser to [http://localhost:5173](http://localhost:5173).

---

## 🔧 Troubleshooting

- **Python Command**: If you encounter errors, ensure `python3` is in your PATH. You may need to adjust `server/index.js` to use `python` or `python3` depending on your environment.
- **Port Conflict**: If port 3000 or 5173 is already in use, close the conflicting process and try again.
