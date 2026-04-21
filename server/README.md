# ⚙️ Music Classification Backend
[![Node.js](https://img.shields.io/badge/Node.js-18.x-339933.svg)](https://nodejs.org/)
[![Express](https://img.shields.io/badge/Express-4.x-000000.svg)](https://expressjs.com/)

The backend orchestration layer for the AI Music Genre Classification system. This service manages file uploads, user persistence, and serves as a bridge to the Python-powered ML engine.

## 🚀 Core Functionality

- **Subprocess Orchestration**: Sophisticated handling of Python scripts using `child_process`. It passes audio file paths to the ML engine and parses JSON results for the frontend.
- **File Management**: Securely handles multi-part audio uploads using `multer`.
- **Virtual Database**: A lightweight, file-based JSON store for managing user classifications and historical data.
- **CORS & Security**: Configured for cross-origin resource sharing with the Vite-based frontend.

## 🛠️ API Endpoints

### Classification
- `POST /api/classify`: Accepts an audio file, triggers individual model analysis (CNN/GMM/RF), and returns formatted results.
- `GET /api/history`: Retrieves the last 20 classification results from the virtual database.

### Auth (Development Mode)
- `POST /api/login`: Mock authentication for session management.

## 📁 Repository Structure

- `index.js`: Main entry point and Express configuration.
- `db.js`: Virtual database handler.
- `auth.js`: Authentication middleware and routes.
- `uploads/`: Temporary storage for tracks pending analysis.

## 🔧 Running Independently

```bash
# Install dependencies
npm install

# Start backend server
node index.js
```

The API will be available at `http://localhost:3000`.

---
© 2026 AI Music Research Group
