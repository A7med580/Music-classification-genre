const express = require('express');
const cors = require('cors');
const multer = require('multer');
const path = require('path');
const { spawn } = require('child_process');
const { historyDB } = require('./db');
const fs = require('fs');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// Configure Multer for file uploads
const upload = multer({
  dest: path.join(__dirname, '../uploads/'),
  limits: { fileSize: 10 * 1024 * 1024 } // 10MB limit
});

// Ensure uploads directory exists
if (!fs.existsSync(path.join(__dirname, '../uploads'))) {
  fs.mkdirSync(path.join(__dirname, '../uploads'));
}

// --- CLASSIFICATION ROUTES ---

app.get('/api/history', (req, res) => {
  const history = historyDB.getAll();
  res.json(history);
});

app.delete('/api/history/:id', (req, res) => {
  const success = historyDB.delete(req.params.id);
  res.json({ success });
});

app.post('/api/classify', upload.single('audio'), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No audio file provided' });
  }

  const filePath = req.file.path;
  const originalName = req.file.originalname;
  const modelName = req.body.model || 'CNN';

  // Call Python script
  const venvPath = path.join(__dirname, '../venv/bin/python3');
  const pythonCmd = fs.existsSync(venvPath) ? venvPath : 'python3';
  const pythonScript = path.join(__dirname, '../ml/predict.py');
  
  console.log(`Running classification with: ${pythonCmd} --model ${modelName}`);
  const pythonProcess = spawn(pythonCmd, [pythonScript, '--file', filePath, '--model', modelName]);

  let resultData = '';
  let errorData = '';

  pythonProcess.stdout.on('data', (data) => {
    resultData += data.toString();
  });

  pythonProcess.stderr.on('data', (data) => {
    errorData += data.toString();
  });

  pythonProcess.on('close', (code) => {
    // Clean up uploaded file
    fs.unlink(filePath, (err) => {
      if (err) console.error("Error deleting temp file:", err);
    });

    console.log(`Classification result: ${resultData.trim()}`);
    if (errorData) console.error(`Classification error: ${errorData}`);

    if (code !== 0) {
      return res.status(500).json({ error: 'Classification failed', details: errorData });
    }

    try {
      // Find the last valid JSON output (in case there are tf warnings printed)
      const outputLines = resultData.trim().split('\n');
      let prediction = null;
      for (let i = outputLines.length - 1; i >= 0; i--) {
          try {
              prediction = JSON.parse(outputLines[i]);
              break;
          } catch(e) {}
      }

      if (!prediction) {
          throw new Error("Could not find JSON output");
      }

      // Save to DB
      const entry = historyDB.add({
        filename: originalName,
        genre: prediction.genre,
        genre_confidence: prediction.genre_confidence,
        emotion: prediction.emotion,
        emotion_confidence: prediction.emotion_confidence,
        model_used: prediction.model_used
      });

      res.json(entry);
    } catch (e) {
      console.error("Error parsing Python output:", e);
      res.status(500).json({ error: 'Invalid response from ML model', raw: resultData });
    }
  });
});

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
