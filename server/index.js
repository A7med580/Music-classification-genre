const express = require('express');
const cors = require('cors');
const multer = require('multer');
const path = require('path');
const { spawn } = require('child_process');
const { userDB, historyDB } = require('./db');
const { authMiddleware, JWT_SECRET } = require('./auth');
const jwt = require('jsonwebtoken');
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

// --- AUTH ROUTES ---

app.post('/api/auth/register', async (req, res) => {
  const { username, password } = req.body;
  if (!username || !password) return res.status(400).json({ error: 'Missing credentials' });

  const result = await userDB.register(username, password);
  if (result.error) return res.status(400).json(result);

  res.status(201).json(result);
});

app.post('/api/auth/login', async (req, res) => {
  const { username, password } = req.body;
  const user = await userDB.login(username, password);

  if (!user) return res.status(401).json({ error: 'Invalid username or password' });

  const token = jwt.sign({ id: user.id, username: user.username }, JWT_SECRET, { expiresIn: '24h' });
  res.json({ user, token });
});

app.get('/api/auth/me', authMiddleware, (req, res) => {
  const user = userDB.getUser(req.userId);
  if (!user) return res.status(404).json({ error: 'User not found' });
  res.json(user);
});

app.put('/api/users/settings', authMiddleware, (req, res) => {
  const settings = userDB.updateSettings(req.userId, req.body);
  res.json(settings);
});

// --- CLASSIFICATION ROUTES ---

app.get('/api/history', authMiddleware, (req, res) => {
  const history = historyDB.getForUser(req.userId);
  res.json(history);
});

app.delete('/api/history/:id', authMiddleware, (req, res) => {
  const success = historyDB.delete(req.userId, req.params.id);
  res.json({ success });
});

app.post('/api/classify', authMiddleware, upload.single('audio'), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No audio file provided' });
  }

  const filePath = req.file.path;
  const originalName = req.file.originalname;

  // Call Python script
  const pythonScript = path.join(__dirname, '../ml/predict.py');
  const pythonProcess = spawn('python3', [pythonScript, filePath]);

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

    if (code !== 0) {
      return res.status(500).json({ error: 'Classification failed', details: errorData });
    }

    try {
      const prediction = JSON.parse(resultData.trim());

      // Save to DB for specific user
      const entry = historyDB.add(req.userId, {
        filename: originalName,
        genre: prediction.genre,
        confidence: prediction.confidence
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
