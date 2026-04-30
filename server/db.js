const fs = require('fs');
const path = require('path');

const DB_PATH = path.join(__dirname, 'database.json');

// Initialize DB if not exists or empty
const initDB = () => {
    if (!fs.existsSync(DB_PATH)) {
        fs.writeFileSync(DB_PATH, JSON.stringify({ history: [] }, null, 2));
    }
    const data = JSON.parse(fs.readFileSync(DB_PATH, 'utf8'));
    if (!data.history) data.history = [];
    fs.writeFileSync(DB_PATH, JSON.stringify(data, null, 2));
};

initDB();

const getData = () => JSON.parse(fs.readFileSync(DB_PATH, 'utf8'));
const saveData = (data) => fs.writeFileSync(DB_PATH, JSON.stringify(data, null, 2));

const historyDB = {
    getAll: () => {
        const data = getData();
        return data.history.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
    },

    add: (entry) => {
        const data = getData();
        const newEntry = {
            id: Date.now().toString(),
            timestamp: new Date().toISOString(),
            ...entry
        };
        data.history.push(newEntry);
        saveData(data);
        return newEntry;
    },

    delete: (entryId) => {
        const data = getData();
        const originalCount = data.history.length;
        data.history = data.history.filter(h => h.id !== entryId);
        saveData(data);
        return data.history.length < originalCount;
    }
};

module.exports = { historyDB };
