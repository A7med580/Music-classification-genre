const fs = require('fs');
const path = require('path');
const bcrypt = require('bcryptjs');

const DB_PATH = path.join(__dirname, 'database.json');

// Initialize DB if not exists or empty
const initDB = () => {
    if (!fs.existsSync(DB_PATH)) {
        fs.writeFileSync(DB_PATH, JSON.stringify({ users: [], history: [] }, null, 2));
    }
    const data = JSON.parse(fs.readFileSync(DB_PATH, 'utf8'));
    if (!data.users) data.users = [];
    if (!data.history) data.history = [];
    fs.writeFileSync(DB_PATH, JSON.stringify(data, null, 2));
};

initDB();

const getData = () => JSON.parse(fs.readFileSync(DB_PATH, 'utf8'));
const saveData = (data) => fs.writeFileSync(DB_PATH, JSON.stringify(data, null, 2));

const userDB = {
    register: async (username, password) => {
        const data = getData();
        if (data.users.find(u => u.username === username)) return { error: "User already exists" };

        const hashedPassword = await bcrypt.hash(password, 10);
        const newUser = {
            id: Date.now().toString(),
            username,
            password: hashedPassword,
            settings: { theme: 'midnight', notifications: true },
            createdAt: new Date().toISOString()
        };

        data.users.push(newUser);
        saveData(data);
        const { password: _, ...userSafe } = newUser;
        return userSafe;
    },

    login: async (username, password) => {
        const data = getData();
        const user = data.users.find(u => u.username === username);
        if (!user) return null;

        const isValid = await bcrypt.compare(password, user.password);
        if (!isValid) return null;

        const { password: _, ...userSafe } = user;
        return userSafe;
    },

    getUser: (id) => {
        const data = getData();
        const user = data.users.find(u => u.id === id);
        if (!user) return null;
        const { password: _, ...userSafe } = user;
        return userSafe;
    },

    updateSettings: (id, settings) => {
        const data = getData();
        const idx = data.users.findIndex(u => u.id === id);
        if (idx === -1) return null;
        data.users[idx].settings = { ...data.users[idx].settings, ...settings };
        saveData(data);
        return data.users[idx].settings;
    }
};

const historyDB = {
    getForUser: (userId) => {
        const data = getData();
        return data.history.filter(h => h.userId === userId).sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
    },

    add: (userId, entry) => {
        const data = getData();
        const newEntry = {
            id: Date.now().toString(),
            userId,
            timestamp: new Date().toISOString(),
            ...entry
        };
        data.history.push(newEntry);
        saveData(data);
        return newEntry;
    },

    delete: (userId, entryId) => {
        const data = getData();
        const originalCount = data.history.length;
        data.history = data.history.filter(h => !(h.id === entryId && h.userId === userId));
        saveData(data);
        return data.history.length < originalCount;
    }
};

module.exports = { userDB, historyDB };
