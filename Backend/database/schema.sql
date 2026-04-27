CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    full_name TEXT NOT NULL,
    farm_name TEXT DEFAULT '',
    preferred_crops TEXT DEFAULT '',
    organic_preference INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    image_path TEXT NOT NULL,
    soil_type TEXT NOT NULL,
    confidence REAL NOT NULL,
    properties TEXT NOT NULL,        -- JSON string
    recommendations TEXT NOT NULL,   -- JSON string
    crop_type TEXT DEFAULT 'general',
    notes TEXT DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);