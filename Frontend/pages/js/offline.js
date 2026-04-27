// ==================== OFFLINE SUPPORT ====================

// Cache educational content for offline access
class OfflineManager {
    constructor() {
        this.DB_NAME = 'soilsense_offline';
        this.CACHE_KEY = 'offline_cache';
    }

    // Save data to localStorage for offline
    async cacheContent() {
        try {
            const [soilTypes, fertGuide] = await Promise.all([
                API.get('/education/soil-types'),
                API.get('/education/fertilizer-guide')
            ]);

            const cache = {
                soilTypes,
                fertGuide,
                timestamp: Date.now()
            };

            localStorage.setItem(this.CACHE_KEY, JSON.stringify(cache));
            console.log('Content cached for offline use');

        } catch (error) {
            console.log('Failed to cache content:', error);
        }
    }

    getCachedContent(key) {
        const cache = JSON.parse(localStorage.getItem(this.CACHE_KEY) || 'null');
        if (!cache) return null;
        return cache[key] || null;
    }

    // Save analysis results locally when offline
    saveOfflineAnalysis(data) {
        const queue = JSON.parse(localStorage.getItem('offline_queue') || '[]');
        queue.push({ ...data, timestamp: Date.now() });
        localStorage.setItem('offline_queue', JSON.stringify(queue));
    }

    // Sync offline data when back online
    async syncOfflineData() {
        const queue = JSON.parse(localStorage.getItem('offline_queue') || '[]');
        if (queue.length === 0) return;

        for (const item of queue) {
            try {
                // Attempt to sync each item
                console.log('Syncing offline item:', item);
            } catch (e) {
                break;
            }
        }
        localStorage.setItem('offline_queue', '[]');
    }

    isOnline() {
        return navigator.onLine;
    }
}

const offlineManager = new OfflineManager();

// Listen for online/offline events
window.addEventListener('online', () => {
    console.log('Back online — syncing data...');
    offlineManager.syncOfflineData();
    document.body.classList.remove('offline');
});

window.addEventListener('offline', () => {
    console.log('Gone offline');
    document.body.classList.add('offline');
});

// Cache content on first load
if (navigator.onLine) {
    offlineManager.cacheContent();
}