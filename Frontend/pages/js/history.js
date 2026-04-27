// ==================== HISTORY PAGE ====================

let allHistory = [];

document.addEventListener('DOMContentLoaded', async () => {
    if (!requireAuth()) return;
    await loadHistory();
});

async function loadHistory() {
    try {
        allHistory = await API.get('/history');
        renderHistory(allHistory);
    } catch (error) {
        document.getElementById('historyList').innerHTML =
            `<p class="empty-state">Failed to load history: ${error.message}</p>`;
    }
}

function renderHistory(items) {
    const container = document.getElementById('historyList');

    if (items.length === 0) {
        container.innerHTML = `
            <p class="empty-state">
                No analyses found. 
                <a href="/pages/capture.html">Analyze your first soil sample!</a>
            </p>
        `;
        return;
    }

    container.innerHTML = items.map(item => `
        <div class="history-card" data-soil-type="${item.soil_type}">
            <img src="${item.image_url}" class="history-card-image" 
                 alt="${item.soil_type} soil" loading="lazy">
            <div class="history-card-body">
                <h3>${item.soil_type.charAt(0).toUpperCase() + item.soil_type.slice(1)}</h3>
                <p><strong>Confidence:</strong> ${(item.confidence * 100).toFixed(1)}%</p>
                <p><strong>Crop:</strong> ${item.crop_type}</p>
                <p style="font-size:0.85rem; color:var(--text-muted);">
                    ${new Date(item.created_at).toLocaleDateString('en-US', {
                        year: 'numeric', month: 'long', day: 'numeric',
                        hour: '2-digit', minute: '2-digit'
                    })}
                </p>
            </div>
            <div class="history-card-footer">
                <button class="btn btn-outline" onclick="viewDetails(${item.id})">
                    View Details
                </button>
                <button class="btn btn-danger" onclick="deleteAnalysis(${item.id})">
                    🗑️ Delete
                </button>
            </div>
        </div>
    `).join('');
}

function filterHistory() {
    const filter = document.getElementById('filterSoilType').value;

    if (filter === 'all') {
        renderHistory(allHistory);
    } else {
        const filtered = allHistory.filter(item => item.soil_type === filter);
        renderHistory(filtered);
    }
}

async function deleteAnalysis(id) {
    if (!confirm('Are you sure you want to delete this analysis?')) return;

    try {
        await API.delete(`/history/${id}`);
        allHistory = allHistory.filter(item => item.id !== id);
        renderHistory(allHistory);
    } catch (error) {
        alert('Failed to delete: ' + error.message);
    }
}

function viewDetails(id) {
    const item = allHistory.find(a => a.id === id);
    if (!item) return;

    // Store in sessionStorage and redirect
    sessionStorage.setItem('viewAnalysis', JSON.stringify(item));
    window.location.href = '/pages/capture.html?view=true';
}