// ==================== DASHBOARD CHARTS ====================

async function loadDashboard() {
    if (!requireAuth()) return;

    const user = JSON.parse(localStorage.getItem('user') || '{}');
    document.getElementById('userName').textContent = user.full_name || 'User';

    try {
        const stats = await API.get('/stats');

        // Update stat cards
        document.getElementById('totalAnalyses').textContent = stats.total_analyses;
        document.getElementById('avgConfidence').textContent =
            (stats.average_confidence * 100).toFixed(1) + '%';

        // Find most common soil type
        if (stats.soil_distribution.length > 0) {
            const topSoil = stats.soil_distribution.reduce(
                (max, item) => item.count > max.count ? item : max
            );
            document.getElementById('topSoilType').textContent =
                topSoil.soil_type.charAt(0).toUpperCase() + topSoil.soil_type.slice(1);
        }

        // Draw charts
        drawDistributionChart(stats.soil_distribution);
        drawConfidenceTimeline(stats.recent_analyses);

        // Load recent analyses
        loadRecentAnalyses();

    } catch (error) {
        console.error('Failed to load dashboard:', error);
    }
}

function drawDistributionChart(distribution) {
    const ctx = document.getElementById('soilDistributionChart');
    if (!ctx) return;

    const soilColors = {
        chalky: '#D5D8DC',
        clay: '#D35400',
        loamy: '#8B4513',
        peaty: '#1C1C1C',
        sandy: '#F4D03F',
        silty: '#85929E'
    };

    if (distribution.length === 0) {
        ctx.parentElement.innerHTML += '<p class="empty-state">No data yet</p>';
        return;
    }

    new Chart(ctx, {
        type: 'pie',
        data: {
            labels: distribution.map(d =>
                d.soil_type.charAt(0).toUpperCase() + d.soil_type.slice(1)
            ),
            datasets: [{
                data: distribution.map(d => d.count),
                backgroundColor: distribution.map(d => soilColors[d.soil_type] || '#999'),
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { position: 'bottom' }
            }
        }
    });
}

function drawConfidenceTimeline(analyses) {
    const ctx = document.getElementById('confidenceChart');
    if (!ctx) return;

    if (analyses.length === 0) {
        ctx.parentElement.innerHTML += '<p class="empty-state">No data yet</p>';
        return;
    }

    const reversed = [...analyses].reverse();

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: reversed.map(a => {
                const date = new Date(a.date);
                return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
            }),
            datasets: [{
                label: 'Confidence %',
                data: reversed.map(a => (a.confidence * 100).toFixed(1)),
                borderColor: '#2E7D32',
                backgroundColor: 'rgba(46, 125, 50, 0.1)',
                fill: true,
                tension: 0.4,
                pointRadius: 5,
                pointBackgroundColor: '#2E7D32'
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: { callback: v => v + '%' }
                }
            }
        }
    });
}

async function loadRecentAnalyses() {
    try {
        const history = await API.get('/history');
        const container = document.getElementById('recentAnalyses');

        if (history.length === 0) {
            container.innerHTML = `
                <p class="empty-state">
                    No analyses yet. 
                    <a href="/pages/capture.html">Start your first analysis!</a>
                </p>
            `;
            return;
        }

        // Show last 6
        container.innerHTML = history.slice(0, 6).map(item => `
            <div class="history-card">
                <img src="${item.image_url}" class="history-card-image" 
                     alt="${item.soil_type} soil" loading="lazy">
                <div class="history-card-body">
                    <h3>${item.soil_type}</h3>
                    <p>${(item.confidence * 100).toFixed(1)}% confidence</p>
                    <p style="color: var(--text-muted); font-size: 0.85rem;">
                        ${new Date(item.created_at).toLocaleDateString()}
                    </p>
                </div>
            </div>
        `).join('');

    } catch (error) {
        console.error('Failed to load recent analyses:', error);
    }
}