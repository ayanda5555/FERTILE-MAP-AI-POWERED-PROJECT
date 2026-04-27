// ==================== EDUCATIONAL CONTENT ====================

document.addEventListener('DOMContentLoaded', () => {
    loadSoilTypes();
    loadFertilizerGuide();
});

function switchTab(tabId, button) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));

    // Show selected
    document.getElementById(tabId).classList.add('active');
    button.classList.add('active');
}

async function loadSoilTypes() {
    try {
        const data = await API.get('/education/soil-types');
        const container = document.getElementById('soilTypesContent');

        container.innerHTML = Object.entries(data).map(([key, soil]) => `
            <div class="education-card">
                <div class="education-card-header">
                    <h3>${soil.title}</h3>
                </div>
                <div class="education-card-body">
                    <p>${soil.description}</p>

                    <h4 style="margin-top:1rem;">Characteristics:</h4>
                    <ul>
                        ${soil.characteristics.map(c => `<li>${c}</li>`).join('')}
                    </ul>

                    <h4 style="margin-top:1rem;">Best Crops:</h4>
                    <div style="display:flex;flex-wrap:wrap;gap:0.5rem;margin-top:0.5rem;">
                        ${soil.best_crops.map(c => `
                            <span style="background:#E8F5E9;padding:0.3rem 0.8rem;
                                         border-radius:20px;font-size:0.85rem;">
                                ${c}
                            </span>
                        `).join('')}
                    </div>
                </div>
            </div>
        `).join('');

    } catch (error) {
        console.error('Failed to load soil types:', error);
    }
}

async function loadFertilizerGuide() {
    try {
        const data = await API.get('/education/fertilizer-guide');
        const container = document.getElementById('fertilizerGuideContent');

        let html = '';

        // NPK Guide
        const npk = data.npk_explained;
        html += `
            <div class="education-card" style="margin-bottom:1.5rem;">
                <div class="education-card-header">
                    <h3>${npk.title}</h3>
                </div>
                <div class="education-card-body">
                    <p>${npk.content}</p>
                    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:1rem;margin-top:1rem;">
                        ${Object.values(npk.details).map(nutrient => `
                            <div style="background:#F1F8E9;padding:1rem;border-radius:8px;">
                                <h4 style="color:var(--primary);">${nutrient.symbol} — ${nutrient.role}</h4>
                                <p><strong>Deficiency:</strong> ${nutrient.deficiency_signs}</p>
                                <p><strong>Excess:</strong> ${nutrient.excess_signs}</p>
                            </div>
                        `).join('')}
                    </div>
                </div>
            </div>
        `;

        // Application Methods
        const methods = data.application_methods;
        html += `
            <div class="education-card" style="margin-bottom:1.5rem;">
                <div class="education-card-header">
                    <h3>${methods.title}</h3>
                </div>
                <div class="education-card-body">
                    ${methods.methods.map(m => `
                        <div style="border-left:3px solid var(--primary);padding:0.8rem 1rem;margin-bottom:0.8rem;">
                            <h4>${m.name}</h4>
                            <p>${m.description}</p>
                            <p style="color:var(--text-muted);font-size:0.9rem;">Best for: ${m.best_for}</p>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;

        // Organic vs Synthetic
        const ovs = data.organic_vs_synthetic;
        html += `
            <div class="education-card">
                <div class="education-card-header">
                    <h3>${ovs.title}</h3>
                </div>
                <div class="education-card-body">
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;">
                        <div>
                            <h4 style="color:var(--primary);">🌿 Organic</h4>
                            <p><strong>Pros:</strong></p>
                            <ul>${ovs.organic.pros.map(p => `<li>${p}</li>`).join('')}</ul>
                            <p><strong>Cons:</strong></p>
                            <ul>${ovs.organic.cons.map(c => `<li>${c}</li>`).join('')}</ul>
                        </div>
                        <div>
                            <h4 style="color:var(--secondary);">⚗️ Synthetic</h4>
                            <p><strong>Pros:</strong></p>
                            <ul>${ovs.synthetic.pros.map(p => `<li>${p}</li>`).join('')}</ul>
                            <p><strong>Cons:</strong></p>
                            <ul>${ovs.synthetic.cons.map(c => `<li>${c}</li>`).join('')}</ul>
                        </div>
                    </div>
                </div>
            </div>
        `;

        container.innerHTML = html;

    } catch (error) {
        console.error('Failed to load fertilizer guide:', error);
    }
}