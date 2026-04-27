// ==================== DISPLAY ANALYSIS RESULTS ====================

function displayResults(data) {
    const { prediction, recommendations, image_url } = data;

    const resultsSection = document.getElementById('resultsSection');
    resultsSection.style.display = 'block';

    // Soil Type
    document.getElementById('resultSoilType').textContent = prediction.soil_type;

    // Confidence
    const confidencePercent = prediction.confidence_percent;
    document.getElementById('confidenceFill').style.width = `${confidencePercent}%`;
    document.getElementById('confidenceText').textContent = `${confidencePercent}% confidence`;

    // Image
    document.getElementById('resultImage').src = image_url;

    // Properties
    const propertiesList = document.getElementById('propertiesList');
    propertiesList.innerHTML = '';
    const props = prediction.properties;
    for (const [key, value] of Object.entries(props)) {
        const label = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
        propertiesList.innerHTML += `
            <div class="property-item">
                <span class="property-label">${label}</span>
                <span class="property-value">${value}</span>
            </div>
        `;
    }

    // Status Badge
    document.getElementById('soilStatus').textContent = recommendations.status;

    // Fertilizer Cards
    const fertilizerList = document.getElementById('fertilizerList');
    fertilizerList.innerHTML = '';
    if (recommendations.general_fertilizers) {
        recommendations.general_fertilizers.forEach(fert => {
            fertilizerList.innerHTML += `
                <div class="fertilizer-card">
                    <h4>${fert.name}</h4>
                    <p><strong>Purpose:</strong> ${fert.purpose}</p>
                    <p><strong>Application:</strong> ${fert.application}</p>
                    <p><strong>Frequency:</strong> ${fert.frequency}</p>
                </div>
            `;
        });
    }

    // Crop Specific
    const cropCard = document.getElementById('cropSpecificCard');
    const cropList = document.getElementById('cropSpecificList');
    if (recommendations.crop_specific_fertilizers && recommendations.crop_specific_fertilizers.length > 0) {
        cropCard.style.display = 'block';
        cropList.innerHTML = `<h4>For ${recommendations.selected_crop}</h4>`;
        recommendations.crop_specific_fertilizers.forEach(fert => {
            cropList.innerHTML += `
                <div class="fertilizer-card">
                    <h4>${fert.name}</h4>
                    <p><strong>Dose:</strong> ${fert.dose}</p>
                    <p><strong>Timing:</strong> ${fert.timing}</p>
                </div>
            `;
        });
    } else {
        cropCard.style.display = 'none';
    }

    // Organic Alternatives
    const organicList = document.getElementById('organicList');
    organicList.innerHTML = '';
    if (recommendations.organic_alternatives) {
        recommendations.organic_alternatives.forEach(item => {
            organicList.innerHTML += `<li>${item}</li>`;
        });
    }

    // Improvement Tips
    const tipsList = document.getElementById('tipsList');
    tipsList.innerHTML = '';
    if (recommendations.improvement_tips) {
        recommendations.improvement_tips.forEach(tip => {
            tipsList.innerHTML += `<div class="tip-card">${tip}</div>`;
        });
    }

    // Charts
    drawPredictionsChart(prediction.all_predictions);
    drawCompositionChart(prediction.soil_type);

    // Smooth scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

function drawPredictionsChart(predictions) {
    const ctx = document.getElementById('predictionsChart').getContext('2d');

    // Destroy existing chart if any
    if (window.predictionsChartInstance) {
        window.predictionsChartInstance.destroy();
    }

    const labels = Object.keys(predictions).map(
        s => s.charAt(0).toUpperCase() + s.slice(1)
    );
    const values = Object.values(predictions);

    const colors = ['#D5D8DC', '#D35400', '#8B4513', '#1C1C1C', '#F4D03F', '#85929E'];

    window.predictionsChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Confidence %',
                data: values,
                backgroundColor: colors,
                borderRadius: 8
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
                    ticks: {
                        callback: value => value + '%'
                    }
                }
            }
        }
    });
}

function drawCompositionChart(soilType) {
    const ctx = document.getElementById('compositionChart').getContext('2d');

    if (window.compositionChartInstance) {
        window.compositionChartInstance.destroy();
    }

    const compositions = {
        loamy: { Sand: 40, Silt: 40, Clay: 20 },
        sandy: { Sand: 70, Silt: 15, Clay: 15 },
        clay: { Sand: 20, Silt: 20, Clay: 60 },
        silty: { Sand: 20, Silt: 60, Clay: 20 },
        peaty: { 'Organic Matter': 70, Mineral: 30 },
        chalky: { 'Calcium Carbonate': 40, Sand: 30, Clay: 30 }
    };

    const data = compositions[soilType] || compositions.loamy;

    window.compositionChartInstance = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: Object.keys(data),
            datasets: [{
                data: Object.values(data),
                backgroundColor: ['#F4D03F', '#85929E', '#D35400', '#8B4513'],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        }
    });
}