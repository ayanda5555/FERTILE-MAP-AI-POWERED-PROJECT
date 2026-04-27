// ==================== MAIN APP LOGIC ====================

document.addEventListener('DOMContentLoaded', () => {
    checkAuth();

    // Check if viewing a past analysis
    const params = new URLSearchParams(window.location.search);
    if (params.get('view') === 'true') {
        const analysisData = JSON.parse(sessionStorage.getItem('viewAnalysis') || 'null');
        if (analysisData) {
            // Format data for displayResults function
            const formattedData = {
                prediction: {
                    soil_type: analysisData.soil_type,
                    confidence: analysisData.confidence,
                    confidence_percent: (analysisData.confidence * 100).toFixed(2),
                    properties: analysisData.properties,
                    all_predictions: {} // Not available from history
                },
                recommendations: analysisData.recommendations,
                image_url: analysisData.image_url
            };

            // Hide upload section and show results
            const uploadSection = document.getElementById('uploadSection');
            if (uploadSection) uploadSection.style.display = 'none';

            if (typeof displayResults === 'function') {
                displayResults(formattedData);
            }

            sessionStorage.removeItem('viewAnalysis');
        }
    }
});