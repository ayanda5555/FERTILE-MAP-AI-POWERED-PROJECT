// ==================== IMAGE CAPTURE & UPLOAD ====================

let selectedFile = null;

document.addEventListener('DOMContentLoaded', () => {
    if (!requireAuth()) return;
    setupDropZone();
    setupFileInput();
});

function setupDropZone() {
    const dropZone = document.getElementById('dropZone');
    if (!dropZone) return;

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    dropZone.addEventListener('click', () => {
        document.getElementById('fileInput').click();
    });
}

function setupFileInput() {
    const fileInput = document.getElementById('fileInput');
    if (!fileInput) return;

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });
}

function handleFile(file) {
    // Validate file type
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
    if (!allowedTypes.includes(file.type)) {
        alert('Please upload a JPG, PNG, or WEBP image.');
        return;
    }

    // Validate file size (max 16MB)
    if (file.size > 16 * 1024 * 1024) {
        alert('File size must be less than 16MB.');
        return;
    }

    selectedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        document.getElementById('imagePreview').src = e.target.result;
        document.getElementById('previewSection').style.display = 'block';
        document.getElementById('dropZone').style.display = 'none';
    };
    reader.readAsDataURL(file);
}

function openCamera() {
    const input = document.getElementById('fileInput');
    input.setAttribute('capture', 'environment');
    input.click();
    // Remove capture attribute after click so file picker works normally next time
    setTimeout(() => input.removeAttribute('capture'), 1000);
}

function clearImage() {
    selectedFile = null;
    document.getElementById('imagePreview').src = '';
    document.getElementById('previewSection').style.display = 'none';
    document.getElementById('dropZone').style.display = 'block';
    document.getElementById('fileInput').value = '';
}

async function analyzeSoil() {
    if (!selectedFile) {
        alert('Please select an image first.');
        return;
    }

    const cropType = document.getElementById('cropType').value;

    // Show loading
    document.getElementById('previewSection').style.display = 'none';
    document.getElementById('loadingSection').style.display = 'block';

    const formData = new FormData();
    formData.append('image', selectedFile);
    formData.append('crop_type', cropType);

    try {
        const result = await API.postForm('/analyze', formData);

        // Hide loading, show results
        document.getElementById('loadingSection').style.display = 'none';
        document.getElementById('uploadSection').style.display = 'none';

        displayResults(result);

    } catch (error) {
        document.getElementById('loadingSection').style.display = 'none';
        document.getElementById('previewSection').style.display = 'block';
        alert('Analysis failed: ' + error.message);
    }
}

function newAnalysis() {
    selectedFile = null;
    document.getElementById('resultsSection').style.display = 'none';
    document.getElementById('uploadSection').style.display = 'block';
    document.getElementById('dropZone').style.display = 'block';
    document.getElementById('previewSection').style.display = 'none';
    document.getElementById('fileInput').value = '';
}