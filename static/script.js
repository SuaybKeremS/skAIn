// ===================== DOM ELEMENTS =====================
const uploadZone = document.getElementById('upload-zone');
const imageInput = document.getElementById('image-input');
const imagePreview = document.getElementById('image-preview');
const uploadContent = uploadZone.querySelector('.upload-content');
const form = document.getElementById('prediction-form');
const submitBtn = document.getElementById('submit-btn');
const btnText = submitBtn.querySelector('.btn-text');
const btnLoader = submitBtn.querySelector('.btn-loader');
const resultsSection = document.getElementById('results-section');
const resultsContainer = document.getElementById('results-container');

// ===================== IMAGE UPLOAD =====================

// Click to upload
uploadZone.addEventListener('click', () => imageInput.click());

// Drag and drop handlers
uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
});

uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('dragover');
});

uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].type.startsWith('image/')) {
        handleImageUpload(files[0]);
    }
});

// File input change
imageInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleImageUpload(e.target.files[0]);
    }
});

// Handle image upload
function handleImageUpload(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        imagePreview.style.display = 'block';
        uploadContent.style.display = 'none';
        
        // Add success animation
        uploadZone.style.borderColor = '#10b981';
        setTimeout(() => {
            uploadZone.style.borderColor = '';
        }, 1000);
    };
    reader.readAsDataURL(file);
    
    // Update file input
    const dataTransfer = new DataTransfer();
    dataTransfer.items.add(file);
    imageInput.files = dataTransfer.files;
}

// ===================== FORM SUBMISSION =====================

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // Check if image is uploaded
    if (!imageInput.files.length) {
        showError('Lütfen bir görsel yükleyin');
        return;
    }
    
    // Show loading state
    setLoading(true);
    
    try {
        const formData = new FormData(form);
        
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.error) {
            showError(data.error);
        } else {
            displayResults(data.predictions);
        }
    } catch (error) {
        showError('Bağlantı hatası. Lütfen tekrar deneyin.');
        console.error(error);
    } finally {
        setLoading(false);
    }
});

// ===================== LOADING STATE =====================

function setLoading(loading) {
    submitBtn.disabled = loading;
    btnText.style.display = loading ? 'none' : 'inline';
    btnLoader.style.display = loading ? 'inline-flex' : 'none';
}

// ===================== DISPLAY RESULTS =====================

function displayResults(predictions) {
    resultsContainer.innerHTML = '';
    
    predictions.forEach((pred, index) => {
        const card = document.createElement('div');
        card.className = `result-card${index === 0 ? ' top-result' : ''}`;
        
        card.innerHTML = `
            <div class="result-rank">#${index + 1}</div>
            <div class="result-info">
                <div class="result-class">${pred.class}</div>
                <div class="result-description">${pred.description}</div>
            </div>
            <div class="result-probability">
                <div class="probability-value">${pred.probability.toFixed(1)}%</div>
                <div class="probability-bar">
                    <div class="probability-fill" style="width: 0%"></div>
                </div>
            </div>
        `;
        
        resultsContainer.appendChild(card);
        
        // Animate probability bar
        setTimeout(() => {
            const fill = card.querySelector('.probability-fill');
            fill.style.width = `${pred.probability}%`;
        }, 100 + index * 100);
    });
    
    resultsSection.style.display = 'block';
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ===================== ERROR HANDLING =====================

function showError(message) {
    // Create toast notification
    const toast = document.createElement('div');
    toast.className = 'error-toast';
    toast.innerHTML = `
        <span>⚠️ ${message}</span>
    `;
    toast.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
        padding: 16px 24px;
        border-radius: 8px;
        font-weight: 500;
        box-shadow: 0 8px 25px rgba(239, 68, 68, 0.4);
        z-index: 1000;
        animation: slideIn 0.3s ease-out;
    `;
    
    // Add animation keyframes
    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideIn {
            from { opacity: 0; transform: translateX(50px); }
            to { opacity: 1; transform: translateX(0); }
        }
        @keyframes slideOut {
            from { opacity: 1; transform: translateX(0); }
            to { opacity: 0; transform: translateX(50px); }
        }
    `;
    document.head.appendChild(style);
    
    document.body.appendChild(toast);
    
    // Remove after 4 seconds
    setTimeout(() => {
        toast.style.animation = 'slideOut 0.3s ease-out forwards';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// ===================== INITIALIZE =====================

console.log('🏥 Skin Disease Prediction App loaded');
