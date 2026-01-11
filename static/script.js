// Initialize Bootstrap Toast
const errorToast = document.getElementById('errorToast');
const bsToast = errorToast ? new bootstrap.Toast(errorToast, { autohide: true, delay: 5000 }) : null;

// Navigation
const navItems = document.querySelectorAll('.nav-link-custom');
const contentSections = document.querySelectorAll('.content-section');

navItems.forEach(item => {
    item.addEventListener('click', (e) => {
        const href = item.getAttribute('href');
        
        // Only prevent default for hash-based navigation (sections on same page)
        // Allow normal navigation for full URLs (/samples, /privacy-policy, etc.)
        if (href && href.startsWith('#')) {
            e.preventDefault();
            const targetSection = item.getAttribute('data-section');
            
            // Update active nav item
            navItems.forEach(nav => nav.classList.remove('active'));
            item.classList.add('active');
            
            // Show target section
            contentSections.forEach(section => {
                section.classList.add('d-none');
                section.classList.remove('active');
            });
            const target = document.getElementById(targetSection);
            if (target) {
                target.classList.remove('d-none');
                target.classList.add('active');
            }
            
            // Scroll to top
            window.scrollTo({ top: 0, behavior: 'smooth' });
        }
        // For full URLs (/samples, /privacy-policy, /), let the browser handle navigation normally
    });
});

// File input element
const imageInput = document.getElementById('imageInput');
const uploadBox = document.getElementById('uploadBox');
const previewSection = document.getElementById('previewSection');
const previewImage = document.getElementById('previewImage');
const predictBtn = document.getElementById('predictBtn');
const predictBtnText = document.getElementById('predictBtnText');
const predictBtnLoader = document.getElementById('predictBtnLoader');
const resultSection = document.getElementById('resultSection');
const resultCard = document.getElementById('resultCard');
const errorMessage = document.getElementById('errorMessage');

// Initialize file upload handlers
function initFileUpload() {
    const browseBtn = document.querySelector('.browse-btn');
    
    // Handle browse button click - this should be the primary way to trigger file input
    if (browseBtn) {
        browseBtn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            e.stopImmediatePropagation();
            if (imageInput) {
                imageInput.click();
            }
            return false;
        }, true); // Use capture phase to handle it first
    }

    // Drag and drop handlers
    if (uploadBox) {
        uploadBox.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.stopPropagation();
            uploadBox.classList.add('dragover');
        });

        uploadBox.addEventListener('dragleave', (e) => {
            e.preventDefault();
            e.stopPropagation();
            uploadBox.classList.remove('dragover');
        });

        uploadBox.addEventListener('drop', (e) => {
            e.preventDefault();
            e.stopPropagation();
            uploadBox.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFileSelect(files[0]);
            }
        });

        // Handle click on upload box area (but not on buttons)
        uploadBox.addEventListener('click', (e) => {
            // Don't trigger if clicking on the browse button or any button
            if (e.target.closest('.browse-btn') || e.target.closest('button') || e.target.tagName === 'BUTTON' || e.target.closest('i')) {
                return;
            }
            // Only trigger if clicking on the upload box background
            if (imageInput) {
                imageInput.click();
            }
        });
    }

    // Handle file input change
    if (imageInput) {
        imageInput.addEventListener('change', (e) => {
            if (e.target.files && e.target.files.length > 0) {
                handleFileSelect(e.target.files[0]);
            }
        });
    }
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initFileUpload);
} else {
    // DOM already loaded, initialize immediately
    initFileUpload();
}

function handleFileSelect(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        showError('Please select a valid image file.');
        return;
    }
    
    // Validate file size (16MB max)
    if (file.size > 16 * 1024 * 1024) {
        showError('File size must be less than 16MB.');
        return;
    }
    
    // Hide error message
    hideError();
    
    // Switch to "The App" section if not already there
    const appSection = document.getElementById('the-app');
    if (appSection && !appSection.classList.contains('active')) {
        navItems.forEach(nav => nav.classList.remove('active'));
        const appNav = document.querySelector('[data-section="the-app"]');
        if (appNav) appNav.classList.add('active');
        contentSections.forEach(section => {
            section.classList.add('d-none');
            section.classList.remove('active');
        });
        appSection.classList.remove('d-none');
        appSection.classList.add('active');
    }
    
    // Hide upload box and show preview
    if (uploadBox) uploadBox.style.display = 'none';
    if (previewSection) previewSection.style.display = 'block';
    if (resultCard) resultCard.style.display = 'none';
    const placeholder = resultSection?.querySelector('.result-placeholder');
    if (placeholder) placeholder.style.display = 'flex';
    
    // Display preview
    const reader = new FileReader();
    reader.onload = (e) => {
        if (previewImage) previewImage.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

function removeImage() {
    if (imageInput) imageInput.value = '';
    if (uploadBox) uploadBox.style.display = 'flex';
    if (previewSection) previewSection.style.display = 'none';
    if (resultCard) resultCard.style.display = 'none';
    const placeholder = resultSection?.querySelector('.result-placeholder');
    if (placeholder) placeholder.style.display = 'flex';
    hideError();
}

function predictBMI() {
    const file = imageInput?.files[0];
    
    if (!file) {
        showError('Please select an image first.');
        return;
    }
    
    // Show loading state
    if (predictBtn) {
        predictBtn.disabled = true;
        predictBtnText.style.display = 'none';
        predictBtnLoader.style.display = 'inline-block';
    }
    hideError();
    
    // Create form data
    const formData = new FormData();
    formData.append('image', file);
    
    // Send prediction request
    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        // Hide loading state
        if (predictBtn) {
            predictBtn.disabled = false;
            predictBtnText.style.display = 'inline';
            predictBtnLoader.style.display = 'none';
        }
        
        if (data.success) {
            displayResult(data);
        } else {
            showError(data.error || 'Prediction failed. Please try again.');
        }
    })
    .catch(error => {
        // Hide loading state
        if (predictBtn) {
            predictBtn.disabled = false;
            predictBtnText.style.display = 'inline';
            predictBtnLoader.style.display = 'none';
        }
        
        showError('Network error. Please check your connection and try again.');
        console.error('Error:', error);
    });
}

function displayResult(data) {
    // Hide placeholder, show result card
    const placeholder = resultSection?.querySelector('.result-placeholder');
    if (placeholder) placeholder.style.display = 'none';
    if (resultCard) resultCard.style.display = 'block';
    
    // Update BMI value
    const bmiValueEl = document.getElementById('bmiValue');
    if (bmiValueEl) bmiValueEl.textContent = data.bmi;
    
    // Update category
    const categoryElement = document.getElementById('bmiCategory');
    if (categoryElement) {
        categoryElement.textContent = data.category;
        
        // Update category color
        const categoryColors = {
            'Underweight': '#3498db',
            'Normal weight': '#2ecc71',
            'Overweight': '#f39c12',
            'Obese': '#e74c3c'
        };
        categoryElement.style.color = categoryColors[data.category] || '#666';
    }
    
    // Update message
    const messageEl = document.getElementById('bmiMessage');
    if (messageEl) messageEl.textContent = data.message || '';
    
    // Update scale indicator position
    updateScaleIndicator(data.bmi);
    
    // Scroll to results
    if (resultCard) {
        resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
}

function updateScaleIndicator(bmi) {
    const indicator = document.getElementById('scaleIndicator');
    if (!indicator) return;
    
    // Calculate position based on BMI
    // BMI ranges: Underweight (<18.5), Normal (18.5-25), Overweight (25-30), Obese (30+)
    let position;
    
    if (bmi < 18.5) {
        // Underweight range: 0-18.5, map to first 25% of scale
        position = (bmi / 18.5) * 25;
    } else if (bmi < 25) {
        // Normal range: 18.5-25, map to 25-50% of scale
        position = 25 + ((bmi - 18.5) / (25 - 18.5)) * 25;
    } else if (bmi < 30) {
        // Overweight range: 25-30, map to 50-75% of scale
        position = 50 + ((bmi - 25) / (30 - 25)) * 25;
    } else {
        // Obese range: 30+, map to 75-100% of scale (cap at 100%)
        position = Math.min(75 + ((bmi - 30) / 10) * 25, 100);
    }
    
    indicator.style.left = `${position}%`;
    indicator.style.display = 'block';
}

function resetApp() {
    removeImage();
    hideError();
}

function showError(message) {
    if (errorMessage) {
        errorMessage.textContent = message;
    }
    if (bsToast) {
        bsToast.show();
    }
}

function hideError() {
    if (bsToast) {
        bsToast.hide();
    }
}
