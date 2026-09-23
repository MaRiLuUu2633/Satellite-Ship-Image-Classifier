function showSection(sectionId) {
    // Hide all sections
    document.querySelectorAll('main section').forEach(sec => {
        sec.style.display = 'none';
        sec.classList.remove('active');
    });
    // Remove active class from buttons
    document.querySelectorAll('.nav-btn').forEach(btn => btn.classList.remove('active'));
    
    // Show target section
    const target = document.getElementById(sectionId);
    target.style.display = 'block';
    // Small delay to allow display:block to apply before animating opacity
    setTimeout(() => target.classList.add('active'), 10);
    
    // Set active button
    event.currentTarget.classList.add('active');
}

// Drag and drop handling
const dropZone = document.getElementById('drop-zone');
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
    if (e.dataTransfer.files.length) {
        processFile(e.dataTransfer.files[0]);
    }
});

function handleFileUpload(event) {
    if (event.target.files.length) {
        processFile(event.target.files[0]);
    }
}

function processFile(file) {
    if (!file.type.startsWith('image/')) return;
    
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        document.getElementById('preview-img').src = e.target.result;
        document.getElementById('preview-img').style.display = 'inline-block';
    };
    reader.readAsDataURL(file);

    // Send to API
    const formData = new FormData();
    formData.append('file', file);
    
    document.getElementById('prediction-text').innerText = 'Analyzing...';
    document.getElementById('prediction-text').style.color = '#fff';
    document.getElementById('confidence-bar-container').style.display = 'none';

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        const textElem = document.getElementById('prediction-text');
        const confBar = document.getElementById('confidence-bar');
        const confText = document.getElementById('confidence-text');
        
        const isShip = data.prediction === 'ship';
        textElem.innerText = isShip ? 'SHIP DETECTED' : 'NO SHIP DETECTED';
        textElem.style.color = isShip ? 'var(--success-color)' : 'var(--danger-color)';
        
        document.getElementById('confidence-bar-container').style.display = 'block';
        confBar.style.width = `${(data.confidence * 100).toFixed(1)}%`;
        confBar.style.backgroundColor = isShip ? 'var(--success-color)' : 'var(--danger-color)';
        confText.innerText = `Confidence: ${(data.confidence * 100).toFixed(2)}%`;
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('prediction-text').innerText = 'Error analyzing image';
    });
}

let confusionMatrixChart = null;

function runEvaluation() {
    document.getElementById('eval-btn').style.display = 'none';
    document.getElementById('loading-spinner').style.display = 'block';
    document.getElementById('metrics-grid').style.display = 'none';
    document.getElementById('charts-container').style.display = 'none';

    fetch('/evaluate')
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert(data.error);
            document.getElementById('eval-btn').style.display = 'block';
            document.getElementById('loading-spinner').style.display = 'none';
            return;
        }

        document.getElementById('loading-spinner').style.display = 'none';
        document.getElementById('metrics-grid').style.display = 'grid';
        document.getElementById('charts-container').style.display = 'block';

        // Animate numbers
        animateValue('metric-accuracy', 0, data.accuracy * 100, 1500);
        animateValue('metric-precision', 0, data.precision * 100, 1500);
        animateValue('metric-recall', 0, data.recall * 100, 1500);

        renderChart(data.confusion_matrix);
        document.getElementById('eval-btn').style.display = 'block';
        document.getElementById('eval-btn').innerText = 'Run Evaluation Again';
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('eval-btn').style.display = 'block';
        document.getElementById('loading-spinner').style.display = 'none';
        alert('Evaluation failed. Is the backend running?');
    });
}

function animateValue(id, start, end, duration) {
    const obj = document.getElementById(id);
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        obj.innerHTML = (progress * (end - start) + start).toFixed(2) + '%';
        if (progress < 1) {
            window.requestAnimationFrame(step);
        }
    };
    window.requestAnimationFrame(step);
}

function renderChart(cm) {
    const ctx = document.getElementById('confusionMatrix').getContext('2d');
    
    if (confusionMatrixChart) {
        confusionMatrixChart.destroy();
    }
    
    // We'll use a Bar chart to represent the 4 quadrants of Confusion Matrix
    confusionMatrixChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['True Positives (Ship)', 'True Negatives (No Ship)', 'False Positives', 'False Negatives'],
            datasets: [{
                label: 'Number of Images',
                data: [cm.tp, cm.tn, cm.fp, cm.fn],
                backgroundColor: [
                    'rgba(16, 185, 129, 0.7)', // Green TP
                    'rgba(59, 130, 246, 0.7)', // Blue TN
                    'rgba(239, 68, 68, 0.7)',  // Red FP
                    'rgba(245, 158, 11, 0.7)'  // Orange FN
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    display: false
                },
                title: {
                    display: true,
                    text: 'Confusion Matrix Distribution',
                    color: 'white',
                    font: { size: 16 }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: 'rgba(255,255,255,0.1)' },
                    ticks: { color: '#cbd5e1' }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#cbd5e1' }
                }
            }
        }
    });
}
