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

let testDayFiles = [];
function handleTestDayUpload(event) {
    const files = event.target.files;
    if (!files.length) return;
    
    document.getElementById('test-day-status').innerText = 'Uploading ' + files.length + ' images...';
    
    const formData = new FormData();
    for(let i=0; i<files.length; i++) {
        formData.append('files', files[i]);
    }
    
    fetch('/upload_test_images', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        testDayFiles = data.filenames;
        document.getElementById('test-day-status').innerText = 'Uploaded successfully. Please label them:';
        
        const gallery = document.getElementById('gallery-grid');
        gallery.innerHTML = '';
        
        testDayFiles.forEach(filename => {
            const div = document.createElement('div');
            div.className = 'gallery-item';
            div.id = 'td-item-' + filename.replace(/[^a-zA-Z0-9]/g, '-');
            
            const img = document.createElement('img');
            img.src = '/test_images/' + filename + '?' + new Date().getTime(); // cache bust
            
            const select = document.createElement('select');
            select.id = 'label-' + filename;
            const opt0 = document.createElement('option'); opt0.value = "0"; opt0.text = "No Ship";
            const opt1 = document.createElement('option'); opt1.value = "1"; opt1.text = "Ship";
            select.appendChild(opt0);
            select.appendChild(opt1);
            
            const predText = document.createElement('div');
            predText.id = 'pred-text-' + filename;
            predText.style.fontSize = '12px';
            predText.style.marginTop = '4px';
            
            div.appendChild(img);
            div.appendChild(select);
            div.appendChild(predText);
            gallery.appendChild(div);
        });
        
        document.getElementById('labeling-container').style.display = 'block';
        document.getElementById('test-day-results').style.display = 'none';
    })
    .catch(err => {
        console.error(err);
        document.getElementById('test-day-status').innerText = 'Upload failed.';
    });
}

let tdChart = null;
function runTestDayEvaluation() {
    const labels = {};
    testDayFiles.forEach(filename => {
        const select = document.getElementById('label-' + filename);
        labels[filename] = parseInt(select.value);
    });
    
    document.getElementById('run-test-day-btn').innerText = 'Running Inference...';
    
    fetch('/evaluate_test_day', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({labels: labels})
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('run-test-day-btn').innerText = 'Run Inference & Evaluate';
        document.getElementById('test-day-results').style.display = 'block';
        
        animateValue('td-accuracy', 0, data.accuracy * 100, 1000);
        animateValue('td-precision', 0, data.precision * 100, 1000);
        animateValue('td-recall', 0, data.recall * 100, 1000);
        
        // Render Chart
        const ctx = document.getElementById('tdConfusionMatrix').getContext('2d');
        if (tdChart) tdChart.destroy();
        tdChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['True Positives (Ship)', 'True Negatives (No Ship)', 'False Positives', 'False Negatives'],
                datasets: [{
                    label: 'Number of Images',
                    data: [data.confusion_matrix.tp, data.confusion_matrix.tn, data.confusion_matrix.fp, data.confusion_matrix.fn],
                    backgroundColor: [
                        'rgba(16, 185, 129, 0.7)',
                        'rgba(59, 130, 246, 0.7)',
                        'rgba(239, 68, 68, 0.7)',
                        'rgba(245, 158, 11, 0.7)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { display: false },
                    title: { display: true, text: 'Confusion Matrix Distribution', color: 'white' }
                },
                scales: {
                    y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.1)' }, ticks: { color: '#cbd5e1' } },
                    x: { grid: { display: false }, ticks: { color: '#cbd5e1' } }
                }
            }
        });
        
        // Highlight gallery items
        data.results_detail.forEach(r => {
            const item = document.getElementById('td-item-' + r.filename.replace(/[^a-zA-Z0-9]/g, '-'));
            const text = document.getElementById('pred-text-' + r.filename);
            item.classList.remove('correct-pred', 'incorrect-pred');
            if(r.true_label === r.prediction) {
                item.classList.add('correct-pred');
                text.innerText = "Correct (" + (r.prediction===1?'Ship':'No Ship') + ")";
                text.style.color = 'var(--success-color)';
            } else {
                item.classList.add('incorrect-pred');
                text.innerText = "Wrong (Pred: " + (r.prediction===1?'Ship':'No Ship') + ")";
                text.style.color = 'var(--danger-color)';
            }
        });
    })
    .catch(err => {
        console.error(err);
        document.getElementById('run-test-day-btn').innerText = 'Error running inference';
    });
}

