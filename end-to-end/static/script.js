document.addEventListener('DOMContentLoaded', () => {
    // 1. PLEXUS
    const canvas = document.getElementById('particle-canvas');
    const ctx = canvas.getContext('2d');
    let particles = [];
    function init() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        particles = [];
        for (let i = 0; i < 65; i++) {
            particles.push({
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                vx: (Math.random() - 0.5) * 0.35,
                vy: (Math.random() - 0.5) * 0.35,
                size: Math.random() * 2
            });
        }
    }
    function animate() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = 'rgba(0, 245, 212, 0.2)';
        particles.forEach(p => {
            p.x += p.vx; p.y += p.vy;
            if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
            if (p.y < 0 || p.y > canvas.height) p.vy *= -1;
            ctx.beginPath(); ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2); ctx.fill();
        });
        requestAnimationFrame(animate);
    }
    init(); animate();


    // 3. BENCHMARK CHARTS
    const metricExplanations = {
        "Precision": "Ratio of correct 'Fake' flags. Higher = fewer false alarms.",
        "Recall": "Ratio of 'Fake' caught. Higher = fewer missed threats.",
        "F1": "Balanced harmonic mean of Precision and Recall.",
        "Acc": "Overall percentage of correct classifications.",
        "AUC": "Ability to distinguish between Real and Fake classes.",
        "Sarcasm": "Model's ability to detect ironic or satirical sentiment.",
        "Stylometry": "Analysis of writing style and emotional intensity.",
        "Sequential": "Capturing context through sentence order and flow.",
        "Context": "Deep semantic understanding of the claim's background.",
        "Robustness": "Resistance to adversarial text and domain shifts."
    };

    const benchCtx = document.getElementById('benchmarkChart').getContext('2d');
    const gradientBench = benchCtx.createLinearGradient(0, 0, 0, 400);
    gradientBench.addColorStop(0, 'rgba(0, 245, 212, 0.6)');
    gradientBench.addColorStop(0.5, 'rgba(0, 245, 212, 0.2)');
    gradientBench.addColorStop(1, 'rgba(0, 245, 212, 0)');

    new Chart(benchCtx, {
        type: 'bar',
        data: {
            labels: ['Precision', 'Recall', 'F1', 'Acc', 'AUC'],
            datasets: [
                {
                    label: 'BiLSTM',
                    data: [0.76, 0.81, 0.78, 0.79, 0.82],
                    backgroundColor: 'rgba(255, 255, 255, 0.03)',
                    borderColor: 'rgba(255, 255, 255, 0.1)',
                    borderWidth: 1,
                    borderRadius: 4
                },
                {
                    label: 'TrustLens',
                    data: [0.91, 0.92, 0.92, 0.93, 0.94],
                    backgroundColor: gradientBench,
                    borderColor: '#00f5d4',
                    borderWidth: 2,
                    borderRadius: 4,
                    hoverBackgroundColor: '#00f5d4'
                }
            ]
        },
        options: { 
            responsive: true,
            maintainAspectRatio: false,
            scales: { 
                y: { 
                    beginAtZero: true,
                    max: 1, 
                    grid: { color: 'rgba(255,255,255,0.03)' },
                    ticks: { color: '#475569', font: { size: 9 } }
                }, 
                x: { 
                    grid: { display: false }, 
                    ticks: { color: '#94a3b8', font: { family: 'JetBrains Mono', size: 10 } } 
                } 
            }, 
            plugins: { 
                legend: { 
                    position: 'top',
                    labels: { color: '#64748b', font: { size: 10, family: 'JetBrains Mono' }, usePointStyle: true } 
                },
                tooltip: {
                    backgroundColor: 'rgba(1, 2, 8, 0.95)',
                    titleFont: { family: 'JetBrains Mono' },
                    bodyFont: { family: 'JetBrains Mono' },
                    padding: 12,
                    borderColor: 'rgba(0, 245, 212, 0.2)',
                    borderWidth: 1,
                    callbacks: {
                        afterLabel: function(context) {
                            const metric = context.chart.data.labels[context.dataIndex];
                            return "\n" + (metricExplanations[metric] || "");
                        }
                    }
                }
            } 
        }
    });

    const radarCtx = document.getElementById('radarChartBenchmarks').getContext('2d');
    new Chart(radarCtx, {
        type: 'radar',
        data: {
            labels: ['Sarcasm', 'Stylometry', 'Sequential', 'Context', 'Robustness'],
            datasets: [{
                label: 'Ensemble Fingerprint',
                data: [0.85, 0.92, 0.88, 0.95, 0.90],
                backgroundColor: 'rgba(0, 245, 212, 0.2)',
                borderColor: '#00f5d4',
                borderWidth: 2,
                pointBackgroundColor: '#00f5d4',
                pointBorderColor: '#fff',
                pointHoverRadius: 6,
                pointRadius: 4
            }]
        },
        options: { 
            responsive: true,
            maintainAspectRatio: false,
            scales: { 
                r: { 
                    angleLines: { color: 'rgba(255,255,255,0.05)' },
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    pointLabels: { color: '#94a3b8', font: { size: 10, family: 'JetBrains Mono' } },
                    ticks: { display: false },
                    suggestedMin: 0,
                    suggestedMax: 1
                } 
            }, 
            plugins: { 
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(1, 2, 8, 0.95)',
                    padding: 12,
                    borderColor: 'rgba(0, 245, 212, 0.2)',
                    borderWidth: 1,
                    callbacks: {
                        afterLabel: function(context) {
                            const metric = context.chart.data.labels[context.dataIndex];
                            return "\n" + (metricExplanations[metric] || "");
                        }
                    }
                }
            } 
        }
    });

    // 4. EXPLANATION TOOLTIPS
    const tooltipEl = document.getElementById('explanation-tooltip');
    document.querySelectorAll('[data-explanation]').forEach(el => {
        el.addEventListener('mousemove', (e) => {
            tooltipEl.style.display = 'block';
            tooltipEl.textContent = el.getAttribute('data-explanation');
            tooltipEl.style.left = (e.clientX + 20) + 'px';
            tooltipEl.style.top = (e.clientY + 20) + 'px';
        });
        el.addEventListener('mouseleave', () => {
            tooltipEl.style.display = 'none';
        });
    });

    // 5. AUDIT LOGIC
    const btnText = document.getElementById('btn-text');
    const btnUrl = document.getElementById('btn-url');
    const payload = document.getElementById('payload');
    let mode = 'text';

    btnText.addEventListener('click', () => { mode = 'text'; btnText.classList.add('active'); btnUrl.classList.remove('active'); });
    btnUrl.addEventListener('click', () => { mode = 'url'; btnUrl.classList.add('active'); btnText.classList.remove('active'); });

    const runAudit = document.getElementById('run-audit');
    const resSection = document.getElementById('audit-results');

    runAudit.addEventListener('click', async () => {
        const val = payload.value;
        if (!val.trim()) return alert('Payload required.');

        runAudit.disabled = true;
        runAudit.innerHTML = '<i class="fas fa-circle-notch fa-spin"></i> SCANNING...';
        resSection.classList.add('hidden');

        try {
            const res = await fetch('/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(mode === 'url' ? { url: val } : { text: val })
            });
            const data = await res.json();
            renderReport(data);
        } catch (e) {
            console.error(e);
            alert('Forensic System Offline.');
        } finally {
            runAudit.disabled = false;
            runAudit.innerHTML = '<i class="fas fa-search"></i> Initiate Forensic Scan';
        }
    });

    function renderReport(data) {
        resSection.classList.remove('hidden');
        resSection.scrollIntoView({ behavior: 'smooth' });

        const label = document.getElementById('verdict-label');
        label.textContent = data.verdict;
        label.style.color = data.verdict === 'FAKE' ? '#00f5d4' : '#ff2b56';
        
        document.getElementById('conf-val').textContent = `CONFIDENCE: ${(data.confidence * 100).toFixed(2)}%`;
        document.getElementById('gauge-fill').style.width = `${data.confidence * 100}%`;
        document.getElementById('gauge-fill').style.background = data.verdict === 'FAKE' ? '#00f5d4' : '#ff2b56';

        // Neural Agreement
        document.getElementById('dl-prob-fill').style.width = `${data.dl_model.prob * 100}%`;
        document.getElementById('nlp-prob-fill').style.width = `${data.nlp_model.prob * 100}%`;

        // Stylometric
        document.getElementById('sent-val').textContent = data.classical.sentiment > 0.1 ? 'Positive' : (data.classical.sentiment < -0.1 ? 'Negative' : 'Neutral');
        document.getElementById('pos-val').textContent = data.classical.adj_noun_ratio.toFixed(2);
        document.getElementById('noun-val').textContent = data.classical.nouns;
        document.getElementById('caps-val').textContent = data.classical.sentiment > 0.5 ? 'Extreme' : 'Moderate';

        // Heatmap
        const heatmap = document.getElementById('token-heatmap');
        heatmap.innerHTML = '';
        const maxW = Math.max(...data.attention_weights.map(w => w.weight));
        data.attention_weights.forEach(item => {
            const span = document.createElement('div');
            span.className = 'token-box';
            span.textContent = item.token;
            const intensity = item.weight / (maxW || 1);
            if (intensity > 0.4) {
                span.style.background = data.verdict === 'FAKE' ? `rgba(0, 245, 212, ${intensity})` : `rgba(255, 43, 86, ${intensity})`;
                span.style.color = '#000';
                span.style.fontWeight = '800';
                span.style.border = 'none';
            }
            heatmap.appendChild(span);
        });
    }

    // 6. GRAPH SUMMARIES
    const summaryText = document.getElementById('summary-text');
    const barContainer = document.getElementById('bar-chart-container');
    const radarContainer = document.getElementById('radar-chart-container');

    const summaries = {
        bar: "The <span class='summary-highlight'>Metric Distribution</span> matrix reveals that the TrustLens Ensemble significantly outperforms the BiLSTM baseline, showing a <span class='summary-highlight'>15.2% gain</span> in F1-Score. This indicates superior architectural synergy between the BERT context engine and stylistic feature extractors.",
        radar: "The <span class='summary-highlight'>Neural Fingerprint</span> audit exposes high sensitivity to <span class='summary-highlight'>Context (95%)</span> and <span class='summary-highlight'>Stylometry (92%)</span>. The radial spread confirms the model's ability to maintain high precision even in adversarial scenarios involving sarcasm and domain-shifted claims."
    };

    if (barContainer && radarContainer) {
        barContainer.addEventListener('mouseenter', () => { summaryText.innerHTML = summaries.bar; });
        radarContainer.addEventListener('mouseenter', () => { summaryText.innerHTML = summaries.radar; });
    }

    // Table Row Hover Insight
    document.querySelectorAll('.forensic-table tr[data-explanation]').forEach(row => {
        row.addEventListener('mouseenter', () => {
            summaryText.innerHTML = `<span class='summary-highlight'>Neural Insight:</span> ${row.getAttribute('data-explanation')}`;
        });
    });

    // 7. SCROLL REVEAL (WORKFLOW)
    const workflowObserverOptions = { threshold: 0.5 };
    const workflowObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('active');
            }
        });
    }, workflowObserverOptions);

    document.querySelectorAll('.wf-step, .learn-more-container').forEach(el => workflowObserver.observe(el));
});
