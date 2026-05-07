document.addEventListener('DOMContentLoaded', () => {
    // 1. Cursor Glow
    const glow = document.getElementById('cursor-glow');
    document.addEventListener('mousemove', (e) => {
        glow.style.transform = `translate(${e.clientX}px, ${e.clientY}px)`;
    });

    // 2. Navigation
    const views = {
        home: document.getElementById('view-home'),
        analyzer: document.getElementById('view-analyzer'),
        architecture: document.getElementById('view-architecture'),
        history: document.getElementById('view-history')
    };

    const navItems = document.querySelectorAll('.nav-item');

    window.switchView = (viewName) => {
        Object.keys(views).forEach(key => {
            if (views[key]) views[key].classList.toggle('hidden', key !== viewName);
        });
        navItems.forEach(item => {
            item.classList.toggle('active', item.dataset.view === viewName);
        });
        if (viewName === 'history') updateAuditLog();
    };

    navItems.forEach(item => {
        item.addEventListener('click', () => switchView(item.dataset.view));
    });

    // 3. Mini Radar Chart (Home)
    const miniRadarCtx = document.getElementById('miniRadar').getContext('2d');
    const miniRadar = new Chart(miniRadarCtx, {
        type: 'radar',
        data: {
            labels: ['Context', 'Style', 'Sentiment', 'Keywords', 'Coherence'],
            datasets: [{
                data: [86, 89, 74, 82, 85],
                backgroundColor: 'rgba(99, 102, 241, 0.2)',
                borderColor: 'rgba(99, 102, 241, 1)',
                borderWidth: 2,
                pointRadius: 0
            }]
        },
        options: {
            scales: { r: { grid: { color: 'rgba(255,255,255,0.05)' }, angleLines: { display: false }, pointLabels: { display: false }, ticks: { display: false } } },
            plugins: { legend: { display: false } },
            responsive: true,
            maintainAspectRatio: false
        }
    });

    // 4. Analyzer Prob Chart
    const probCtx = document.getElementById('probChart').getContext('2d');
    const probChart = new Chart(probCtx, {
        type: 'bar',
        data: {
            labels: ['DL', 'NLP', 'Ens'],
            datasets: [{
                data: [0, 0, 0],
                backgroundColor: ['#6366f1', '#00f2ff', '#ffffff'],
                borderRadius: 5
            }]
        },
        options: {
            indexAxis: 'y',
            scales: { x: { display: false, max: 1 }, y: { ticks: { color: 'rgba(255,255,255,0.5)', font: { size: 10 } }, grid: { display: false } } },
            plugins: { legend: { display: false } },
            responsive: true,
            maintainAspectRatio: false
        }
    });

    // 5. Analyzer Engine
    const analyzeBtn = document.getElementById('analyze-btn');
    const textInput = document.getElementById('text-input');
    const urlInput = document.getElementById('url-input');
    const resultsDashboard = document.getElementById('results-dashboard');

    document.querySelectorAll('.tab-opt').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.tab-opt').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const isText = btn.dataset.mode === 'text';
            textInput.classList.toggle('hidden', !isText);
            urlInput.classList.toggle('hidden', isText);
        });
    });

    analyzeBtn.addEventListener('click', async () => {
        const mode = document.querySelector('.tab-opt.active').dataset.mode;
        const text = mode === 'text' ? textInput.value : urlInput.value;
        if (!text) return;

        analyzeBtn.disabled = true;
        analyzeBtn.textContent = '...';
        document.getElementById('processing-steps').classList.remove('hidden');

        try {
            const res = await fetch('/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(mode === 'text' ? { text } : { url: text })
            });
            const data = await res.json();
            renderResults(data);
            saveToHistory(text, data.verdict);
        } catch (e) { console.error(e); }
        finally {
            analyzeBtn.disabled = false;
            analyzeBtn.textContent = 'AUDIT';
        }
    });

    function renderResults(data) {
        resultsDashboard.classList.remove('hidden');
        const label = document.getElementById('verdict-label');
        label.textContent = data.verdict;
        label.style.color = data.verdict === 'FAKE' ? 'var(--fake)' : 'var(--real)';
        document.getElementById('trust-bar').style.width = `${data.ensemble_score * 100}%`;
        document.getElementById('trust-bar').style.background = data.verdict === 'FAKE' ? 'var(--fake)' : 'var(--real)';
        
        probChart.data.datasets[0].data = [data.dl_model.prob, data.nlp_model.prob, data.ensemble_score];
        probChart.update();

        const container = document.getElementById('heatmap-container');
        container.innerHTML = '';
        const max = Math.max(...data.attention_weights.map(w => w.weight));
        data.attention_weights.forEach(item => {
            const span = document.createElement('span');
            span.textContent = item.token + ' ';
            const intensity = item.weight / (max || 1);
            span.style.background = `rgba(99, 102, 241, ${intensity * 0.6})`;
            span.style.padding = '0 2px';
            container.appendChild(span);
        });
    }

    function saveToHistory(text, verdict) {
        const history = JSON.parse(localStorage.getItem('trustlens_history') || '[]');
        history.unshift({ text: text.substring(0, 50) + '...', verdict, date: new Date().toLocaleTimeString() });
        localStorage.setItem('trustlens_history', JSON.stringify(history.slice(0, 10)));
    }

    function updateAuditLog() {
        const container = document.getElementById('history-container');
        const history = JSON.parse(localStorage.getItem('trustlens_history') || '[]');
        container.innerHTML = history.length ? '' : '<p style="text-align:center; opacity:0.3;">EMPTY</p>';
        history.forEach(item => {
            const div = document.createElement('div');
            div.className = 'history-item';
            div.style.padding = '1rem';
            div.style.background = 'rgba(255,255,255,0.02)';
            div.style.borderRadius = '10px';
            div.style.marginBottom = '0.5rem';
            div.style.display = 'flex';
            div.style.justifyContent = 'space-between';
            div.innerHTML = `<span style="font-size:0.7rem;">${item.text}</span><span style="font-size:0.7rem; color:${item.verdict==='FAKE'?'var(--fake)':'var(--real)'}">${item.verdict}</span>`;
            container.appendChild(div);
        });
    }
});
