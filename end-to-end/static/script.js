document.addEventListener('DOMContentLoaded', () => {
    // 1. Cursor Glow
    const glow = document.getElementById('cursor-glow');
    document.addEventListener('mousemove', (e) => {
        glow.style.transform = `translate(${e.clientX}px, ${e.clientY}px)`;
    });

    // 2. Navigation Logic
    const views = {
        home: document.getElementById('view-home'),
        evidence: document.getElementById('view-evidence'),
        metrics: document.getElementById('view-metrics'),
        architecture: document.getElementById('view-architecture'),
        analyzer: document.getElementById('view-analyzer')
    };

    const navItems = document.querySelectorAll('.nav-item');

    window.switchView = (viewName) => {
        Object.keys(views).forEach(key => {
            if (views[key]) views[key].classList.toggle('hidden', key !== viewName);
        });
        navItems.forEach(item => {
            item.classList.toggle('active', item.dataset.view === viewName);
        });
        window.scrollTo({ top: 0, behavior: 'smooth' });
    };

    navItems.forEach(item => {
        item.addEventListener('click', () => switchView(item.dataset.view));
    });

    // 3. Analyzer Toggles (FIXED)
    const tabBtns = document.querySelectorAll('.tab-opt');
    const textInput = document.getElementById('text-input');
    const urlInput = document.getElementById('url-input');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            const mode = btn.dataset.mode;
            if (mode === 'text') {
                textInput.classList.remove('hidden');
                urlInput.classList.add('hidden');
            } else {
                textInput.classList.add('hidden');
                urlInput.classList.remove('hidden');
            }
        });
    });

    // 4. Execution Engine
    const analyzeBtn = document.getElementById('analyze-btn');
    const resultsDashboard = document.getElementById('results-dashboard');
    const welcomeMsg = document.getElementById('welcome-msg');

    analyzeBtn.addEventListener('click', async () => {
        const mode = document.querySelector('.tab-opt.active').dataset.mode;
        const payload = mode === 'text' ? { text: textInput.value } : { url: urlInput.value };
        
        if (!payload.text && !payload.url) return alert('Input required.');

        analyzeBtn.disabled = true;
        analyzeBtn.textContent = 'AUDITING...';
        document.getElementById('processing-steps').classList.remove('hidden');
        resultsDashboard.classList.add('hidden');
        welcomeMsg.classList.add('hidden');

        const logSteps = [1, 2, 3, 4];
        for (const s of logSteps) {
            const el = document.getElementById(`step-${s}`);
            el.style.color = 'var(--neon)';
            await new Promise(r => setTimeout(r, 600));
        }

        try {
            const res = await fetch('/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            renderResults(data);
        } catch (e) {
            console.error(e);
            alert('Core system error.');
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtn.textContent = 'RUN SYSTEM AUDIT';
        }
    });

    function renderResults(data) {
        resultsDashboard.classList.remove('hidden');
        resultsDashboard.scrollIntoView({ behavior: 'smooth' });

        const label = document.getElementById('verdict-label');
        label.textContent = data.verdict;
        label.style.color = data.verdict === 'FAKE' ? 'var(--fake)' : 'var(--real)';
        
        const bar = document.getElementById('trust-bar');
        bar.style.width = `${data.ensemble_score * 100}%`;
        bar.style.background = data.verdict === 'FAKE' ? 'var(--fake)' : 'var(--real)';
        bar.style.boxShadow = `0 0 30px ${data.verdict === 'FAKE' ? 'var(--fake)' : 'var(--real)'}`;
        
        document.getElementById('confidence-val').textContent = `CONFIDENCE: ${(data.confidence * 100).toFixed(1)}%`;

        renderHeatmap(data.attention_weights);

        document.getElementById('ratio-bar').style.width = `${Math.min(data.classical.adj_noun_ratio * 100, 100)}%`;
        const sentimentPos = ((data.classical.sentiment + 1) / 2) * 100;
        document.getElementById('sentiment-marker').style.left = `${sentimentPos}%`;

        const tags = document.getElementById('keyword-tags');
        tags.innerHTML = '';
        data.classical.tfidf_top.forEach(t => {
            const s = document.createElement('span');
            s.style.fontSize = '0.7rem';
            s.style.background = 'rgba(255,255,255,0.05)';
            s.style.padding = '4px 12px';
            s.style.borderRadius = '20px';
            s.style.border = '1px solid var(--border)';
            s.textContent = t;
            tags.appendChild(s);
        });
    }

    function renderHeatmap(weightsData) {
        const container = document.getElementById('heatmap-container');
        container.innerHTML = '';
        const max = Math.max(...weightsData.map(w => w.weight));
        weightsData.forEach(item => {
            const span = document.createElement('span');
            span.textContent = item.token + ' ';
            const intensity = item.weight / (max || 1);
            span.style.background = `rgba(99, 102, 241, ${intensity * 0.7})`;
            span.style.padding = '0 4px';
            container.appendChild(span);
        });
    }

    // 5. Evidence Modal Logic
    const evidenceModal = document.getElementById('evidence-modal');
    const cases = {
        'case-1': { title: 'REAL / POLITICAL', text: 'The unemployment rate in Ohio has dropped to 4.2%, its lowest level since 2001, according to data from the Bureau of Labor Statistics. The report suggests that manufacturing gains have offset losses in the service sector...' },
        'case-2': { title: 'FAKE / SENSATIONAL', text: 'NASA scientists find evidence of alien structures on Mars using high-resolution images from the Perseverance rover. The images appear to show geometric shapes that could not be natural formations...' },
        'case-3': { title: 'FAKE / PROPAGANDA', text: 'Secret documents reveal planned takeover of power grid by private military contractors. The documents, allegedly leaked from a high-level briefing, suggest that the grid will be privatized within six months...' }
    };

    window.showModal = (id) => {
        const c = cases[id];
        document.getElementById('modal-title').textContent = c.title;
        document.getElementById('modal-text').textContent = c.text;
        evidenceModal.classList.remove('hidden');
    };

    window.closeModal = () => {
        evidenceModal.classList.add('hidden');
    };
});
