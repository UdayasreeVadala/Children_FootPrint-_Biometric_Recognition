/* ============================================================
   FootprintID — Frontend Application Logic
   ============================================================ */

(function () {
    'use strict';

    // ── State ──────────────────────────────────────────────
    let enrollFiles = [];
    let identifyFile = null;
    let trainingPollTimer = null;
    let trainingActive = false;
    let datasetLoaded = false;
    let modelsLoaded = false;

    // ── DOM References ─────────────────────────────────────
    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => document.querySelectorAll(sel);

    const tabBtns = () => $$('.tab-btn');
    const tabPanel = (name) => $(`#${name}-panel`);

    const childNameInput = () => $('#child-name');
    const childIdInput = () => $('#child-id');
    const enrollDropzone = () => $('#enroll-dropzone');
    const enrollFileInput = () => $('#enroll-file-input');
    const enrollPreviews = () => $('#enroll-previews');
    const enrollBtn = () => $('#enroll-btn');
    const enrollLoading = () => $('#enroll-loading');
    const enrollResults = () => $('#enroll-results');

    const identifyDropzone = () => $('#identify-dropzone');
    const identifyFileInput = () => $('#identify-file-input');
    const identifyPreviews = () => $('#identify-previews');
    const identifyBtn = () => $('#identify-btn');
    const identifyLoading = () => $('#identify-loading');
    const identifyResults = () => $('#identify-results');

    const enrolledContainer = () => $('#enrolled-children-container');
    const refreshEnrolledBtn = () => $('#refresh-enrolled-btn');
    const modelStatus = () => $('#model-status');
    const modelDetails = () => $('#model-details');
    const retrainBtn = () => $('#retrain-btn');
    const retrainLoading = () => $('#retrain-loading');
    const retrainResults = () => $('#retrain-results');
    const openModelsBtn = () => $('#open-models-btn');

    const datasetSummary = () => $('#dataset-summary');
    const modelComparison = () => $('#model-comparison');
    const trainBtn = () => $('#train-btn');
    const trainEpochs = () => $('#train-epochs');
    const trainFineEpochs = () => $('#train-fine-epochs');
    const trainBatchSize = () => $('#train-batch-size');
    const trainIncludePhone = () => $('#train-include-phone');
    const trainProgress = () => $('#train-progress');
    const trainStage = () => $('#train-stage');
    const trainPercent = () => $('#train-percent');
    const trainProgressFill = () => $('#train-progress-fill');
    const trainDetail = () => $('#train-detail');
    const trainResults = () => $('#train-results');
    const enrollSampleBtn = () => $('#enroll-sample-btn');
    const sampleSubject = () => $('#sample-subject');
    const sampleLimit = () => $('#sample-limit');
    const enrollSampleResults = () => $('#enroll-sample-results');

    // ── Utility Functions ──────────────────────────────────

    function formatConfidence(value) {
        return `${parseFloat(value).toFixed(1)}%`;
    }

    function formatFraction(value) {
        if (value === null || value === undefined) return '—';
        return `${(parseFloat(value) * 100).toFixed(1)}%`;
    }

    function formatNumber(value) {
        if (value === null || value === undefined) return '—';
        return Number(value).toLocaleString();
    }

    function formatSeconds(value) {
        if (value === null || value === undefined) return '—';
        const seconds = Number(value);
        if (seconds < 60) return `${seconds.toFixed(1)}s`;
        return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
    }

    function getConfidenceClass(value, threshold) {
        const cutoff = (threshold || 0.70) * 100;
        if (value >= Math.max(cutoff, 80)) return 'high';
        if (value >= cutoff) return 'medium';
        return 'low';
    }

    function showLoading(container) {
        if (container) container.hidden = false;
    }

    function hideLoading(container) {
        if (container) container.hidden = true;
    }

    function createImageElement(base64data, altText) {
        const img = document.createElement('img');
        if (base64data.startsWith('data:')) {
            img.src = base64data;
        } else {
            img.src = `data:image/png;base64,${base64data}`;
        }
        img.alt = altText || 'Footprint image';
        img.loading = 'lazy';
        return img;
    }

    function buildImageBox(base64, label) {
        const box = document.createElement('div');
        box.className = 'image-box';
        box.appendChild(createImageElement(base64, label));
        const lbl = document.createElement('p');
        lbl.className = 'image-label';
        lbl.textContent = label;
        box.appendChild(lbl);
        return box;
    }

    function escapeHtml(str) {
        const div = document.createElement('div');
        div.appendChild(document.createTextNode(str === null || str === undefined ? '' : String(str)));
        return div.innerHTML;
    }

    function titleCase(value) {
        return String(value || '')
            .replace(/_/g, ' ')
            .replace(/\b\w/g, (char) => char.toUpperCase());
    }

    function taskLabel(task) {
        const labels = {
            identity: 'Child identity',
            side: 'Foot side (Left vs Right)',
            visit: 'Visit',
            sensor: 'Sensor',
        };
        return labels[task] || titleCase(task);
    }

    function setButtonText(button, text) {
        const textNode = button.querySelector('.btn-text');
        if (textNode) textNode.textContent = text;
        else button.textContent = text;
    }

    // ── Tab Switching ──────────────────────────────────────

    function activateTab(name) {
        tabBtns().forEach((button) => {
            const isActive = button.dataset.tab === name;
            button.classList.toggle('active', isActive);
            button.setAttribute('aria-selected', isActive ? 'true' : 'false');
        });
        tabBtns().forEach((button) => {
            const panel = tabPanel(button.dataset.tab);
            if (!panel) return;
            const isActive = button.dataset.tab === name;
            panel.classList.toggle('active', isActive);
            panel.hidden = !isActive;
        });

        if (name === 'models') {
            refreshDatasetInfo();
            refreshModels();
            resumeTrainingStatus();
        }
    }

    function initTabs() {
        tabBtns().forEach((button) => {
            button.addEventListener('click', () => activateTab(button.dataset.tab));
        });
    }

    // ── Drag & Drop / File Handling ────────────────────────

    function preventDefaults(event) {
        event.preventDefault();
        event.stopPropagation();
    }

    function setupDropzone(dropzone, fileInput, onFilesAdded) {
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach((event) => {
            dropzone.addEventListener(event, preventDefaults);
        });
        ['dragenter', 'dragover'].forEach((event) => {
            dropzone.addEventListener(event, () => dropzone.classList.add('dragover'));
        });
        ['dragleave', 'drop'].forEach((event) => {
            dropzone.addEventListener(event, () => dropzone.classList.remove('dragover'));
        });
        dropzone.addEventListener('drop', (event) => {
            const files = Array.from(event.dataTransfer.files).filter((file) =>
                file.type.startsWith('image/')
            );
            if (files.length) onFilesAdded(files);
        });
        fileInput.addEventListener('change', () => {
            const files = Array.from(fileInput.files).filter((file) =>
                file.type.startsWith('image/')
            );
            if (files.length) onFilesAdded(files);
            fileInput.value = '';
        });
    }

    function renderEnrollPreviews() {
        const container = enrollPreviews();
        container.innerHTML = '';
        enrollFiles.forEach((file, index) => {
            const item = document.createElement('div');
            item.className = 'preview-item';

            const img = document.createElement('img');
            img.src = URL.createObjectURL(file);
            img.alt = `Footprint preview ${index + 1}`;
            item.appendChild(img);

            const removeBtn = document.createElement('button');
            removeBtn.className = 'preview-remove';
            removeBtn.innerHTML = '✕';
            removeBtn.title = 'Remove image';
            removeBtn.setAttribute('aria-label', `Remove image ${index + 1}`);
            removeBtn.addEventListener('click', (event) => {
                event.stopPropagation();
                enrollFiles.splice(index, 1);
                renderEnrollPreviews();
                updateEnrollBtnState();
            });
            item.appendChild(removeBtn);
            container.appendChild(item);
        });
    }

    function onEnrollFilesAdded(files) {
        enrollFiles = enrollFiles.concat(files);
        renderEnrollPreviews();
        updateEnrollBtnState();
    }

    function renderIdentifyPreview() {
        const container = identifyPreviews();
        container.innerHTML = '';
        if (!identifyFile) return;

        const item = document.createElement('div');
        item.className = 'preview-item';

        const img = document.createElement('img');
        img.src = URL.createObjectURL(identifyFile);
        img.alt = 'Footprint to identify';
        item.appendChild(img);

        const removeBtn = document.createElement('button');
        removeBtn.className = 'preview-remove';
        removeBtn.innerHTML = '✕';
        removeBtn.title = 'Remove image';
        removeBtn.setAttribute('aria-label', 'Remove image');
        removeBtn.addEventListener('click', (event) => {
            event.stopPropagation();
            identifyFile = null;
            renderIdentifyPreview();
            updateIdentifyBtnState();
        });
        item.appendChild(removeBtn);
        container.appendChild(item);
    }

    function onIdentifyFilesAdded(files) {
        identifyFile = files[0];
        renderIdentifyPreview();
        updateIdentifyBtnState();
    }

    function updateEnrollBtnState() {
        const name = childNameInput().value.trim();
        const id = childIdInput().value.trim();
        enrollBtn().disabled = !(name && id && enrollFiles.length > 0);
    }

    function updateIdentifyBtnState() {
        identifyBtn().disabled = !identifyFile;
    }

    // ── Enroll API Call ────────────────────────────────────

    async function handleEnroll() {
        const name = childNameInput().value.trim();
        const id = childIdInput().value.trim();
        if (!name || !id || enrollFiles.length === 0) return;

        enrollResults().hidden = true;
        enrollResults().innerHTML = '';
        showLoading(enrollLoading());
        enrollBtn().disabled = true;

        const formData = new FormData();
        formData.append('child_name', name);
        formData.append('child_id', id);
        enrollFiles.forEach((file) => formData.append('files', file));

        try {
            const response = await fetch('/api/enroll', { method: 'POST', body: formData });
            const data = await response.json();
            if (response.ok && data.success) renderEnrollSuccess(data);
            else renderEnrollError(data.detail || data.message || 'Enrollment failed.');
        } catch (error) {
            console.error('Enroll error:', error);
            renderEnrollError('Network error. Please try again.');
        } finally {
            hideLoading(enrollLoading());
            enrollBtn().disabled = false;
            enrollFiles = [];
            renderEnrollPreviews();
            childNameInput().value = '';
            childIdInput().value = '';
            updateEnrollBtnState();
            refreshEnrolledList();
            refreshModelInfo();
        }
    }

    function renderEnrollSuccess(data) {
        const container = enrollResults();
        container.innerHTML = '';
        container.hidden = false;

        const banner = document.createElement('div');
        banner.className = 'success-message';
        banner.innerHTML = `
            <span class="success-icon">✅</span>
            <span>Successfully enrolled <strong>${escapeHtml(data.child_id || '')}</strong> with ${data.images_enrolled || 0} image(s).</span>
        `;
        container.appendChild(banner);

        (data.preprocessing_previews || []).forEach((preview, index) => {
            const card = document.createElement('div');
            card.className = 'result-card';
            card.innerHTML = `<div class="result-card-title">Preprocessing Pipeline — Image ${index + 1}</div>`;

            if (preview.error) {
                const notice = document.createElement('div');
                notice.className = 'error-message';
                notice.innerHTML = `<span>⚠️</span><span>${escapeHtml(preview.error)}</span>`;
                card.appendChild(notice);
            }

            const grid = document.createElement('div');
            grid.className = 'image-comparison pipeline';
            if (preview.original) grid.appendChild(buildImageBox(preview.original, 'Original'));
            if (preview.roi) grid.appendChild(buildImageBox(preview.roi, 'ROI Detected'));
            if (preview.enhanced) grid.appendChild(buildImageBox(preview.enhanced, 'Gabor Enhanced'));
            card.appendChild(grid);
            container.appendChild(card);
        });
    }

    function renderEnrollError(message) {
        const container = enrollResults();
        container.innerHTML = '';
        container.hidden = false;
        const banner = document.createElement('div');
        banner.className = 'error-message';
        banner.innerHTML = `<span>⚠️</span><span>${escapeHtml(message)}</span>`;
        container.appendChild(banner);
    }

    // ── Identify API Call ──────────────────────────────────

    async function handleIdentify() {
        if (!identifyFile) return;

        identifyResults().hidden = true;
        identifyResults().innerHTML = '';
        showLoading(identifyLoading());
        identifyBtn().disabled = true;

        const formData = new FormData();
        formData.append('file', identifyFile);

        try {
            const response = await fetch('/api/identify', { method: 'POST', body: formData });
            const data = await response.json();
            if (response.ok && data.success) renderIdentifyResults(data);
            else renderIdentifyError(data.detail || data.message || 'Identification failed.');
        } catch (error) {
            console.error('Identify error:', error);
            renderIdentifyError('Network error. Please try again.');
        } finally {
            hideLoading(identifyLoading());
            identifyBtn().disabled = false;
        }
    }

    function renderIdentifyResults(data) {
        const container = identifyResults();
        container.innerHTML = '';
        container.hidden = false;

        const prediction = data.prediction || {};
        const gradcam = data.gradcam || {};
        const preprocessing = data.preprocessing || {};
        const confidence = prediction.confidence_percent || 0;
        const threshold = prediction.threshold || 0.70;
        const confClass = getConfidenceClass(confidence, threshold);
        const isRecognized = prediction.is_recognized;
        const isEmbedding = prediction.match_mode === 'embedding';

        // ── Prediction Card ──
        const predCard = document.createElement('div');
        predCard.className = 'result-card';
        predCard.innerHTML = `<div class="result-card-title">Recognition Result</div>`;

        const header = document.createElement('div');
        header.className = 'prediction-header';
        header.innerHTML = `
            <div>
                <div class="predicted-name">${escapeHtml(prediction.predicted_name || 'Unknown')}</div>
                <div class="predicted-id">ID: ${escapeHtml(String(prediction.predicted_id ?? '—'))}</div>
            </div>
            <span class="status-badge ${isRecognized ? 'recognized' : 'unknown'}">
                ${isRecognized ? '✓ Recognized' : '⚠ Unknown Child'}
            </span>
        `;
        predCard.appendChild(header);

        // Mode / architecture chips
        const chips = document.createElement('div');
        chips.className = 'mode-chip-row';
        const modeText = prediction.is_demo
            ? 'Demo mode'
            : isEmbedding
                ? 'Embedding verification'
                : 'Softmax identity';
        chips.innerHTML = `
            <span class="mode-badge ${isEmbedding ? 'embedding' : 'softmax'}">${escapeHtml(modeText)}</span>
            ${prediction.architecture_label ? `<span class="mode-badge neutral">${escapeHtml(prediction.architecture_label)}</span>` : ''}
            ${prediction.task ? `<span class="mode-badge neutral">${escapeHtml(taskLabel(prediction.task))}</span>` : ''}
        `;
        predCard.appendChild(chips);

        // Confidence Meter
        const confSection = document.createElement('div');
        confSection.className = 'confidence-section';
        confSection.innerHTML = `
            <div class="confidence-label">
                <span class="confidence-label-text">${isEmbedding ? 'Calibrated confidence' : 'Confidence'}</span>
                <span class="confidence-value ${confClass}">${formatConfidence(confidence)}</span>
            </div>
            <div class="confidence-track">
                <div class="confidence-fill ${confClass}" style="width: ${Math.min(100, confidence)}%"></div>
            </div>
            <p class="threshold-note">Rejection threshold: ${formatFraction(threshold)}</p>
        `;
        predCard.appendChild(confSection);

        // Similarity + calibration details
        if (prediction.similarity !== null && prediction.similarity !== undefined) {
            const similaritySection = document.createElement('div');
            similaritySection.className = 'similarity-row';
            const calibration = prediction.calibration || {};
            similaritySection.innerHTML = `
                <div class="similarity-item">
                    <span class="similarity-label">Cosine similarity</span>
                    <span class="similarity-value">${formatConfidence(prediction.similarity)}</span>
                </div>
                <div class="similarity-item">
                    <span class="similarity-label">Enrolled children</span>
                    <span class="similarity-value">${prediction.enrolled_children || 0}</span>
                </div>
                ${
                    calibration.center !== undefined
                        ? `<div class="similarity-item">
                               <span class="similarity-label">Calibration</span>
                               <span class="similarity-value">c=${calibration.center} · s=${calibration.scale}</span>
                           </div>`
                        : ''
                }
            `;
            predCard.appendChild(similaritySection);
        }

        if (prediction.message) {
            const notice = document.createElement('div');
            notice.className = 'info-message';
            notice.innerHTML = `<span>ℹ️</span><span>${escapeHtml(prediction.message)}</span>`;
            predCard.appendChild(notice);
        }

        if (prediction.is_demo) {
            const demoNotice = document.createElement('div');
            demoNotice.className = 'info-message';
            demoNotice.innerHTML = `<span>🧪</span><span>Demo mode — results are simulated until a trained model is loaded.</span>`;
            predCard.appendChild(demoNotice);
        }

        if (prediction.top_3 && prediction.top_3.length > 0) {
            const topSection = document.createElement('div');
            topSection.className = 'top-predictions';
            topSection.innerHTML = `<div class="top-predictions-title">Top Predictions</div>`;
            const list = document.createElement('ul');
            list.className = 'prediction-list';

            prediction.top_3.forEach((item, index) => {
                const li = document.createElement('li');
                li.className = 'prediction-list-item';
                const similarity = item.similarity !== null && item.similarity !== undefined
                    ? `<span class="similarity-inline">sim ${formatConfidence(item.similarity)}</span>`
                    : '';
                li.innerHTML = `
                    <span class="rank">#${index + 1}</span>
                    <span class="name">${escapeHtml(item.name || '—')} · ${escapeHtml(item.child_id || '—')}</span>
                    ${similarity}
                    <span class="conf">${formatConfidence(item.confidence || 0)}</span>
                `;
                list.appendChild(li);
            });
            topSection.appendChild(list);
            predCard.appendChild(topSection);
        }

        container.appendChild(predCard);

        // ── Grad-CAM Card ──
        if (gradcam.overlay || gradcam.heatmap || gradcam.raw_heatmap) {
            const gcCard = document.createElement('div');
            gcCard.className = 'result-card';
            gcCard.innerHTML = `<div class="result-card-title">Grad-CAM Visualization</div>`;

            const grid = document.createElement('div');
            grid.className = 'image-comparison pipeline';
            if (gradcam.original) grid.appendChild(buildImageBox(gradcam.original, 'Preprocessed input'));
            if (gradcam.raw_heatmap) grid.appendChild(buildImageBox(gradcam.raw_heatmap, 'Raw heatmap'));
            if (gradcam.heatmap) grid.appendChild(buildImageBox(gradcam.heatmap, 'Colour heatmap'));
            if (gradcam.overlay) grid.appendChild(buildImageBox(gradcam.overlay, 'Overlay'));
            gcCard.appendChild(grid);

            const caption = document.createElement('p');
            caption.className = 'image-caption';
            caption.textContent = gradcam.method === 'embedding_similarity'
                ? 'Grad-CAM of the embedding similarity against the matched child: bright regions pushed the verification score up.'
                : 'Highlighted regions show the areas the model focused on for recognition — typically crease patterns and ridge details.';
            gcCard.appendChild(caption);
            container.appendChild(gcCard);
        } else if (gradcam.error) {
            const gcNotice = document.createElement('div');
            gcNotice.className = 'error-message';
            gcNotice.innerHTML = `<span>ℹ️</span><span>Grad-CAM is unavailable: ${escapeHtml(gradcam.error)}</span>`;
            container.appendChild(gcNotice);
        }

        // ── Preprocessing Pipeline Card ──
        if (preprocessing.original || preprocessing.roi || preprocessing.enhanced) {
            const ppCard = document.createElement('div');
            ppCard.className = 'result-card';
            ppCard.innerHTML = `<div class="result-card-title">Preprocessing Pipeline</div>`;

            const grid = document.createElement('div');
            grid.className = 'image-comparison pipeline';
            if (preprocessing.original) grid.appendChild(buildImageBox(preprocessing.original, 'Original'));
            if (preprocessing.roi) grid.appendChild(buildImageBox(preprocessing.roi, 'ROI Detected'));
            if (preprocessing.enhanced) grid.appendChild(buildImageBox(preprocessing.enhanced, 'Gabor Enhanced'));
            ppCard.appendChild(grid);
            container.appendChild(ppCard);
        }
    }

    function renderIdentifyError(message) {
        const container = identifyResults();
        container.innerHTML = '';
        container.hidden = false;
        const banner = document.createElement('div');
        banner.className = 'error-message';
        banner.innerHTML = `<span>⚠️</span><span>${escapeHtml(message)}</span>`;
        container.appendChild(banner);
    }

    // ── Enrolled Children List ─────────────────────────────

    async function refreshEnrolledList() {
        const container = enrolledContainer();
        try {
            const response = await fetch('/api/enrolled');
            const data = await response.json();
            container.innerHTML = '';

            if (!data.children || data.children.length === 0) {
                container.innerHTML = '<p class="empty-state">No children enrolled yet.</p>';
                return;
            }

            data.children.forEach((child) => {
                const item = document.createElement('div');
                item.className = 'enrolled-child-item';
                item.innerHTML = `
                    <div class="child-info">
                        <span class="child-name">${escapeHtml(child.child_name || child.child_id)}</span>
                        <span class="child-meta">${escapeHtml(child.child_id)}</span>
                    </div>
                    <span class="child-image-count">📷 ${child.image_count}</span>
                `;

                const deleteBtn = document.createElement('button');
                deleteBtn.className = 'btn-danger-sm';
                deleteBtn.innerHTML = '✕';
                deleteBtn.title = `Remove ${child.child_name || child.child_id}`;
                deleteBtn.setAttribute('aria-label', `Delete ${child.child_name || child.child_id}`);
                deleteBtn.addEventListener('click', () => deleteChild(child.child_id));
                item.appendChild(deleteBtn);
                container.appendChild(item);
            });
        } catch (error) {
            console.error('Failed to fetch enrolled list:', error);
            container.innerHTML = '<p class="empty-state">Could not load enrolled children.</p>';
        }
    }

    async function deleteChild(childId) {
        if (!confirm(`Remove enrolled child "${childId}"? This cannot be undone.`)) return;
        try {
            await fetch(`/api/enrolled/${encodeURIComponent(childId)}`, { method: 'DELETE' });
        } catch (error) {
            console.error('Delete error:', error);
        }
        refreshEnrolledList();
        refreshModelInfo();
    }

    // ── Model Lifecycle ───────────────────────────────────

    async function refreshModelInfo() {
        try {
            const response = await fetch('/api/model-info');
            const info = await response.json();
            if (!response.ok) throw new Error(info.detail || 'Could not load model status.');

            modelStatus().textContent = info.is_demo
                ? 'Demo'
                : (info.architecture_label || 'Trained');
            modelStatus().classList.toggle('trained', !info.is_demo);

            const mode = info.match_mode === 'embedding' ? 'verification' : (info.match_mode || 'softmax');
            modelDetails().textContent = [
                info.is_demo ? 'Simulated results' : `${info.architecture_label || info.architecture}`,
                `${mode} mode`,
                `${(info.threshold * 100).toFixed(0)}% threshold`,
                `${info.enrolled_children || 0} enrolled`,
            ].join(' · ');
        } catch (error) {
            modelStatus().textContent = 'Unavailable';
            modelStatus().classList.remove('trained');
            modelDetails().textContent = 'Model status could not be loaded.';
        }
    }

    function renderRetrainMessage(message, isError = false) {
        retrainResults().innerHTML = `
            <div class="${isError ? 'error-message' : 'success-message'}">
                <span>${isError ? '⚠️' : '✓'}</span><span>${escapeHtml(message)}</span>
            </div>`;
    }

    async function handleRetrain() {
        retrainResults().innerHTML = '';
        retrainBtn().disabled = true;
        showLoading(retrainLoading());
        try {
            const response = await fetch('/api/retrain?epochs=12', { method: 'POST' });
            const data = await response.json();
            if (!response.ok || !data.success) {
                throw new Error(data.detail || 'Training could not be started.');
            }
            renderRetrainMessage(
                `Training complete: ${data.children_used.length} children, ${data.images_used} images, ${data.training_accuracy}% accuracy.`
            );
            await refreshModelInfo();
            await refreshModels();
        } catch (error) {
            renderRetrainMessage(error.message || 'Training failed. Please try again.', true);
        } finally {
            hideLoading(retrainLoading());
            retrainBtn().disabled = false;
        }
    }

    // ── Dataset Overview ──────────────────────────────────

    async function refreshDatasetInfo(force = false) {
        if (datasetLoaded && !force) return;
        const container = datasetSummary();
        try {
            const response = await fetch('/api/dataset-info');
            const data = await response.json();
            if (!response.ok) throw new Error(data.detail || 'Dataset could not be inspected.');
            datasetLoaded = true;
            renderDatasetSummary(data);
        } catch (error) {
            container.innerHTML = `<div class="error-message"><span>⚠️</span><span>${escapeHtml(error.message)}</span></div>`;
        }
    }

    function renderDatasetSummary(data) {
        const container = datasetSummary();
        const stats = [
            ['Images', data.total_images, 'total usable scans'],
            ['Subjects', data.subject_count, (data.subjects || []).join(', ') || '—'],
            ['Sides', (data.sides || []).length, (data.sides || []).join(', ') || '—'],
            ['Visits', (data.visits || []).length, (data.visits || []).join(', ') || '—'],
            ['Sensors', (data.sensors || []).length, (data.sensors || []).join(', ') || '—'],
        ];

        const statHtml = stats
            .map(
                ([label, value, detail]) => `
                <div class="dataset-stat">
                    <span class="dataset-value">${escapeHtml(value)}</span>
                    <span class="dataset-label">${escapeHtml(label)}</span>
                    <span class="dataset-detail">${escapeHtml(detail)}</span>
                </div>`
            )
            .join('');

        const visits = data.by_visit || {};
        const visitRows = Object.entries(visits)
            .sort((a, b) => a[0].localeCompare(b[0], undefined, { numeric: true }))
            .map(([visit, counts]) => {
                const total = counts.total || 0;
                const maximum = Math.max(...Object.values(visits).map((entry) => entry.total || 0), 1);
                return `
                    <div class="visit-row">
                        <span class="visit-name">${escapeHtml(visit)}</span>
                        <div class="visit-track"><div class="visit-fill" style="width:${(total / maximum) * 100}%"></div></div>
                        <span class="visit-count">${total} · L${counts.left || 0}/R${counts.right || 0}</span>
                    </div>`;
            })
            .join('');

        container.innerHTML = `
            <div class="dataset-stats">${statHtml}</div>
            <div class="dataset-meta">
                <span class="mode-badge neutral">Task: ${escapeHtml(taskLabel(data.task))}</span>
                <span class="mode-badge ${data.cache_exists ? 'softmax' : 'neutral'}">${data.cache_exists ? 'Preprocessing cache ready' : 'Cache will be built on first training run'}</span>
                <span class="mode-badge ${data.registry_exists ? 'softmax' : 'neutral'}">${data.registry_exists ? 'Trained registry found' : 'No trained registry yet'}</span>
            </div>
            ${visitRows ? `<div class="visit-breakdown">${visitRows}</div>` : ''}
        `;
    }

    // ── Model Comparison ──────────────────────────────────

    async function refreshModels(force = false) {
        if (modelsLoaded && !force) return;
        const container = modelComparison();
        try {
            const response = await fetch('/api/models');
            const data = await response.json();
            modelsLoaded = true;
            renderModelComparison(data);
        } catch (error) {
            container.innerHTML = `<div class="error-message"><span>⚠️</span><span>${escapeHtml(error.message)}</span></div>`;
        }
    }

    function renderModelComparison(data) {
        const container = modelComparison();
        const architectures = data.architectures || {};
        const entries = Object.entries(architectures);

        if (!data.has_registry || entries.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <p>${escapeHtml(data.message || 'No dataset-trained models yet. Start training above.')}</p>
                </div>`;
            return;
        }

        const activeModel = data.active_model || data.best_model;
        const task = data.task;

        entries.sort((a, b) => {
            const left = a[1].test_accuracy || 0;
            const right = b[1].test_accuracy || 0;
            return right - left;
        });

        const cards = entries
            .map(([key, metrics], index) => {
                const accuracy = metrics.test_accuracy;
                const isActive = key === activeModel;
                const isBest = key === data.best_model;
                const barWidth = accuracy === null || accuracy === undefined ? 0 : Math.min(100, accuracy * 100);
                const initialisation = metrics.initialization === 'imagenet'
                    ? 'ImageNet transfer'
                    : metrics.initialization === 'scratch'
                        ? 'From scratch'
                        : '—';
                const calibration = (data.calibration_report || {})[key];
                const calibrationLine = calibration
                    ? `<div class="model-calibration">
                           calibrated c=${calibration.center} · FAR ${formatFraction(calibration.false_accept_rate)} · FRR ${formatFraction(calibration.false_reject_rate)}
                       </div>`
                    : '';
                return `
                    <article class="model-card ${isActive ? 'active' : ''}">
                        <div class="model-card-head">
                            <div>
                                <div class="model-rank">#${index + 1}</div>
                                <h4 class="model-name">${escapeHtml(metrics.architecture_label || titleCase(key))}</h4>
                                <div class="model-subtitle">${escapeHtml(initialisation)} · ${formatNumber(metrics.parameters)} params</div>
                            </div>
                            <div class="model-badges">
                                ${isActive ? '<span class="model-badge active">Active</span>' : ''}
                                ${isBest && !isActive ? '<span class="model-badge best">Best</span>' : ''}
                            </div>
                        </div>

                        <div class="model-metric-grid">
                            <div class="model-metric">
                                <span class="metric-value">${formatFraction(accuracy)}</span>
                                <span class="metric-label">Test accuracy</span>
                            </div>
                            <div class="model-metric">
                                <span class="metric-value">${formatFraction(metrics.test_top3_accuracy)}</span>
                                <span class="metric-label">Top-3</span>
                            </div>
                            <div class="model-metric">
                                <span class="metric-value">${formatFraction(metrics.test_macro_f1)}</span>
                                <span class="metric-label">Macro F1</span>
                            </div>
                            <div class="model-metric">
                                <span class="metric-value">${formatSeconds(metrics.training_seconds)}</span>
                                <span class="metric-label">Train time</span>
                            </div>
                        </div>

                        <div class="metric-bar">
                            <div class="metric-bar-fill" style="width:${barWidth}%"></div>
                        </div>
                        ${calibrationLine}

                        <div class="model-foot">
                            <span>${metrics.epochs_completed || 0} epochs · ${metrics.test_images || 0} test images</span>
                            <button type="button" class="btn btn-secondary btn-sm select-model-btn ${isActive ? 'is-active' : ''}"
                                    data-architecture="${escapeHtml(key)}" ${isActive ? 'disabled' : ''}>
                                ${isActive ? 'In use' : 'Use this model'}
                            </button>
                        </div>
                        <div class="model-artifacts" data-architecture="${escapeHtml(key)}"></div>
                    </article>`;
            })
            .join('');

        const note = task && task !== 'identity'
            ? `<div class="info-message">
                   <span>ℹ️</span>
                   <span>The dataset contains a single child, so models are trained on the
                   <strong>${escapeHtml(taskLabel(task))}</strong> task. Their 256-D penultimate
                   embeddings are used for child verification with cosine similarity.</span>
               </div>`
            : '';

        container.innerHTML = `${note}<div class="model-cards">${cards}</div>`;
        loadModelArtifacts();
    }

    async function loadModelArtifacts() {
        const targets = Array.from(document.querySelectorAll('.model-artifacts'));
        await Promise.allSettled(
            targets.map(async (target) => {
                const architecture = target.dataset.architecture;
                const kinds = [
                    ['training', 'Training curves'],
                    ['confusion', 'Confusion matrix'],
                ];
                const results = await Promise.allSettled(
                    kinds.map(async ([kind, label]) => {
                        const response = await fetch(
                            `/api/models/artifact?architecture=${encodeURIComponent(architecture)}&kind=${kind}`
                        );
                        if (!response.ok) throw new Error('unavailable');
                        const data = await response.json();
                        return { ...data, label };
                    })
                );

                results.forEach((result) => {
                    if (result.status !== 'fulfilled' || !result.value.image) return;
                    const box = document.createElement('div');
                    box.className = 'model-artifact';
                    const img = document.createElement('img');
                    img.src = `data:image/png;base64,${result.value.image}`;
                    img.alt = `${result.value.label} for ${architecture}`;
                    img.loading = 'lazy';
                    const caption = document.createElement('p');
                    caption.className = 'image-label';
                    caption.textContent = result.value.label;
                    box.appendChild(img);
                    box.appendChild(caption);
                    target.appendChild(box);
                });
            })
        );
    }

    async function handleSelectModel(architecture) {
        try {
            const response = await fetch(
                `/api/models/select?architecture=${encodeURIComponent(architecture)}`,
                { method: 'POST' }
            );
            const data = await response.json();
            if (!response.ok || !data.success) {
                throw new Error(data.detail || 'Could not activate the model.');
            }
            await refreshModelInfo();
            await refreshModels(true);
            trainResults().innerHTML = `
                <div class="success-message"><span>✓</span><span>Active model switched to ${escapeHtml(architecture)}.</span></div>`;
        } catch (error) {
            trainResults().innerHTML = `
                <div class="error-message"><span>⚠️</span><span>${escapeHtml(error.message)}</span></div>`;
        }
    }

    // ── Training ──────────────────────────────────────────

    function selectedArchitectures() {
        return Array.from(
            document.querySelectorAll('#architecture-checkboxes input[type="checkbox"]:checked')
        ).map((checkbox) => checkbox.value);
    }

    function setTrainingControlsDisabled(disabled) {
        trainBtn().disabled = disabled;
        [trainEpochs(), trainFineEpochs(), trainBatchSize(), trainIncludePhone()].forEach((input) => {
            if (input) input.disabled = disabled;
        });
        document.querySelectorAll('#architecture-checkboxes input').forEach((input) => {
            input.disabled = disabled;
        });
    }

    async function handleTrain() {
        const architectures = selectedArchitectures();
        if (architectures.length === 0) {
            trainResults().innerHTML = `
                <div class="error-message"><span>⚠️</span><span>Select at least one architecture.</span></div>`;
            return;
        }

        trainResults().innerHTML = '';
        trainingActive = true;
        setTrainingControlsDisabled(true);
        trainProgress().hidden = false;
        trainStage().textContent = 'Starting training…';
        trainPercent().textContent = '0%';
        trainProgressFill().style.width = '2%';
        trainProgressFill().style.background = '';
        trainProgressFill().classList.remove('indeterminate');
        trainDetail().textContent = 'Preparing the dataset and starting the background job.';

        const params = new URLSearchParams({
            architectures: architectures.join(','),
            epochs: trainEpochs().value || '18',
            fine_tune_epochs: trainFineEpochs().value || '6',
            batch_size: trainBatchSize().value || '16',
            include_phone: trainIncludePhone().checked ? 'true' : 'false',
        });

        try {
            const response = await fetch(`/api/train?${params.toString()}`, { method: 'POST' });
            const data = await response.json();
            if (response.status === 409) {
                trainDetail().textContent = 'A training job is already running — attaching to it.';
            } else if (!response.ok) {
                throw new Error(data.detail || 'Training could not be started.');
            }
            startTrainingPolling();
        } catch (error) {
            trainingActive = false;
            setTrainingControlsDisabled(false);
            trainProgress().hidden = true;
            trainResults().innerHTML = `
                <div class="error-message"><span>⚠️</span><span>${escapeHtml(error.message)}</span></div>`;
        }
    }

    function startTrainingPolling() {
        if (trainingPollTimer) return;
        trainingActive = true;
        setTrainingControlsDisabled(true);
        trainProgress().hidden = false;

        const tick = async () => {
            try {
                const response = await fetch('/api/train/status');
                const status = await response.json();
                renderTrainingStatus(status);
                if (!status.running) {
                    stopTrainingPolling(status);
                }
            } catch (error) {
                console.error('Training status error:', error);
            }
        };
        tick();
        trainingPollTimer = setInterval(tick, 2000);
    }

    function stopTrainingPolling(status) {
        if (trainingPollTimer) {
            clearInterval(trainingPollTimer);
            trainingPollTimer = null;
        }
        trainingActive = false;
        setTrainingControlsDisabled(false);

        if (!status) return;
        if (status.stage === 'complete') {
            trainProgressFill().style.width = '100%';
            trainPercent().textContent = '100%';
            trainStage().textContent = 'Training complete';
            trainDetail().textContent = '';
            const best = status.result && status.result.best_model;
            trainResults().innerHTML = `
                <div class="success-message"><span>✅</span><span>
                    Training complete${best ? ` — best model: <strong>${escapeHtml(titleCase(best))}</strong>` : ''}.
                </span></div>`;
            refreshModels(true);
            refreshDatasetInfo(true);
            refreshModelInfo();
        } else if (status.stage === 'failed') {
            trainStage().textContent = 'Training failed';
            trainProgressFill().classList.remove('indeterminate');
            trainProgressFill().style.width = '100%';
            trainProgressFill().style.background = 'linear-gradient(90deg, #e03e4e, #ff4757)';
            trainPercent().textContent = '—';
            trainDetail().textContent = status.error || '';
            trainResults().innerHTML = `
                <div class="error-message"><span>⚠️</span><span>${escapeHtml(status.error || 'Training failed.')}</span></div>`;
        }
    }

    function resumeTrainingStatus() {
        fetch('/api/train/status')
            .then((response) => response.json())
            .then((status) => {
                if (status.running) {
                    startTrainingPolling();
                }
            })
            .catch(() => {});
    }

    function renderTrainingStatus(status) {
        if (!status || status.stage === 'idle') return;

        trainProgress().hidden = false;
        const events = status.events || [];
        const last = events.length ? events[events.length - 1] : {};
        const stage = status.stage || 'starting';
        let percent = 0;
        let detail = status.detail || '';
        let stageText = titleCase(stage);
        let indeterminate = false;

        if (stage === 'starting') {
            percent = 2;
            stageText = 'Starting training…';
            detail = 'Loading TensorFlow and preparing the trainer.';
        } else if (stage === 'discovered') {
            percent = 6;
            stageText = 'Dataset discovered';
            detail = `${last.dataset ? last.dataset.total_images || '' : ''} images · task ${taskLabel(last.task)}`;
        } else if (stage === 'preprocessing') {
            const total = status.total || last.total || 0;
            const current = status.current || last.current || 0;
            percent = total ? 5 + (current / total) * 15 : 5;
            stageText = 'Preprocessing images';
            detail = total ? `${current}/${total} · ${status.detail || ''}` : status.detail || '';
        } else if (stage === 'split') {
            percent = 22;
            stageText = 'Building grouped split';
            detail = `Train/validation/test split by capture group.`;
        } else if (stage === 'training') {
            const index = last.index || 0;
            const total = last.architectures_total || last.total || 1;
            percent = 22 + (index / Math.max(total, 1)) * 74;
            stageText = `Training ${last.architecture_label || titleCase(last.architecture || '')}`;
            detail = `Architecture ${index + 1} of ${total}.`;
            indeterminate = true;
        } else if (stage === 'epoch') {
            const index = last.index || 0;
            const total = last.architectures_total || 1;
            const epochFraction = last.total_epochs ? (last.epoch || 0) / last.total_epochs : 0;
            percent = 22 + ((index + epochFraction) / Math.max(total, 1)) * 74;
            const phase = last.phase === 'fine_tune' ? 'fine-tuning' : 'head';
            stageText = `${last.architecture_label || titleCase(last.architecture || '')} · ${phase}`;
            const logs = last.logs || {};
            const metrics = [];
            if (logs.accuracy !== undefined) metrics.push(`acc ${formatFraction(logs.accuracy)}`);
            if (logs.val_accuracy !== undefined) metrics.push(`val ${formatFraction(logs.val_accuracy)}`);
            if (logs.loss !== undefined) metrics.push(`loss ${Number(logs.loss).toFixed(3)}`);
            detail = `Epoch ${last.epoch || 0}/${last.total_epochs || '?'}${metrics.length ? ' · ' + metrics.join(' · ') : ''}`;
        } else if (stage === 'architecture_complete') {
            const metrics = last.metrics || {};
            const index = last.index || 0;
            const total = last.architectures_total || 1;
            percent = 22 + ((index + 1) / Math.max(total, 1)) * 74;
            stageText = `${metrics.architecture_label || titleCase(last.architecture || '')} complete`;
            detail = `Test accuracy ${formatFraction(metrics.test_accuracy)} · F1 ${formatFraction(metrics.test_macro_f1)}`;
        } else if (stage === 'complete') {
            percent = 100;
            stageText = 'Training complete';
            detail = '';
            trainProgressFill().style.background = '';
        } else if (stage === 'failed') {
            percent = 100;
            stageText = 'Training failed';
            detail = status.error || '';
            trainProgressFill().style.background = 'linear-gradient(90deg, #e03e4e, #ff4757)';
        }

        trainStage().textContent = stageText;
        trainPercent().textContent = `${Math.round(percent)}%`;
        trainProgressFill().style.width = `${Math.min(100, Math.max(2, percent))}%`;
        trainProgressFill().classList.toggle('indeterminate', indeterminate && status.running);
        trainDetail().textContent = detail;
    }

    // ── Dataset Sample Enrollment ─────────────────────────

    async function handleEnrollSample() {
        enrollSampleResults().innerHTML = '';
        enrollSampleBtn().disabled = true;
        setButtonText(enrollSampleBtn(), 'Enrolling…');

        const params = new URLSearchParams({
            subject: sampleSubject().value.trim(),
            limit: sampleLimit().value || '12',
        });

        try {
            const response = await fetch(`/api/enroll-sample?${params.toString()}`, { method: 'POST' });
            const data = await response.json();
            if (!response.ok || !data.success) {
                throw new Error(data.detail || 'Sample enrollment failed.');
            }
            const previews = (data.preprocessing_previews || [])
                .slice(0, 3)
                .map(
                    (preview) => `
                    <div class="image-box">
                        ${preview.enhanced ? `<img src="data:image/png;base64,${preview.enhanced}" alt="Enhanced sample">` : ''}
                        <p class="image-label">Enhanced sample</p>
                    </div>`
                )
                .join('');

            enrollSampleResults().innerHTML = `
                <div class="success-message"><span>✅</span><span>
                    Enrolled <strong>${escapeHtml(data.child_id)}</strong> with ${data.images_enrolled} images.
                </span></div>
                ${previews ? `<div class="image-comparison pipeline">${previews}</div>` : ''}
                <p class="threshold-note">Now switch to <strong>Identify</strong> and upload a footprint — verification uses the active model's embedding.</p>
            `;
            refreshEnrolledList();
            refreshModelInfo();
        } catch (error) {
            enrollSampleResults().innerHTML = `
                <div class="error-message"><span>⚠️</span><span>${escapeHtml(error.message)}</span></div>`;
        } finally {
            enrollSampleBtn().disabled = false;
            setButtonText(enrollSampleBtn(), 'Enroll Dataset Sample');
        }
    }

    // ── Initialization ─────────────────────────────────────

    document.addEventListener('DOMContentLoaded', () => {
        initTabs();

        setupDropzone(enrollDropzone(), enrollFileInput(), onEnrollFilesAdded);
        setupDropzone(identifyDropzone(), identifyFileInput(), onIdentifyFilesAdded);

        childNameInput().addEventListener('input', updateEnrollBtnState);
        childIdInput().addEventListener('input', updateEnrollBtnState);

        enrollBtn().addEventListener('click', handleEnroll);
        identifyBtn().addEventListener('click', handleIdentify);
        refreshEnrolledBtn().addEventListener('click', refreshEnrolledList);

        retrainBtn().addEventListener('click', handleRetrain);
        openModelsBtn().addEventListener('click', () => activateTab('models'));

        trainBtn().addEventListener('click', handleTrain);
        enrollSampleBtn().addEventListener('click', handleEnrollSample);

        modelComparison().addEventListener('click', (event) => {
            const button = event.target.closest('.select-model-btn');
            if (button && !button.disabled) {
                handleSelectModel(button.dataset.architecture);
            }
        });

        refreshEnrolledList();
        refreshModelInfo();
        refreshDatasetInfo();
        refreshModels();
        resumeTrainingStatus();
    });
})();
