/**
 * LesionLens — Glass UI Frontend Controller
 * Vanilla JavaScript (Zero External Dependencies)
 */

document.addEventListener('DOMContentLoaded', () => {
  // Application State
  const state = {
    selectedFile: null,
    selectedPresetId: null,
    activeView: 'split',
    analysisResult: null,
    isDraggingSlider: false,
  };

  // DOM Elements
  const modelPill = document.getElementById('modelPill');
  const deviceText = document.getElementById('deviceText');
  const ckptText = document.getElementById('ckptText');
  const modelParams = document.getElementById('modelParams');
  const bestDiceVal = document.getElementById('bestDiceVal');

  const presetsRow = document.getElementById('presetsRow');
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');
  const dropzoneEmpty = document.getElementById('dropzoneEmpty');
  const dropzonePreview = document.getElementById('dropzonePreview');
  const previewImg = document.getElementById('previewImg');
  const clearBtn = document.getElementById('clearBtn');
  const previewMeta = document.getElementById('previewMeta');

  const thresholdSlider = document.getElementById('thresholdSlider');
  const thresholdValue = document.getElementById('thresholdValue');
  const ttaToggle = document.getElementById('ttaToggle');
  const analyzeBtn = document.getElementById('analyzeBtn');
  const analyzeSpinner = document.getElementById('analyzeSpinner');

  const placeholderStage = document.getElementById('placeholderStage');
  const analysisStage = document.getElementById('analysisStage');

  const viewTabs = document.getElementById('viewTabs');
  const splitViewContainer = document.getElementById('splitViewContainer');
  const singleViewContainer = document.getElementById('singleViewContainer');
  const splitImgBefore = document.getElementById('splitImgBefore');
  const splitImgAfter = document.getElementById('splitImgAfter');
  const splitLayerAfter = document.getElementById('splitLayerAfter');
  const sliderHandle = document.getElementById('sliderHandle');
  const singleViewImg = document.getElementById('singleViewImg');
  const singleLayerCaption = document.getElementById('singleLayerCaption');

  const inferenceTimeTag = document.getElementById('inferenceTimeTag');
  const coverageText = document.getElementById('coverageText');
  const pixelCountText = document.getElementById('pixelCountText');
  const coverageFill = document.getElementById('coverageFill');

  const downloadMaskBtn = document.getElementById('downloadMaskBtn');
  const downloadOverlayBtn = document.getElementById('downloadOverlayBtn');
  const exportAuditBtn = document.getElementById('exportAuditBtn');

  // ABCDE Elements
  const riskBadge = document.getElementById('riskBadge');
  const riskText = document.getElementById('riskText');
  const asymScore = document.getElementById('asymScore');
  const asymH = document.getElementById('asymH');
  const asymV = document.getElementById('asymV');
  const borderScore = document.getElementById('borderScore');
  const borderLabel = document.getElementById('borderLabel');
  const compactnessVal = document.getElementById('compactnessVal');
  const perimeterVal = document.getElementById('perimeterVal');
  const colorScore = document.getElementById('colorScore');
  const rgbStdVal = document.getElementById('rgbStdVal');
  const tonesVal = document.getElementById('tonesVal');
  const diameterMm = document.getElementById('diameterMm');
  const majorAxisVal = document.getElementById('majorAxisVal');
  const minorAxisVal = document.getElementById('minorAxisVal');
  const uncertScore = document.getElementById('uncertScore');
  const highUncertVal = document.getElementById('highUncertVal');
  const ttaViewsVal = document.getElementById('ttaViewsVal');
  const advisoryDesc = document.getElementById('advisoryDesc');

  // 1. Initialize System Health & Presets
  initHealth();
  initPresets();

  async function initHealth() {
    try {
      const res = await fetch('/api/health');
      const data = await res.json();
      if (data.status === 'healthy') {
        deviceText.textContent = data.device.toUpperCase();
        ckptText.textContent = data.checkpoint_name;
        if (data.total_params_m) {
          modelParams.textContent = `${data.total_params_m}M`;
        }
        if (data.best_val_dice && data.best_val_dice > 0) {
          bestDiceVal.textContent = data.best_val_dice.toFixed(4);
        }
      } else {
        deviceText.textContent = 'Degraded';
      }
    } catch (e) {
      console.warn('Health check failed:', e);
      deviceText.textContent = 'Ready';
    }
  }

  async function initPresets() {
    try {
      const res = await fetch('/api/presets');
      const data = await res.json();
      presetsRow.innerHTML = '';
      if (!data.presets || data.presets.length === 0) {
        presetsRow.innerHTML = '<span class="dim">No presets found</span>';
        return;
      }
      data.presets.forEach((preset, index) => {
        const card = document.createElement('div');
        card.className = 'preset-card';
        card.title = `${preset.id}: ${preset.title}`;
        card.innerHTML = `<img src="${preset.url}" alt="${preset.id}" loading="lazy">`;
        card.addEventListener('click', () => selectPreset(preset, card));
        presetsRow.appendChild(card);
      });
    } catch (e) {
      console.error('Failed to load presets:', e);
      presetsRow.innerHTML = '<span class="dim">Failed to load presets</span>';
    }
  }

  // 2. Preset Selection
  function selectPreset(preset, cardEl) {
    document.querySelectorAll('.preset-card').forEach(c => c.classList.remove('active'));
    cardEl.classList.add('active');

    state.selectedPresetId = preset.id;
    state.selectedFile = null;

    previewImg.src = preset.url;
    previewMeta.textContent = `${preset.id} (${preset.title})`;
    dropzoneEmpty.style.display = 'none';
    dropzonePreview.style.display = 'block';

    analyzeBtn.disabled = false;
  }

  // 3. Dropzone & File Upload Handling
  dropzone.addEventListener('click', (e) => {
    if (e.target.closest('#clearBtn')) return;
    fileInput.click();
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleUserFile(e.target.files[0]);
    }
  });

  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });

  dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('dragover');
  });

  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleUserFile(e.dataTransfer.files[0]);
    }
  });

  clearBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    resetUpload();
  });

  function handleUserFile(file) {
    if (!file.type.startsWith('image/')) {
      alert('Please upload a valid image file (JPG, PNG).');
      return;
    }
    state.selectedFile = file;
    state.selectedPresetId = null;
    document.querySelectorAll('.preset-card').forEach(c => c.classList.remove('active'));

    const reader = new FileReader();
    reader.onload = (event) => {
      previewImg.src = event.target.result;
      previewMeta.textContent = `${file.name} (${Math.round(file.size / 1024)} KB)`;
      dropzoneEmpty.style.display = 'none';
      dropzonePreview.style.display = 'block';
      analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  function resetUpload() {
    state.selectedFile = null;
    state.selectedPresetId = null;
    fileInput.value = '';
    previewImg.src = '';
    dropzonePreview.style.display = 'none';
    dropzoneEmpty.style.display = 'block';
    analyzeBtn.disabled = true;
    document.querySelectorAll('.preset-card').forEach(c => c.classList.remove('active'));
  }

  // 4. Threshold Slider
  thresholdSlider.addEventListener('input', (e) => {
    const val = parseFloat(e.target.value).toFixed(2);
    thresholdValue.textContent = val;
  });

  // 5. Submit Analysis
  analyzeBtn.addEventListener('click', async () => {
    if (!state.selectedFile && !state.selectedPresetId) return;

    analyzeBtn.disabled = true;
    analyzeSpinner.style.display = 'inline-block';

    const formData = new FormData();
    if (state.selectedFile) {
      formData.append('file', state.selectedFile);
    } else if (state.selectedPresetId) {
      formData.append('preset_id', state.selectedPresetId);
    }

    formData.append('threshold', thresholdSlider.value);
    formData.append('use_tta', ttaToggle.checked ? 'true' : 'false');

    try {
      const response = await fetch('/api/analyze', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || 'Segmentation failed');
      }

      const result = await response.json();
      state.analysisResult = result;
      renderAnalysis(result);

    } catch (err) {
      alert(`Error analyzing image: ${err.message}`);
      console.error(err);
    } finally {
      analyzeBtn.disabled = false;
      analyzeSpinner.style.display = 'none';
    }
  });

  // 6. Render Results & Diagnostics
  function renderAnalysis(data) {
    placeholderStage.style.display = 'none';
    analysisStage.style.display = 'flex';

    // Inference duration tag
    inferenceTimeTag.innerHTML = `<span>⏱️ ${data.metadata.inference_duration_s}s · ${data.metadata.tta_views} TTA views</span>`;

    // Visualizations
    const vis = data.visualizations;
    splitImgBefore.src = vis.original;
    splitImgAfter.src = vis.overlay;

    // Reset split slider to 50%
    setSplitSlider(50);

    // Default to split view
    setActiveView('split');

    // Coverage & Pixels
    const morph = data.morphology;
    coverageText.textContent = `${morph.coverage_pct}%`;
    pixelCountText.textContent = `${morph.area_pixels.toLocaleString()} / ${data.metadata.original_width * data.metadata.original_height} pixels`;
    coverageFill.style.width = `${Math.min(100, morph.coverage_pct)}%`;

    // Risk Assessment Badge
    const risk = morph.risk_assessment;
    riskBadge.className = `risk-badge ${risk.badge || 'success'}`;
    riskText.textContent = risk.level;
    advisoryDesc.textContent = risk.description;

    // Card A: Asymmetry
    asymScore.textContent = `${morph.asymmetry.score}%`;
    asymH.textContent = `${morph.asymmetry.horizontal}%`;
    asymV.textContent = `${morph.asymmetry.vertical}%`;

    // Card B: Border
    borderScore.textContent = `${morph.border.score}%`;
    borderLabel.textContent = morph.border.regularity;
    compactnessVal.textContent = morph.compactness.toFixed(3);
    perimeterVal.textContent = `${morph.perimeter_pixels.toLocaleString()} px`;

    // Card C: Color
    colorScore.textContent = `${morph.color.score}%`;
    rgbStdVal.textContent = `±${morph.color.mean_std}`;
    tonesVal.textContent = morph.color.zones_detected.join(', ') || 'Uniform';

    // Card D: Diameter
    diameterMm.textContent = `${morph.diameter.est_diam_mm} mm`;
    majorAxisVal.textContent = `${morph.diameter.major_axis_px} px`;
    minorAxisVal.textContent = `${morph.diameter.minor_axis_px} px`;

    // Card E: Uncertainty
    uncertScore.textContent = `${morph.uncertainty.mean_disagreement}%`;
    highUncertVal.textContent = `${morph.uncertainty.high_disagreement_pct}%`;
    ttaViewsVal.textContent = `${data.metadata.tta_views} views`;

    // Setup Export Handlers
    downloadMaskBtn.onclick = () => downloadDataUri(vis.mask, 'lesionlens_mask.png');
    downloadOverlayBtn.onclick = () => downloadDataUri(vis.overlay, 'lesionlens_overlay.png');
    exportAuditBtn.onclick = () => downloadJsonAudit(data);

    // Scroll smoothly into view
    analysisStage.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  // 7. Layer View Tab Switching
  viewTabs.querySelectorAll('.view-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      const viewKey = tab.dataset.view;
      setActiveView(viewKey);
    });
  });

  function setActiveView(viewKey) {
    state.activeView = viewKey;
    viewTabs.querySelectorAll('.view-tab').forEach(t => {
      t.classList.toggle('active', t.dataset.view === viewKey);
    });

    if (!state.analysisResult) return;
    const vis = state.analysisResult.visualizations;

    if (viewKey === 'split') {
      singleViewContainer.style.display = 'none';
      splitViewContainer.style.display = 'block';
    } else {
      splitViewContainer.style.display = 'none';
      singleViewContainer.style.display = 'flex';

      const viewCaptions = {
        overlay: 'Red Segmentation Overlay',
        contour: 'Cyan Boundary Contour Line',
        mask: 'Crisp Binary Segmentation Mask',
        probability_map: 'Continuous Probability Heatmap (Magma Colormap)',
        uncertainty_map: 'TTA Disagreement / Uncertainty Proxy (Inferno Colormap)',
        lesion_crop: 'Segmented Lesion Interior (Surrounding Skin Cropped)',
      };

      singleViewImg.src = vis[viewKey] || vis.original;
      singleLayerCaption.textContent = viewCaptions[viewKey] || viewKey;
    }
  }

  // 8. Interactive Split Comparison Slider Controls
  function setSplitSlider(percentage) {
    const clamped = Math.max(0, Math.min(100, percentage));
    sliderHandle.style.left = `${clamped}%`;
    splitLayerAfter.style.clipPath = `polygon(${clamped}% 0, 100% 0, 100% 100%, ${clamped}% 100%)`;
  }

  function handleSliderMove(clientX) {
    const rect = splitViewContainer.getBoundingClientRect();
    const x = clientX - rect.left;
    const pct = (x / rect.width) * 100;
    setSplitSlider(pct);
  }

  splitViewContainer.addEventListener('mousedown', (e) => {
    state.isDraggingSlider = true;
    handleSliderMove(e.clientX);
  });

  window.addEventListener('mousemove', (e) => {
    if (state.isDraggingSlider) {
      handleSliderMove(e.clientX);
    }
  });

  window.addEventListener('mouseup', () => {
    state.isDraggingSlider = false;
  });

  // Touch Support
  splitViewContainer.addEventListener('touchstart', (e) => {
    state.isDraggingSlider = true;
    if (e.touches[0]) handleSliderMove(e.touches[0].clientX);
  }, { passive: true });

  window.addEventListener('touchmove', (e) => {
    if (state.isDraggingSlider && e.touches[0]) {
      handleSliderMove(e.touches[0].clientX);
    }
  }, { passive: true });

  window.addEventListener('touchend', () => {
    state.isDraggingSlider = false;
  });

  // 9. Download Helpers
  function downloadDataUri(dataUri, filename) {
    const link = document.createElement('a');
    link.href = dataUri;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }

  function downloadJsonAudit(data) {
    const sanitized = {
      timestamp: new Date().toISOString(),
      metadata: data.metadata,
      morphology: data.morphology,
    };
    const blob = new Blob([JSON.stringify(sanitized, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `lesionlens_audit_${data.metadata.checkpoint_name}_${Date.now()}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }

});
