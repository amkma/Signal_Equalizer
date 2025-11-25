/* static/js/main.js */

const $  = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));
function firstSel(...sels){ for(const s of sels){ const el=$(s); if(el) return el; } return null; }
function setStatus(msg){ const el = firstSel("#statusbar","[data-role=status]"); if(el) el.textContent = msg; console.log(msg); }
function downloadBlob(data, filename, type="application/octet-stream"){
  const blob = new Blob([data], {type}); const url = URL.createObjectURL(blob);
  const a = document.createElement("a"); a.href=url; a.download=filename; document.body.appendChild(a); a.click();
  URL.revokeObjectURL(url); a.remove();
}

// ---------- DOM bindings ----------
const fileInput = firstSel("#file-hidden");
const dropZone = firstSel("#drop-zone");
const modeSelect = firstSel("#mode-select");
const btnAIPanel = firstSel("#btn-ai-panel");
const eqPanel = firstSel("#eq-sliders");

// Canvas Elements
const spectrumCanvas = firstSel("#fft-canvas");
const spectrumCtx = spectrumCanvas ? spectrumCanvas.getContext("2d") : null;
const spectrumLoader = firstSel("#spectrum-loader");
const inputCanvas = firstSel("#wave-in");
const outputCanvas = firstSel("#wave-out");
const inCtx = inputCanvas ? inputCanvas.getContext("2d") : null;
const outCtx = outputCanvas ? outputCanvas.getContext("2d") : null;

// New Toggles
const toggleAudiogram = firstSel("#toggle-audiogram");
const toggleBackend = firstSel("#toggle-backend");

// Spectrograms (Unified Canvas)
const specInCanvas = firstSel("#spec-in");
const specOutCanvas = firstSel("#spec-out");
const specInCtx = specInCanvas ? specInCanvas.getContext("2d") : null;
const specOutCtx = specOutCanvas ? specOutCanvas.getContext("2d") : null;

// Equalizer Buttons
const btnClearSubBand = firstSel("#btn-clear-subband");
const btnSaveScheme = firstSel("#btn-scheme-save");
const btnLoadScheme = firstSel("#btn-scheme-load");
const fileSchemeInput = firstSel("#file-scheme");

// AI / Players
const btnRunAI = firstSel("#btn-run-ai");
const audioIn = firstSel("#audio-in");
const audioOut = firstSel("#audio-out");
const btnPlayInput = firstSel("#play-input");
const btnPlayOutput = firstSel("#play-output");
const btnSyncReset = firstSel("#sync-reset");

// ---------- App State ----------
const state = {
  signalId:null, sr:0, duration:0, nSamples:0, fmax:0,
  spectrumMags: [],
  scale:"linear",
  fftBackend: "numpy",
  showSpectrograms:true,
  mode:"generic", subbands:[], customSliders:[],
  selecting:false, selStartX:0, selEndX:0,
  rawSpecIn: null, rawSpecOut: null,
  inputSamples: [], outputSamples: [],
  specInBitmap: null, specOutBitmap: null
};

const redPalette = [
  [0,0,0], [75,0,159], [104,0,251], [131,0,255],
  [155,18,157], [175,42,0], [191,59,0], [223,132,0], [255,252,0]
];

// ---------- API Helpers ----------
async function apiPost(url, data, isJson=true){
  const r = await fetch(url, {method:"POST", headers:isJson?{"Content-Type":"application/json"}:undefined, body:isJson?JSON.stringify(data):data});
  if(!r.ok) throw new Error(await r.text());
  const ct = r.headers.get("content-type")||"";
  return ct.includes("application/json") ? r.json() : r.arrayBuffer();
}
async function apiGet(url){
  const r = await fetch(url); if(!r.ok) throw new Error(await r.text());
  const ct = r.headers.get("content-type")||"";
  return ct.includes("application/json") ? r.json() : r.arrayBuffer();
}

// ---------- Global State Management ----------
function setGlobalState(enabled) {
    const disabled = !enabled;

    if(modeSelect) modeSelect.disabled = false;

    if(btnSaveScheme) btnSaveScheme.disabled = disabled;
    if(fileSchemeInput) fileSchemeInput.disabled = disabled;

    const loadLabel = document.querySelector(".file-btn");
    if(loadLabel) {
        if(disabled) loadLabel.classList.add("disabled");
        else loadLabel.classList.remove("disabled");
    }

    if(disabled && btnClearSubBand) btnClearSubBand.disabled = true;
    if(btnRunAI) btnRunAI.disabled = disabled;
    if(btnPlayInput) btnPlayInput.disabled = disabled;
    if(btnPlayOutput) btnPlayOutput.disabled = disabled;
    if(btnSyncReset) btnSyncReset.disabled = disabled;

    [toggleAudiogram, toggleBackend].forEach(input => {
        if(input) {
            input.disabled = disabled;
            const label = input.closest('.toggle-switch');
            if(label) {
                if(disabled) label.classList.add('disabled');
                else label.classList.remove('disabled');
            }
        }
    });
}

// ---------- File Upload Logic ----------
function bindUpload(){
  if(dropZone){
    dropZone.addEventListener("click", () => fileInput && fileInput.click());
    ["dragenter","dragover"].forEach(ev => dropZone.addEventListener(ev, e => { e.preventDefault(); dropZone.classList.add("drag"); }));
    ["dragleave","drop"].forEach(ev => dropZone.addEventListener(ev, e => { e.preventDefault(); dropZone.classList.remove("drag"); }));
    dropZone.addEventListener("drop", (e) => { const f = e.dataTransfer?.files?.[0]; if(f) doUploadFile(f); });
  }
  if(fileInput) fileInput.addEventListener("change", (e) => { const f = e.target.files?.[0]; if(f) doUploadFile(f); });
}

async function doUploadFile(file){
  try{
    setStatus(`Uploading: ${file.name} ...`);
    if(spectrumLoader) spectrumLoader.classList.remove("hidden");
    const fd = new FormData(); fd.append("signal", file);

    const res = await apiPost("/api/upload/", fd, false);
    const j = typeof res === "object" ? res : JSON.parse(new TextDecoder().decode(res));

    state.signalId = j.signal_id; state.sr = j.sr;
    state.duration = j.duration; state.nSamples = j.n;

    setStatus(`Loaded ${j.file_name} — sr=${j.sr}Hz, len=${j.duration.toFixed(2)}s`);

    setGlobalState(true);

    if(state.mode !== 'generic'){
       await renderCustomizedSliders();
       await applyEqualizer();
    }
    await refreshAll();
  }catch(err){
    console.error(err);
    setStatus(`Upload error: ${err.message}`);
    if(spectrumLoader) spectrumLoader.classList.add("hidden");
  }
}

// ---------- Drawing Logic ----------
function drawGridLogic(ctx, W, H, marginL, marginR, xLabels, yLabels, xTitle, yTitle, gridColor="#444", textColor="#aaa") {
    ctx.strokeStyle = gridColor; ctx.fillStyle = textColor; ctx.lineWidth = 1;
    ctx.font = "10px monospace"; ctx.textAlign = "center";

    const drawH = H - 20;
    const drawW = W - marginL - marginR;

    ctx.beginPath(); ctx.moveTo(0, drawH); ctx.lineTo(W, drawH); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(marginL, 0); ctx.lineTo(marginL, H); ctx.stroke();

    xLabels.forEach(lbl => {
        const x = marginL + (lbl.pos * drawW);
        ctx.beginPath(); ctx.moveTo(x, drawH); ctx.lineTo(x, drawH + 5); ctx.stroke();
        ctx.fillText(lbl.text, x, drawH + 15);
    });

    ctx.textAlign = "right";
    yLabels.forEach(lbl => {
        const y = drawH - (lbl.pos * drawH);
        ctx.beginPath(); ctx.moveTo(marginL - 5, y); ctx.lineTo(marginL, y); ctx.stroke();
        ctx.fillText(lbl.text, marginL - 8, y + 3);
    });

    if(xTitle) { ctx.textAlign = "right"; ctx.fillText(xTitle, W - 10, drawH + 15); }
    if(yTitle) { ctx.save(); ctx.translate(10, H/2); ctx.rotate(-Math.PI/2); ctx.textAlign="center"; ctx.fillText(yTitle, 0, 0); ctx.restore(); }
}

function drawSpectrum(mags, fmax, canvas, ctx){
  if(!canvas||!ctx||!Array.isArray(mags)) return;
  const W=canvas.width, H=canvas.height;
  ctx.clearRect(0,0,W,H); ctx.fillStyle="#000"; ctx.fillRect(0,0,W,H);

  const marginL=30, marginR=20, marginTop=30, marginBottom=20;
  const drawW=W-marginL-marginR;
  const drawH=H-marginTop-marginBottom;

  const xLabels = [];
  for(let i=0; i<=5; i++) {
      const freq = (fmax * i / 5);
      const text = freq >= 1000 ? (freq/1000).toFixed(1) + "k" : freq.toFixed(0);
      xLabels.push({pos: i/5, text: text});
  }

  ctx.strokeStyle = "#444"; ctx.fillStyle = "#aaa"; ctx.lineWidth = 1; ctx.font = "10px monospace"; ctx.textAlign = "center";
  ctx.beginPath(); ctx.moveTo(0, H - marginBottom); ctx.lineTo(W, H - marginBottom); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(marginL, 0); ctx.lineTo(marginL, H); ctx.stroke();

  xLabels.forEach(lbl => {
      const x = marginL + (lbl.pos * drawW);
      ctx.beginPath(); ctx.moveTo(x, H - marginBottom); ctx.lineTo(x, H - marginBottom + 5); ctx.stroke();
      ctx.fillText(lbl.text, x, H - marginBottom + 15);
  });

  ctx.textAlign = "right";
  let y1 = marginTop;
  ctx.beginPath(); ctx.moveTo(marginL - 5, y1); ctx.lineTo(marginL, y1); ctx.stroke();
  ctx.fillText("1", marginL - 8, y1 + 3);

  let y0 = marginTop + drawH;
  ctx.beginPath(); ctx.moveTo(marginL - 5, y0); ctx.lineTo(marginL, y0); ctx.stroke();
  ctx.fillText("0", marginL - 8, y0 + 3);

  ctx.strokeStyle = state.scale === "audiogram" ? "#fa7e1e" : "#d62976";
  ctx.lineWidth=2; ctx.beginPath();

  for(let i=0;i<mags.length;i++){
    const x = marginL + (i/(mags.length-1)) * drawW;
    const y = marginTop + drawH - (mags[i] * drawH);
    if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
  }
  ctx.stroke();

  if(state.mode === "generic" && state.subbands.length > 0){
      state.subbands.forEach(sb => {
          const x1 = marginL + (sb.fmin / state.fmax) * drawW;
          const x2 = marginL + (sb.fmax / state.fmax) * drawW;
          ctx.fillStyle = "rgba(214, 41, 118, 0.20)";
          ctx.fillRect(x1, 0, x2 - x1, H);
          ctx.strokeStyle = "rgba(255, 255, 255, 0.3)"; ctx.lineWidth = 1;
          ctx.strokeRect(x1, 0, x2 - x1, H);
      });
  }

  if(state.mode==="generic" && state.selecting){
    const x1=Math.min(state.selStartX,state.selEndX), x2=Math.max(state.selStartX,state.selEndX);
    ctx.fillStyle="rgba(255, 255, 255, 0.20)";
    ctx.fillRect(Math.max(marginL,x1), 0, x2 - Math.max(marginL,x1), H);
  }
}

function drawWavePreview(canvas, ctx, samples, playheadRatio = null){
  if(!canvas||!ctx||!Array.isArray(samples)) return;
  const W=canvas.width, H=canvas.height;
  ctx.clearRect(0,0,W,H); ctx.fillStyle="#000"; ctx.fillRect(0,0,W,H);

  const marginL=75, marginR=20, drawW=W-marginL-marginR, drawH=H-20, mid=drawH/2;
  const xLabels=[]; const duration=state.duration||0;
  for(let i=0; i<=5; i++) xLabels.push({pos:i/5, text:(duration*i/5).toFixed(1)+"s"});

  drawGridLogic(ctx, W, H, marginL, marginR, xLabels, [{pos:0,text:"-1"},{pos:0.5,text:"0"},{pos:1,text:"1"}], "Time", "Amp");

  ctx.strokeStyle="#a8a8a8"; ctx.lineWidth=1; ctx.beginPath();
  const step=Math.max(1,Math.ceil(samples.length/drawW));
  for(let i=0;i<samples.length;i+=step){
      const x=marginL+(i/(samples.length-1))*drawW;
      const y=mid-(samples[i]*mid);
      if(i===0)ctx.moveTo(x,y); else ctx.lineTo(x,y);
  }
  ctx.stroke();

  if(playheadRatio!==null && playheadRatio>=0 && playheadRatio<=1){
      const cX=marginL+playheadRatio*drawW;
      ctx.strokeStyle="#f00"; ctx.lineWidth=2; ctx.beginPath(); ctx.moveTo(cX,0); ctx.lineTo(cX,drawH); ctx.stroke();
  }
}

function drawSpectrogram(canvas, ctx, b64Data, isInput=true, playheadRatio=null) {
    if(!ctx) return;
    const W = canvas.width, H = canvas.height;
    const marginL = 75, marginR = 20;
    const drawW = W - marginL - marginR;
    const drawH = H - 20;

    ctx.clearRect(0,0,W,H); ctx.fillStyle="#000"; ctx.fillRect(0,0,W,H);

    const yLabels = [];
    for(let i=0; i<=4; i++){
        const norm = i/4;
        const freq = (state.fmax||10000) * (norm*norm);
        yLabels.push({pos:norm, text:(freq/1000).toFixed(1)+"k"});
    }
    const xLabels = []; const duration=state.duration||10;
    for(let i=0; i<=5; i++) xLabels.push({pos:i/5, text:(duration*i/5).toFixed(1)+"s"});

    drawGridLogic(ctx, W, H, marginL, marginR, xLabels, yLabels, "Time", "Freq");

    const bitmap = isInput ? state.specInBitmap : state.specOutBitmap;

    if(b64Data) {
        const img = new Image();
        img.onload = () => {
             const tmpCvs = document.createElement("canvas"); tmpCvs.width=W; tmpCvs.height=H;
             const tCtx = tmpCvs.getContext("2d");
             tCtx.drawImage(img, 0, 0, drawW, drawH);
             const imgD = tCtx.getImageData(0,0,drawW,drawH); const data=imgD.data;
             const interpolateColor = (t, arr) => {
                const i = arr.length - 1; const s = 1/i; const a = Math.floor(t/s); const n = (t - s*a)/s;
                const c1 = arr[Math.min(a, i)]; const c2 = arr[Math.min(a+1, i)];
                return [c1[0] + n*(c2[0]-c1[0]), c1[1] + n*(c2[1]-c1[1]), c1[2] + n*(c2[2]-c1[2])];
             };
             for(let i=0; i<data.length; i+=4) {
                 const val = data[i]/255; const rgb = interpolateColor(val, redPalette);
                 data[i]=rgb[0]; data[i+1]=rgb[1]; data[i+2]=rgb[2];
             }
             createImageBitmap(imgD).then(bmp => {
                 if(isInput) state.specInBitmap = bmp; else state.specOutBitmap = bmp;
                 drawSpectrogram(canvas, ctx, null, isInput, playheadRatio);
             });
        };
        img.src = `data:image/png;base64,${b64Data}`;
        return;
    }

    if(bitmap) {
        ctx.drawImage(bitmap, marginL, 0, drawW, drawH);
    }

    if(playheadRatio!==null && playheadRatio>=0 && playheadRatio<=1){
        const cX=marginL+playheadRatio*drawW;
        ctx.strokeStyle="#f00"; ctx.lineWidth=2; ctx.beginPath(); ctx.moveTo(cX,0); ctx.lineTo(cX,drawH); ctx.stroke();
    }
}

async function refreshSpectrograms() {
    if(!state.signalId) return;
    const specs = await apiGet(`/api/spectrograms/${state.signalId}/?scale_type=logarithmic&backend=${state.fftBackend}&t=${Date.now()}`);
    const jSpecs = typeof specs === "object" ? specs : JSON.parse(new TextDecoder().decode(specs));
    drawSpectrogram(specInCanvas, specInCtx, jSpecs.in_png, true);
    drawSpectrogram(specOutCanvas, specOutCtx, jSpecs.out_png, false);
}

function renderSpectrograms(playheadRatio=null) {
    // This function is kept for fallback but logic is handled in updateVisuals now
    drawSpectrogram(specInCanvas, specInCtx, null, true, playheadRatio);
    drawSpectrogram(specOutCanvas, specOutCtx, null, false, playheadRatio);
}

async function refreshOutputs(){
  if(!state.signalId) return;
  if(spectrumLoader) spectrumLoader.classList.remove("hidden");

  audioOut.src = `/api/audio/${state.signalId}/output.wav?t=${Date.now()}`;
  const spec = await apiGet(`/api/spectrum/${state.signalId}/?scale=${state.scale}&backend=${state.fftBackend}&t=${Date.now()}`);
  const jSpec = typeof spec  === "object" ? spec  : JSON.parse(new TextDecoder().decode(spec));
  state.fmax = jSpec.fmax; state.spectrumMags = jSpec.mags;
  drawSpectrum(jSpec.mags, jSpec.fmax, spectrumCanvas, spectrumCtx);

  const waves = await apiGet(`/api/wave_previews/${state.signalId}/?t=${Date.now()}`);
  const jWaves = typeof waves === "object" ? waves : JSON.parse(new TextDecoder().decode(waves));
  state.outputSamples = jWaves.output;
  drawWavePreview(outputCanvas, outCtx, jWaves.output, 0);

  await refreshSpectrograms();
  if(spectrumLoader) spectrumLoader.classList.add("hidden");
}

async function refreshAll(){
  if(!state.signalId) return;
  if(spectrumLoader) spectrumLoader.classList.remove("hidden");
  const ts = Date.now();
  audioIn.src = `/api/audio/${state.signalId}/input.wav?t=${ts}`;
  audioOut.src = `/api/audio/${state.signalId}/output.wav?t=${ts}`;

  const spec = await apiGet(`/api/spectrum/${state.signalId}/?scale=${state.scale}&backend=${state.fftBackend}&t=${ts}`);
  const jSpec = typeof spec  === "object" ? spec  : JSON.parse(new TextDecoder().decode(spec));
  state.fmax = jSpec.fmax; state.spectrumMags = jSpec.mags;
  drawSpectrum(jSpec.mags, jSpec.fmax, spectrumCanvas, spectrumCtx);

  const waves = await apiGet(`/api/wave_previews/${state.signalId}/?t=${ts}`);
  const jWaves = typeof waves === "object" ? waves : JSON.parse(new TextDecoder().decode(waves));
  state.inputSamples = jWaves.input; state.outputSamples = jWaves.output;
  drawWavePreview(inputCanvas, inCtx, jWaves.input, 0);
  drawWavePreview(outputCanvas, outCtx, jWaves.output, 0);

  renderEqSliders();
  await refreshSpectrograms();
  if(spectrumLoader) spectrumLoader.classList.add("hidden");
}

function bindSpectrumSelection(){
  if(!spectrumCanvas) return;
  const cvs=spectrumCanvas;
  const marginL = 30;

  cvs.addEventListener("mousedown",(e)=>{
    if(state.mode!=="generic" || !state.signalId) return;
    state.selecting=true;
    const r=cvs.getBoundingClientRect(); const scaleX = cvs.width / r.width;
    state.selStartX = (e.clientX - r.left) * scaleX; state.selEndX = state.selStartX;
    redrawSpectrum();
  });
  cvs.addEventListener("mousemove",(e)=>{
    if(!state.selecting) return;
    const r=cvs.getBoundingClientRect(); const scaleX = cvs.width / r.width;
    state.selEndX = (e.clientX - r.left) * scaleX; redrawSpectrum();
  });
  window.addEventListener("mouseup", async ()=>{
    if(!state.selecting) return; state.selecting=false; redrawSpectrum();
    const band=await promptBandFromSelection();
    if(band){ state.subbands.push(band); renderEqSliders(); redrawSpectrum(); await applyEqualizer(); }
  });

  function redrawSpectrum(){ if(state.spectrumMags) drawSpectrum(state.spectrumMags, state.fmax, spectrumCanvas, spectrumCtx); }
  function promptBandFromSelection(){
    const W = spectrumCanvas.width;
    const drawW = W - 30 - 20;
    const x1=Math.min(state.selStartX,state.selEndX), x2=Math.max(state.selStartX,state.selEndX);
    const freq1 = ((Math.max(0, x1 - 30) / drawW) * state.fmax).toFixed(1);
    const freq2 = ((Math.max(0, x2 - 30) / drawW) * state.fmax).toFixed(1);
    const resp = window.prompt(`Sub-band:\nMin Hz, Max Hz, Gain (0..2)\n`, `${freq1}, ${freq2}, 1.0`); if(!resp) return null;
    const p = resp.split(",").map(s=>+s.trim()); if(p.length<3||p.some(Number.isNaN)) return null;
    return {id:`sb${Date.now()}`, fmin:Math.min(p[0],p[1]), fmax:Math.max(p[0],p[1]), gain:Math.max(0,Math.min(2,p[2]))};
  }
}

function renderEqSliders(){
  if(state.mode === 'generic'){ renderGenericSubbands(); $('#generic-tools').style.display = 'flex'; }
  else { renderCustomizedSliders(); $('#generic-tools').style.display = 'none'; }
}

function renderGenericSubbands(){
  if(!eqPanel) return; eqPanel.innerHTML="";
  if(btnClearSubBand) btnClearSubBand.disabled = !(state.signalId && state.subbands.length > 0);
  state.subbands.forEach((b,idx)=>{
    const row=document.createElement("div"); row.className="sb-row";
    row.innerHTML = `<div class="sb-title">SubBand ${idx+1} [${b.fmin.toFixed(1)}–${b.fmax.toFixed(1)} Hz]</div><input type="range" min="0" max="2" step="0.01" value="${b.gain}" data-id="${b.id}"/><span class="sb-gain">${b.gain.toFixed(2)}x</span><button data-act="edit" data-id="${b.id}">Edit</button><button data-act="del" data-id="${b.id}" class="btn-danger">Delete</button>`;
    eqPanel.appendChild(row);
  });
  eqPanel.oninput = async (e)=>{ const r=e.target; if(r.tagName==="INPUT"){ const id=r.dataset.id; const sb=state.subbands.find(s=>s.id===id); if(sb){ sb.gain=+r.value; r.parentElement.querySelector(".sb-gain").textContent=`${sb.gain.toFixed(2)}x`; await applyEqualizerDebounced(); }}};
  eqPanel.onclick  = async (e)=>{ const b=e.target.closest("button"); if(!b) return; const id=b.dataset.id; const sb=state.subbands.find(s=>s.id===id); if(!sb) return;
    if(b.dataset.act==="del"){ state.subbands=state.subbands.filter(s=>s.id!==id); renderEqSliders(); if(state.spectrumMags) drawSpectrum(state.spectrumMags, state.fmax, spectrumCanvas, spectrumCtx); await applyEqualizer(); }
    else { /* edit logic */ }
  };
}

async function renderCustomizedSliders(){
    if(!eqPanel) return; eqPanel.innerHTML = "<p>Loading sliders...</p>";
    try {
        let modeName = 'generic';
        if(state.mode === 'music') modeName = 'musical instruments';
        else if(state.mode === 'animal') modeName = 'animal sounds';
        else if(state.mode === 'human') modeName = 'human voices';
        if(modeName === 'generic') { eqPanel.innerHTML = ""; return; }

        const resp = await apiGet(`/api/custom_conf/0/?mode=${modeName}`);
        state.customSliders = resp.sliders || [];
        eqPanel.innerHTML = "";
        if(state.customSliders.length === 0){ eqPanel.innerHTML = "<p>No sliders defined.</p>"; return; }
        state.customSliders.forEach((slider, idx) => {
            const row = document.createElement("div"); row.className = "sb-row"; slider.id = `custom${idx}`;
            row.innerHTML = `<div class="sb-title">${slider.name}</div><input type="range" min="0" max="2" step="0.01" value="${slider.gain}" data-id="${slider.id}"/><span class="sb-gain">${slider.gain.toFixed(2)}x</span>`;
            eqPanel.appendChild(row);
        });
        eqPanel.oninput = async (e) => { const r = e.target; if (r.tagName === "INPUT") { const id = r.dataset.id; const slider = state.customSliders.find(s => s.id === id); if (slider) { slider.gain = +r.value; r.parentElement.querySelector(".sb-gain").textContent = `${slider.gain.toFixed(2)}x`; await applyEqualizerDebounced(); }}};
    } catch(err){ console.error(err); }
}

let eqTimer=null;
async function applyEqualizerDebounced(){ if(eqTimer) clearTimeout(eqTimer); eqTimer=setTimeout(applyEqualizer,120); }
async function applyEqualizer(){
  if(!state.signalId) return;
  const payload = state.mode==="generic" ? {mode:"generic", subbands:state.subbands} : {mode:state.mode, sliders:state.customSliders};
  try{ if(spectrumLoader) spectrumLoader.classList.remove("hidden"); await apiPost(`/api/equalize/${state.signalId}/`, payload); await refreshOutputs(); }
  catch(err){ console.error(err); setStatus(`Equalize error: ${err.message}`); if(spectrumLoader) spectrumLoader.classList.add("hidden"); }
}

function bindToggles(){
  if(btnClearSubBand) btnClearSubBand.addEventListener("click", async ()=>{
      if(state.mode !== 'generic' || !state.signalId) return;
      state.subbands = []; renderEqSliders(); if(state.spectrumMags) drawSpectrum(state.spectrumMags, state.fmax, spectrumCanvas, spectrumCtx); await applyEqualizer();
  });
  if(modeSelect) modeSelect.addEventListener("change", async e => { state.mode = e.target.value; state.subbands=[]; state.customSliders=[]; renderEqSliders(); if(state.signalId) await applyEqualizer(); });
  if(toggleAudiogram) toggleAudiogram.addEventListener("change", e => { state.scale = e.target.checked ? "audiogram" : "linear"; if(state.signalId) refreshOutputs(); else if(state.spectrumMags) drawSpectrum(state.spectrumMags, state.fmax, spectrumCanvas, spectrumCtx); });
  if(toggleBackend) toggleBackend.addEventListener("change", e => { state.fftBackend = e.target.checked ? "cpp" : "numpy"; if(state.signalId) refreshOutputs(); });
}

function bindSaveLoad(){
  if(btnSaveScheme) btnSaveScheme.addEventListener("click", async ()=>{ if(!state.signalId) return alert("Upload a signal first."); const scheme = state.mode==="generic" ? {mode:"generic", subbands:state.subbands} : {mode:state.mode, sliders:state.customSliders}; const buf = await apiPost(`/api/save_scheme/${state.signalId}/`, scheme); const j = typeof buf==="object" ? buf : JSON.parse(new TextDecoder().decode(buf)); downloadBlob(new TextEncoder().encode(JSON.stringify(j.data,null,2)), j.filename, "application/json"); });
  if(fileInput) fileInput.addEventListener("change", (e) => { const f = e.target.files?.[0]; if(f) doUploadFile(f); });
  const fileSchemeInput = firstSel("#file-scheme");
  if(fileSchemeInput) fileSchemeInput.addEventListener("change", async (e)=>{ const f = e.target.files?.[0]; if(!f) return; const data = JSON.parse(await f.text()); await apiPost(`/api/load_scheme/${state.signalId}/`, data); state.mode=data.mode||"generic"; state.subbands=data.subbands||[]; state.customSliders=data.sliders||[]; if(modeSelect) modeSelect.value=state.mode; renderEqSliders(); await applyEqualizer(); });
}

function bindPlayback(){
  if(!audioIn || !audioOut) return;

  function updateVisuals() {
      requestAnimationFrame(updateVisuals);

      // --- Input Visuals (Independent) ---
      let inRatio = 0;
      if (state.duration > 0) {
          inRatio = audioIn.currentTime / state.duration;
      }

      if(state.inputSamples.length > 0)
          drawWavePreview(inputCanvas, inCtx, state.inputSamples, inRatio);

      if(state.specInBitmap)
          drawSpectrogram(specInCanvas, specInCtx, null, true, inRatio);

      // --- Output Visuals (Independent) ---
      let outRatio = 0;
      if (state.duration > 0) {
          outRatio = audioOut.currentTime / state.duration;
      }

      if(state.outputSamples.length > 0)
          drawWavePreview(outputCanvas, outCtx, state.outputSamples, outRatio);

      if(state.specOutBitmap)
          drawSpectrogram(specOutCanvas, specOutCtx, null, false, outRatio);
  }

  requestAnimationFrame(updateVisuals);

  btnPlayInput.addEventListener("click", () => {
    if(audioIn.paused){ audioIn.play(); btnPlayInput.textContent = "Pause Input"; } else { audioIn.pause(); btnPlayInput.textContent = "Play Input"; }
  });
  btnPlayOutput.addEventListener("click", () => {
    if(audioOut.paused){ audioOut.play(); btnPlayOutput.textContent = "Pause Output"; } else { audioOut.pause(); btnPlayOutput.textContent = "Play Output"; }
  });
  btnSyncReset.addEventListener("click", () => {
    audioIn.pause(); audioOut.pause(); audioIn.currentTime = 0; audioOut.currentTime = 0;
    btnPlayInput.textContent = "Play Input"; btnPlayOutput.textContent = "Play Output";
    // Visuals reset naturally via updateVisuals loop
  });
}

function init(){
    bindUpload();
    bindSpectrumSelection();
    bindPlayback();
    bindToggles();
    bindSaveLoad();
    setGlobalState(false);
    setStatus("Ready.");
}
document.addEventListener("DOMContentLoaded", init);