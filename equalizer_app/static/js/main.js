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
const btnOpen = firstSel("#btn-open");
const fileInput = firstSel("#file-hidden");
const dropZone = firstSel("#drop-zone");
const btnSaveSettings = firstSel("#btn-save-settings");
const btnLoadSettings = firstSel("#btn-load-settings");
const modeSelect = firstSel("#mode-select");
const btnAIPanel = firstSel("#btn-ai-panel");
const eqPanel = firstSel("#eq-sliders");
const spectrumCanvas = firstSel("#fft-canvas");
const spectrumCtx = spectrumCanvas ? spectrumCanvas.getContext("2d") : null;
const spectrumLoader = firstSel("#spectrum-loader");
const inputCanvas = firstSel("#wave-in");
const outputCanvas = firstSel("#wave-out");
const inCtx = inputCanvas ? inputCanvas.getContext("2d") : null;
const outCtx = outputCanvas ? outputCanvas.getContext("2d") : null;
// REMOVED: btnAddSubBand
const btnClearSubBand = firstSel("#btn-clear-subband");
const btnSaveScheme = firstSel("#btn-scheme-save");
const btnLoadScheme = firstSel("#btn-scheme-load");
const audioIn = firstSel("#audio-in");
const audioOut = firstSel("#audio-out");
const btnPlayInput = firstSel("#play-input");
const btnPlayOutput = firstSel("#play-output");
const btnSyncReset = firstSel("#sync-reset");

// New Spectrogram Canvases
const specInCanvas = firstSel("#spec-in-canvas");
const specInAxis = firstSel("#spec-in-axis");
const specOutCanvas = firstSel("#spec-out-canvas");
const specOutAxis = firstSel("#spec-out-axis");

const specInCtx = specInCanvas ? specInCanvas.getContext("2d") : null;
const specInAxisCtx = specInAxis ? specInAxis.getContext("2d") : null;
const specOutCtx = specOutCanvas ? specOutCanvas.getContext("2d") : null;
const specOutAxisCtx = specOutAxis ? specOutAxis.getContext("2d") : null;

// ---------- app state ----------
const state = {
  signalId:null, sr:0, duration:0, nSamples:0, fmax:0,
  spectrumMags: [],
  scale:"audiogram", showSpectrograms:true,
  mode:"generic", subbands:[], customSliders:[],
  selecting:false, selStartX:0, selEndX:0,
  rawSpecIn: null, rawSpecOut: null,

  // Animation & Caching
  inputSamples: [],
  outputSamples: [],
  specInBitmap: null,
  specOutBitmap: null
};

// High-Contrast Red Palette for Dark Theme
const redPalette = [
  [0,0,0],       // Black
  [75,0,159],    // Deep Purple
  [104,0,251],
  [131,0,255],
  [155,18,157],  // Purple-Red
  [175,42,0],    // Red
  [191,59,0],
  [223,132,0],   // Orange
  [255,252,0]    // Yellow (Peaks)
];

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

function bindUpload(){
  if(btnOpen) btnOpen.addEventListener("click", () => fileInput && fileInput.click());
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
    state.signalId = j.signal_id; state.sr = j.sr; state.duration = j.duration; state.nSamples = j.n;
    setStatus(`Loaded ${j.file_name} — sr=${j.sr}Hz, len=${j.duration.toFixed(2)}s`);
    await refreshAll();
  }catch(err){
    console.error(err);
    setStatus(`Upload error: ${err.message}`);
    if(spectrumLoader) spectrumLoader.classList.add("hidden");
  }
}

// ---------- Drawing Logic ----------
function clearCanvas(ctx, cvs){
    if(!ctx||!cvs) return;
    ctx.clearRect(0,0,cvs.width,cvs.height);
}

function drawGrid(ctx, W, H, xLabels, yLabels, xTitle, yTitle) {
    ctx.strokeStyle = "#444";
    ctx.fillStyle = "#aaa";
    ctx.lineWidth = 1;
    ctx.font = "10px monospace";
    ctx.textAlign = "center";
    ctx.beginPath(); ctx.moveTo(0, H - 20); ctx.lineTo(W, H - 20); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(30, 0); ctx.lineTo(30, H); ctx.stroke();
    xLabels.forEach(lbl => {
        const x = 30 + (lbl.pos * (W - 30));
        ctx.beginPath(); ctx.moveTo(x, H - 20); ctx.lineTo(x, H - 15); ctx.stroke();
        ctx.fillText(lbl.text, x, H - 5);
    });
    ctx.textAlign = "right";
    yLabels.forEach(lbl => {
        const y = (H - 20) - (lbl.pos * (H - 20));
        ctx.beginPath(); ctx.moveTo(25, y); ctx.lineTo(30, y); ctx.stroke();
        ctx.fillText(lbl.text, 25, y + 3);
    });
    if(xTitle) { ctx.textAlign = "right"; ctx.fillText(xTitle, W - 10, H - 5); }
}

function drawSpectrum(mags, fmax, canvas, ctx, scale="linear"){
  if(!canvas||!ctx||!Array.isArray(mags)) return;
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.fillStyle="#000"; ctx.fillRect(0,0,canvas.width,canvas.height);
  const W=canvas.width, H=canvas.height;
  const xLabels = [];
  for(let i=0; i<=5; i++) {
      const freq = (fmax * i / 5);
      const text = freq >= 1000 ? (freq/1000).toFixed(1) + "k" : freq.toFixed(0);
      xLabels.push({pos: i/5, text: text});
  }
  drawGrid(ctx, W, H, xLabels, [{pos:0,text:"0"},{pos:1,text:"1"}], "Hz", "Mag");
  ctx.strokeStyle="#d62976"; ctx.lineWidth=2; ctx.beginPath();
  const drawW = W - 30; const drawH = H - 20;
  for(let i=0;i<mags.length;i++){
    const x = 30 + (i/(mags.length-1)) * drawW;
    const y = drawH - (mags[i] * drawH);
    if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
  }
  ctx.stroke();
  if(state.mode === "generic" && state.subbands.length > 0){
      state.subbands.forEach(sb => {
          const x1 = 30 + (sb.fmin / state.fmax) * drawW;
          const x2 = 30 + (sb.fmax / state.fmax) * drawW;
          ctx.fillStyle = "rgba(214, 41, 118, 0.50)";
          ctx.fillRect(x1, 0, x2 - x1, drawH);
          ctx.strokeStyle = "rgba(255, 255, 255, 0.3)";
          ctx.lineWidth = 1;
          ctx.strokeRect(x1, 0, x2 - x1, drawH);
      });
  }
  if(state.mode==="generic" && state.selecting){
    const x1=Math.min(state.selStartX,state.selEndX), x2=Math.max(state.selStartX,state.selEndX);
    ctx.fillStyle="rgba(255, 255, 255, 0.2)";
    ctx.fillRect(Math.max(30,x1), 0, x2 - Math.max(30,x1), drawH);
  }
}

// Draw Wave Function
function drawWavePreview(canvas, ctx, samples, playheadRatio = null){
  if(!canvas||!ctx||!Array.isArray(samples)) return;

  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.fillStyle="#000";
  ctx.fillRect(0,0,canvas.width,canvas.height);

  const W = canvas.width;
  const H = canvas.height;

  // Margins matching Spectrogram
  const marginL = 75;
  const marginR = 20;
  const drawW = W - marginL - marginR;
  const drawH = H - 20;
  const mid = drawH / 2;

  const xLabels = [];
  const duration = state.duration || 0;
  for(let i=0; i<=5; i++) {
      xLabels.push({
          pos: i/5,
          text: (duration * i / 5).toFixed(1) + "s"
      });
  }

  ctx.strokeStyle = "#444";
  ctx.fillStyle = "#aaa";
  ctx.lineWidth = 1;
  ctx.font = "10px monospace";
  ctx.textAlign = "center";

  // X-Axis Line
  ctx.beginPath(); ctx.moveTo(0, drawH); ctx.lineTo(W, drawH); ctx.stroke();

  // Y-Axis Line
  ctx.beginPath(); ctx.moveTo(marginL, 0); ctx.lineTo(marginL, H); ctx.stroke();

  // Draw X Labels
  xLabels.forEach(lbl => {
      const x = marginL + (lbl.pos * drawW);
      ctx.beginPath(); ctx.moveTo(x, drawH); ctx.lineTo(x, drawH + 5); ctx.stroke();
      ctx.fillText(lbl.text, x, drawH + 15);
  });

  // Draw Y Labels
  ctx.textAlign = "right";
  const yLabels = [{pos:0,t:"-1"}, {pos:0.5,t:"0"}, {pos:1,t:"1"}];
  yLabels.forEach(lbl => {
      const y = drawH - (lbl.pos * drawH);
      ctx.beginPath(); ctx.moveTo(marginL - 5, y); ctx.lineTo(marginL, y); ctx.stroke();
      ctx.fillText(lbl.t, marginL - 8, y + 3);
  });

  ctx.fillText("Time", W - 10, drawH + 15);
  ctx.fillText("Amp", marginL - 8, 10);

  // Waveform
  ctx.strokeStyle = "#a8a8a8";
  ctx.lineWidth = 1;
  ctx.beginPath();

  const step = Math.max(1, Math.ceil(samples.length / drawW));

  for(let i=0; i < samples.length; i += step){
      const x = marginL + (i / (samples.length - 1)) * drawW;
      const y = mid - (samples[i] * mid);
      if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
  }
  ctx.stroke();

  // Playhead Pointer
  if (playheadRatio !== null && playheadRatio >= 0 && playheadRatio <= 1) {
      const cursorX = marginL + (playheadRatio * drawW);
      ctx.strokeStyle = "#ff0000";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(cursorX, 0);
      ctx.lineTo(cursorX, drawH);
      ctx.stroke();

      ctx.fillStyle = "#ff0000";
      ctx.beginPath();
      ctx.arc(cursorX, 0, 4, 0, Math.PI * 2);
      ctx.fill();
  }
}

// --- Spectrogram Drawing (Dark Mode + Red Theme) ---
function drawSpectrogramVisuals(canvas, axisCanvas, b64Data, isInput=true) {
    const ctx = canvas.getContext("2d");
    const axisCtx = axisCanvas.getContext("2d");
    const W = canvas.width, H = canvas.height;
    const AxisW = axisCanvas.width, AxisH = axisCanvas.height;

    const img = new Image();
    img.onload = () => {
        ctx.clearRect(0,0,W,H);
        ctx.drawImage(img, 0, 0, W, H);

        const imageData = ctx.getImageData(0, 0, W, H);
        const data = imageData.data;

        const interpolateColor = (t, arr) => {
            const i = arr.length - 1;
            const s = 1/i;
            const a = Math.floor(t/s);
            const n = (t - s*a)/s;
            const c1 = arr[Math.min(a, i)];
            const c2 = arr[Math.min(a+1, i)];
            return [
                c1[0] + n*(c2[0]-c1[0]),
                c1[1] + n*(c2[1]-c1[1]),
                c1[2] + n*(c2[2]-c1[2])
            ];
        };

        for(let i=0; i<data.length; i+=4) {
            const val = data[i];
            const norm = val / 255;
            const rgb = interpolateColor(norm, redPalette);
            data[i] = rgb[0];
            data[i+1] = rgb[1];
            data[i+2] = rgb[2];
        }

        createImageBitmap(imageData).then(bitmap => {
            if(isInput) state.specInBitmap = bitmap;
            else state.specOutBitmap = bitmap;
            ctx.putImageData(imageData, 0, 0);
        });
    };
    img.src = `data:image/png;base64,${b64Data}`;

    // Axes
    axisCtx.clearRect(0,0,AxisW,AxisH);
    const pLeft = 75, pTop = 20;
    const graphW = 540, graphH = 250;

    axisCtx.font = "12px 'Fira Code', monospace";
    axisCtx.textAlign = "right";
    const gridColor = "#363636";
    const labelColor = "#a8a8a8";
    const maxFreq = state.fmax || 10000;

    const yTicks = 11;
    for(let i=0; i<yTicks; i++) {
        const norm = i / (yTicks - 1);
        let freq = 0;
        if(i===0) freq=0;
        else freq = maxFreq * Math.pow(norm, 4);
        const yPos = pTop + graphH - (norm * graphH);

        axisCtx.strokeStyle = gridColor;
        axisCtx.lineWidth = 1;
        axisCtx.beginPath();
        axisCtx.moveTo(pLeft, yPos);
        axisCtx.lineTo(pLeft + graphW, yPos);
        axisCtx.stroke();

        axisCtx.fillStyle = labelColor;
        axisCtx.fillText((freq/1000).toFixed(1) + "k", pLeft - 10, yPos + 4);
    }

    const duration = state.duration || 10;
    const xTicks = 11;
    axisCtx.textAlign = "center";
    for(let i=0; i<xTicks; i++) {
        const norm = i / (xTicks - 1);
        const time = duration * norm;
        const xPos = pLeft + (norm * graphW);

        axisCtx.strokeStyle = gridColor;
        axisCtx.beginPath();
        axisCtx.moveTo(xPos, pTop);
        axisCtx.lineTo(xPos, pTop + graphH);
        axisCtx.stroke();

        axisCtx.fillStyle = labelColor;
        axisCtx.fillText(time.toFixed(1) + "s", xPos, pTop + graphH + 20);
    }

    axisCtx.strokeStyle = "#555";
    axisCtx.lineWidth = 1;
    axisCtx.strokeRect(pLeft, pTop, graphW, graphH);
}

function drawSpecWithCursor(ctx, bitmap, width, height, playheadRatio) {
    if(!bitmap) return;
    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(bitmap, 0, 0);

    if (playheadRatio !== null && playheadRatio >= 0 && playheadRatio <= 1) {
        const cursorX = playheadRatio * width;
        ctx.strokeStyle = "#ff0000";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(cursorX, 0);
        ctx.lineTo(cursorX, height);
        ctx.stroke();
    }
}

async function refreshSpectrograms() {
    if(!state.signalId) return;
    const specs = await apiGet(`/api/spectrograms/${state.signalId}/?scale_type=logarithmic`);
    const jSpecs = typeof specs === "object" ? specs : JSON.parse(new TextDecoder().decode(specs));
    state.rawSpecIn = jSpecs.in_png;
    state.rawSpecOut = jSpecs.out_png;
    renderSpectrograms();
}

function renderSpectrograms() {
    if(state.rawSpecIn) drawSpectrogramVisuals(specInCanvas, specInAxis, state.rawSpecIn, true);
    if(state.rawSpecOut) drawSpectrogramVisuals(specOutCanvas, specOutAxis, state.rawSpecOut, false);
}

// ---------- Main Refresh ----------
async function refreshAll(){
  if(!state.signalId) return;
  if(spectrumLoader) spectrumLoader.classList.remove("hidden");

  audioIn.src = `/api/audio/${state.signalId}/input.wav`;
  audioOut.src = `/api/audio/${state.signalId}/output.wav`;

  const meta = await apiGet(`/api/summary/${state.signalId}/`);
  if(typeof meta === "object"){
    const sb = $("#sb-file"), fs=$("#sb-fs"), ln=$("#sb-len");
    if(sb) sb.textContent = meta.file_name || "—";
    if(fs) fs.textContent = meta.sr || "—";
    if(ln) ln.textContent = (meta.duration ?? 0).toFixed(2);
  }

  const spec = await apiGet(`/api/spectrum/${state.signalId}/?scale=${state.scale}`);
  const jSpec = typeof spec  === "object" ? spec  : JSON.parse(new TextDecoder().decode(spec));

  state.fmax = jSpec.fmax;
  state.spectrumMags = jSpec.mags;
  drawSpectrum(jSpec.mags, jSpec.fmax, spectrumCanvas, spectrumCtx, state.scale);

  const waves = await apiGet(`/api/wave_previews/${state.signalId}/`);
  const jWaves = typeof waves === "object" ? waves : JSON.parse(new TextDecoder().decode(waves));

  state.inputSamples = jWaves.input;
  state.outputSamples = jWaves.output;

  drawWavePreview(inputCanvas, inCtx, jWaves.input, 0);
  drawWavePreview(outputCanvas, outCtx, jWaves.output, 0);

  renderEqSliders();
  await refreshSpectrograms();

  if(spectrumLoader) spectrumLoader.classList.add("hidden");
}

// ---------- Standard Init ----------
function bindSpectrumSelection(){
  if(!spectrumCanvas) return; const cvs=spectrumCanvas;
  cvs.addEventListener("mousedown",(e)=>{ if(state.mode!=="generic") return; state.selecting=true; const r=cvs.getBoundingClientRect(); state.selStartX=e.clientX-r.left; state.selEndX=state.selStartX; redrawSpectrum(); });
  cvs.addEventListener("mousemove",(e)=>{ if(!state.selecting) return; const r=cvs.getBoundingClientRect(); state.selEndX=e.clientX-r.left; redrawSpectrum(); });
  window.addEventListener("mouseup", async ()=>{
    if(!state.selecting) return; state.selecting=false; redrawSpectrum();
    const band=await promptBandFromSelection();
    if(band){ state.subbands.push(band); renderEqSliders(); await applyEqualizer(); }
  });
}
function redrawSpectrum(){ if(state.spectrumMags) drawSpectrum(state.spectrumMags, state.fmax, spectrumCanvas, spectrumCtx, state.scale); }
function pxToFreq(x,W,fmax){ const drawW = W - 30; const relX = Math.max(0, x - 30); return (Math.min(1,Math.max(0,relX/drawW)))*fmax; }
async function promptBandFromSelection(){
  const W = spectrumCanvas.width; const x1=Math.min(state.selStartX,state.selEndX), x2=Math.max(state.selStartX,state.selEndX);
  let fmin=+pxToFreq(x1,W,state.fmax).toFixed(1), fmax=+pxToFreq(x2,W,state.fmax).toFixed(1);
  const resp = window.prompt(`Sub-band:\nMin Hz, Max Hz, Gain (0..2)\n`, `${fmin}, ${fmax}, 1.0`); if(!resp) return null;
  const p = resp.split(",").map(s=>+s.trim()); if(p.length<3||p.some(Number.isNaN)) return null;
  return {id:`sb${Date.now()}`, fmin:Math.min(p[0],p[1]), fmax:Math.max(p[0],p[1]), gain:Math.max(0,Math.min(2,p[2]))};
}
function renderEqSliders(){
  if(state.mode === 'generic'){ renderGenericSubbands(); $('#generic-tools').style.display = 'flex'; }
  else { renderCustomizedSliders(); $('#generic-tools').style.display = 'none'; }
}

function renderGenericSubbands(){
  if(!eqPanel) return;
  eqPanel.innerHTML="";

  // UPDATED: Logic to disable/enable Clear button
  const btnClear = document.querySelector("#btn-clear-subband");
  if(btnClear) {
      if(state.subbands.length > 0) btnClear.removeAttribute("disabled");
      else btnClear.setAttribute("disabled", "true");
  }

  state.subbands.forEach((b,idx)=>{
    const row=document.createElement("div"); row.className="sb-row";
    row.innerHTML = `<div class="sb-title">SubBand ${idx+1} [${b.fmin.toFixed(1)}–${b.fmax.toFixed(1)} Hz]</div><input type="range" min="0" max="2" step="0.01" value="${b.gain}" data-id="${b.id}"/><span class="sb-gain">${b.gain.toFixed(2)}x</span><button data-act="edit" data-id="${b.id}">Edit</button><button data-act="del" data-id="${b.id}" class="btn-danger">Delete</button>`;
    eqPanel.appendChild(row);
  });
  eqPanel.oninput = async (e)=>{ const r=e.target; if(r.tagName==="INPUT"){ const id=r.dataset.id; const sb=state.subbands.find(s=>s.id===id); if(sb){ sb.gain=+r.value; r.parentElement.querySelector(".sb-gain").textContent=`${sb.gain.toFixed(2)}x`; await applyEqualizerDebounced(); }}};
  eqPanel.onclick  = async (e)=>{ const b=e.target.closest("button"); if(!b) return; const id=b.dataset.id; const sb=state.subbands.find(s=>s.id===id); if(!sb) return;
    if(b.dataset.act==="del"){ state.subbands=state.subbands.filter(s=>s.id!==id); renderEqSliders(); await applyEqualizer(); }
    else { const resp=window.prompt(`Edit [min,max,gain]`, `${sb.fmin}, ${sb.fmax}, ${sb.gain}`); if(!resp) return; const p=resp.split(",").map(s=>+s.trim()); if(p.length<3) return; sb.fmin=Math.min(p[0],p[1]); sb.fmax=Math.max(p[0],p[1]); sb.gain=Math.max(0,Math.min(2,p[2])); renderEqSliders(); await applyEqualizer(); }
  };
}

async function renderCustomizedSliders(){
  if(!eqPanel || !state.signalId) return; eqPanel.innerHTML = "<p>Loading sliders...</p>";
  try {
    const modeName = state.mode === 'music' ? 'musical instruments' : (state.mode === 'animals' ? 'animal sounds' : 'human voices');
    const resp = await apiGet(`/api/custom_conf/${state.signalId}/?mode=${modeName}`);
    state.customSliders = resp.sliders || [];
    eqPanel.innerHTML = "";
    if(state.customSliders.length === 0){ eqPanel.innerHTML = "<p>No sliders defined for this mode.</p>"; return; }
    state.customSliders.forEach((slider, idx) => {
      const row = document.createElement("div"); row.className = "sb-row"; slider.id = `custom${idx}`;
      row.innerHTML = `<div class="sb-title">${slider.name}</div><input type="range" min="0" max="2" step="0.01" value="${slider.gain}" data-id="${slider.id}"/><span class="sb-gain">${slider.gain.toFixed(2)}x</span>`;
      eqPanel.appendChild(row);
    });
    eqPanel.oninput = async (e) => { const r = e.target; if (r.tagName === "INPUT") { const id = r.dataset.id; const slider = state.customSliders.find(s => s.id === id); if (slider) { slider.gain = +r.value; r.parentElement.querySelector(".sb-gain").textContent = `${slider.gain.toFixed(2)}x`; await applyEqualizerDebounced(); }}};
  } catch(err) { console.error(err); eqPanel.innerHTML = `<p style="color:var(--danger);">Error loading sliders.</p>`; }
}
let eqTimer=null;
async function applyEqualizerDebounced(){ if(eqTimer) clearTimeout(eqTimer); eqTimer=setTimeout(applyEqualizer,120); }
async function applyEqualizer(){
  if(!state.signalId) return;
  const payload = state.mode==="generic" ? {mode:"generic", subbands:state.subbands} : {mode:state.mode, sliders:state.customSliders};
  try{ if(spectrumLoader) spectrumLoader.classList.remove("hidden"); await apiPost(`/api/equalize/${state.signalId}/`, payload); await refreshOutputs(); }
  catch(err){ console.error(err); setStatus(`Equalize error: ${err.message}`); if(spectrumLoader) spectrumLoader.classList.add("hidden"); }
}
async function refreshOutputs(){
  if(!state.signalId) return;
  if(spectrumLoader) spectrumLoader.classList.remove("hidden");
  audioOut.src = `/api/audio/${state.signalId}/output.wav?t=${Date.now()}`;
  const spec = await apiGet(`/api/spectrum/${state.signalId}/?scale=${state.scale}`);
  const jSpec = typeof spec  === "object" ? spec  : JSON.parse(new TextDecoder().decode(spec));
  state.fmax = jSpec.fmax; state.spectrumMags = jSpec.mags;
  drawSpectrum(jSpec.mags, jSpec.fmax, spectrumCanvas, spectrumCtx, state.scale);
  const waves = await apiGet(`/api/wave_previews/${state.signalId}/`);
  const jWaves = typeof waves === "object" ? waves : JSON.parse(new TextDecoder().decode(waves));

  state.outputSamples = jWaves.output;
  drawWavePreview(outputCanvas, outCtx, jWaves.output, 0);

  await refreshSpectrograms();
  if(spectrumLoader) spectrumLoader.classList.add("hidden");
}

function bindToggles(){
  if(btnAddSubBand) btnAddSubBand.addEventListener("click", ()=> { if(!state.signalId) return alert("Upload a signal first."); alert("Select an interval on the spectrum by dragging with the mouse."); });
  if(btnClearSubBand) btnClearSubBand.addEventListener("click", async ()=>{ if(state.mode !== 'generic' || !state.signalId) return; state.subbands = []; renderEqSliders(); await applyEqualizer(); });
  if(modeSelect) modeSelect.addEventListener("change", async e => { if(!state.signalId) { e.target.value = state.mode; return; } state.mode = e.target.value; state.subbands=[]; state.customSliders=[]; renderEqSliders(); await applyEqualizer(); });
}
function bindSaveLoad(){
  if(btnSaveSettings) btnSaveSettings.addEventListener("click", async ()=>{ if(!state.signalId) return alert("Upload a signal first."); const full = { scale:state.scale, showSpectrograms:state.showSpectrograms, ...(state.mode==="generic"?{mode:"generic",subbands:state.subbands}:{mode:state.mode,sliders:state.customSliders}) }; const buf = await apiPost(`/api/save_settings/${state.signalId}/`, full); const j = typeof buf==="object" ? buf : JSON.parse(new TextDecoder().decode(buf)); downloadBlob(new TextEncoder().encode(JSON.stringify(j.data,null,2)), j.filename, "application/json"); });
  const fileSettingsInput = firstSel("#file-settings");
  if(fileSettingsInput) fileSettingsInput.addEventListener("change", async (e)=>{ const f = e.target.files?.[0]; if(!f) return; const data = JSON.parse(await f.text()); await apiPost(`/api/load_settings/${state.signalId}/`, data); state.scale=data.scale||"linear"; state.showSpectrograms=!!data.showSpectrograms; state.mode=data.mode||"generic"; state.subbands=data.subbands||[]; state.customSliders=data.sliders||[]; if(modeSelect) modeSelect.value=state.mode; await refreshAll(); });
  if(btnSaveScheme) btnSaveScheme.addEventListener("click", async ()=>{ if(!state.signalId) return alert("Upload a signal first."); const scheme = state.mode==="generic" ? {mode:"generic", subbands:state.subbands} : {mode:state.mode, sliders:state.customSliders}; const buf = await apiPost(`/api/save_scheme/${state.signalId}/`, scheme); const j = typeof buf==="object" ? buf : JSON.parse(new TextDecoder().decode(buf)); downloadBlob(new TextEncoder().encode(JSON.stringify(j.data,null,2)), j.filename, "application/json"); });
  const fileSchemeInput = firstSel("#file-scheme");
  if(fileSchemeInput) fileSchemeInput.addEventListener("change", async (e)=>{ const f = e.target.files?.[0]; if(!f) return; const data = JSON.parse(await f.text()); await apiPost(`/api/load_scheme/${state.signalId}/`, data); state.mode=data.mode||"generic"; state.subbands=data.subbands||[]; state.customSliders=data.sliders||[]; if(modeSelect) modeSelect.value=state.mode; renderEqSliders(); await applyEqualizer(); });
}

function bindPlayback(){
  if(!audioIn || !audioOut) return;

  function updateVisuals() {
      requestAnimationFrame(updateVisuals);

      if (!audioIn.paused) {
          const ratio = audioIn.currentTime / state.duration;
          if(state.inputSamples.length > 0)
              drawWavePreview(inputCanvas, inCtx, state.inputSamples, ratio);
          if(state.specInBitmap)
              drawSpecWithCursor(specInCtx, state.specInBitmap, specInCanvas.width, specInCanvas.height, ratio);
      }

      if (!audioOut.paused) {
          const ratio = audioOut.currentTime / state.duration;
          if(state.outputSamples.length > 0)
              drawWavePreview(outputCanvas, outCtx, state.outputSamples, ratio);
          if(state.specOutBitmap)
              drawSpecWithCursor(specOutCtx, state.specOutBitmap, specOutCanvas.width, specOutCanvas.height, ratio);
      }
  }
  requestAnimationFrame(updateVisuals);

  btnPlayInput.addEventListener("click", () => {
    if(audioIn.paused){ audioIn.play(); btnPlayInput.textContent = "Pause Input"; }
    else { audioIn.pause(); btnPlayInput.textContent = "Play Input"; }
  });

  btnPlayOutput.addEventListener("click", () => {
    if(audioOut.paused){ audioOut.play(); btnPlayOutput.textContent = "Pause Output"; }
    else { audioOut.pause(); btnPlayOutput.textContent = "Play Output"; }
  });

  btnSyncReset.addEventListener("click", () => {
    audioIn.pause(); audioOut.pause();
    audioIn.currentTime = 0; audioOut.currentTime = 0;
    btnPlayInput.textContent = "Play Input"; btnPlayOutput.textContent = "Play Output";

    if(state.inputSamples.length > 0) drawWavePreview(inputCanvas, inCtx, state.inputSamples, 0);
    if(state.outputSamples.length > 0) drawWavePreview(outputCanvas, outCtx, state.outputSamples, 0);
    if(state.specInBitmap) drawSpecWithCursor(specInCtx, state.specInBitmap, specInCanvas.width, specInCanvas.height, 0);
    if(state.specOutBitmap) drawSpecWithCursor(specOutCtx, state.specOutBitmap, specOutCanvas.width, specOutCanvas.height, 0);
  });
}

function init(){ bindUpload(); bindSpectrumSelection(); bindPlayback(); bindToggles(); bindSaveLoad(); setStatus("Ready."); }
document.addEventListener("DOMContentLoaded", init);