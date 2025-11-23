/* static/js/main.js — single entry */

// ---------- tiny helpers ----------
const $  = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));
function firstSel(...sels){ for(const s of sels){ const el=$(s); if(el) return el; } return null; }
function setStatus(msg){ const el = firstSel("#statusbar","[data-role=status]"); if(el) el.textContent = msg; console.log(msg); }
function downloadBlob(data, filename, type="application/octet-stream"){
  const blob = new Blob([data], {type}); const url = URL.createObjectURL(blob);
  const a = document.createElement("a"); a.href=url; a.download=filename; document.body.appendChild(a); a.click();
  URL.revokeObjectURL(url); a.remove();
}

// ---------- DOM bindings (match index.html) ----------
const btnOpen         = firstSel("#btn-open");
const fileInput       = firstSel("#file-hidden");
const dropZone        = firstSel("#drop-zone");

const btnSaveSettings = firstSel("#btn-save-settings");
const btnLoadSettings = firstSel("#btn-load-settings");
const modeSelect      = firstSel("#mode-select");
const btnScaleSwitch  = firstSel("#fft-scale");
const chkShowSpec     = firstSel("#chk-spec");
const btnAIPanel      = firstSel("#btn-ai-panel");

const eqPanel         = firstSel("#eq-sliders");

const spectrumCanvas  = firstSel("#fft-canvas");
const spectrumCtx     = spectrumCanvas ? spectrumCanvas.getContext("2d") : null;
const spectrumLoader  = firstSel("#spectrum-loader");

const inputCanvas     = firstSel("#wave-in");
const outputCanvas    = firstSel("#wave-out");
const inCtx           = inputCanvas ? inputCanvas.getContext("2d") : null;
const outCtx          = outputCanvas ? outputCanvas.getContext("2d") : null;

const specInCanvas    = firstSel("#spec-in");
const specOutCanvas   = firstSel("#spec-out");
const specInCtx       = specInCanvas ? specInCanvas.getContext("2d") : null;
const specOutCtx      = specOutCanvas ? specOutCanvas.getContext("2d") : null;

const btnAddSubBand   = firstSel("#btn-add-subband");
const btnClearSubBand = firstSel("#btn-clear-subband");
const btnSaveScheme   = firstSel("#btn-scheme-save");
const btnLoadScheme   = firstSel("#btn-scheme-load");

const audioIn         = firstSel("#audio-in");
const audioOut        = firstSel("#audio-out");
const btnPlayInput    = firstSel("#play-input");
const btnPlayOutput   = firstSel("#play-output");
const btnSyncReset    = firstSel("#sync-reset");

// ---------- app state ----------
const state = {
  signalId:null, sr:0, duration:0, nSamples:0, fmax:0,
  spectrumMags: [],
  scale:"linear", showSpectrograms:true, mode:"generic",
  subbands:[], // For Generic Mode
  customSliders:[], // For Customized Modes
  selecting:false, selStartX:0, selEndX:0
};

// ---------- net ----------
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

// ---------- upload ----------
function bindUpload(){
  if(btnOpen) btnOpen.addEventListener("click", () => fileInput && fileInput.click());

  if(dropZone){
    dropZone.addEventListener("click", () => fileInput && fileInput.click());
    ["dragenter","dragover"].forEach(ev => dropZone.addEventListener(ev, e => { e.preventDefault(); dropZone.classList.add("drag"); }));
    ["dragleave","drop"].forEach(ev => dropZone.addEventListener(ev, e => { e.preventDefault(); dropZone.classList.remove("drag"); }));
    dropZone.addEventListener("drop", (e) => {
      const f = e.dataTransfer?.files?.[0]; if(f) doUploadFile(f);
    });
  }

  if(fileInput){
    fileInput.addEventListener("change", (e) => {
      const f = e.target.files?.[0]; if(f) doUploadFile(f);
    });
  }
}

async function doUploadFile(file){
  try{
    setStatus(`Uploading: ${file.name} ...`);
    if(spectrumLoader) spectrumLoader.classList.remove("hidden");
    const fd = new FormData(); fd.append("signal", file);
    const res = await apiPost("/api/upload/", fd, false); // server returns JSON
    const j   = typeof res === "object" ? res : JSON.parse(new TextDecoder().decode(res));
    state.signalId = j.signal_id; state.sr = j.sr; state.duration = j.duration; state.nSamples = j.n;
    setStatus(`Loaded ${j.file_name} — sr=${j.sr}Hz, len=${j.duration.toFixed(2)}s`);
    await refreshAll();
  }catch(err){
    console.error(err);
    setStatus(`Upload error: ${err.message}`);
    if(spectrumLoader) spectrumLoader.classList.add("hidden");
  }
}

// ---------- drawing ----------
function clearCanvas(ctx, cvs){ if(!ctx||!cvs) return; ctx.clearRect(0,0,cvs.width,cvs.height); ctx.fillStyle="#000000"; ctx.fillRect(0,0,cvs.width,cvs.height); }

// ---
// --- THIS FUNCTION IS NOW FIXED ---
// ---
function drawSpectrum(mags,fmax,canvas,ctx,scale="linear"){
  if(!canvas||!ctx||!Array.isArray(mags)) return;
  clearCanvas(ctx,canvas);

  const W=canvas.width, H=canvas.height;
  ctx.strokeStyle="#d62976"; // Pink
  ctx.beginPath();
  const N=mags.length;

  for(let i=0;i<N;i++){
    const x = (i/(N-1)) * W;
    let yv = mags[i]; // This is already 0-1 and log-scaled

    // yv=Math.log10(1+9*yv); // <-- THIS WAS THE BUG. REMOVED.

    const y = H - (yv * H); // Directly map 0-1 to canvas height

    if(i===0) ctx.moveTo(x,y);
    else ctx.lineTo(x,y);
  }
  ctx.stroke();

  if(state.mode==="generic" && state.selecting){
    const x1=Math.min(state.selStartX,state.selEndX), x2=Math.max(state.selStartX,state.selEndX);
    ctx.fillStyle="rgba(214, 41, 118, 0.25)";
    ctx.fillRect(x1,0,x2-x1,H);
  }

  ctx.strokeStyle="#363636";
  ctx.beginPath();
  ctx.moveTo(0,H-0.5);
  ctx.lineTo(W,H-0.5);
  ctx.stroke();
}

function drawWavePreview(canvas,ctx,samples){
  if(!canvas||!ctx||!Array.isArray(samples)) return; clearCanvas(ctx,canvas);
  const W=canvas.width,H=canvas.height, mid=H/2; ctx.strokeStyle="#a8a8a8"; ctx.beginPath(); const N=samples.length; // Muted text color
  for(let i=0;i<N;i++){ const x=(i/(N-1))*W; const y=mid - samples[i]*mid; if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y); }
  ctx.stroke();
}
function drawImageBase64(canvas,ctx,b64){ const img=new Image(); img.onload=()=>{ canvas.height=img.height; canvas.width=img.width; ctx.drawImage(img,0,0,canvas.width,canvas.height); }; img.src=`data:image/png;base64,${b64}`; }

// ---------- backend refresh ----------
async function refreshAll(){
  if(!state.signalId) return;
  if(spectrumLoader) spectrumLoader.classList.remove("hidden");

  if(audioIn) audioIn.src = `/api/audio/${state.signalId}/input.wav`;
  if(audioOut) audioOut.src = `/api/audio/${state.signalId}/output.wav`;

  const meta   = await apiGet(`/api/summary/${state.signalId}/`);
  const spec   = await apiGet(`/api/spectrum/${state.signalId}/?scale=${state.scale}`);
  const waves  = await apiGet(`/api/wave_previews/${state.signalId}/`);
  const specs  = state.showSpectrograms ? await apiGet(`/api/spectrograms/${state.signalId}/`) : null;

  const jSpec  = typeof spec  === "object" ? spec  : JSON.parse(new TextDecoder().decode(spec));
  const jWaves = typeof waves === "object" ? waves : JSON.parse(new TextDecoder().decode(waves));

  state.fmax = jSpec.fmax;
  state.spectrumMags = jSpec.mags;
  drawSpectrum(jSpec.mags, jSpec.fmax, spectrumCanvas, spectrumCtx, state.scale);
  if(spectrumLoader) spectrumLoader.classList.add("hidden");

  drawWavePreview(inputCanvas, inCtx, jWaves.input);
  drawWavePreview(outputCanvas, outCtx, jWaves.output);

  if(specs){
    const jSpecs = typeof specs === "object" ? specs : JSON.parse(new TextDecoder().decode(specs));
    if(specInCtx && jSpecs.in_png)  drawImageBase64(specInCanvas,  specInCtx,  jSpecs.in_png);
    if(specOutCtx && jSpecs.out_png)drawImageBase64(specOutCanvas, specOutCtx, jSpecs.out_png);
  }

  if(typeof meta === "object"){
    const sb = $("#sb-file"), fs=$("#sb-fs"), ln=$("#sb-len");
    if(sb) sb.textContent = meta.file_name || "—";
    if(fs) fs.textContent = meta.sr || "—";
    if(ln) ln.textContent = (meta.duration ?? 0).toFixed(2);
  }

  renderEqSliders();
}

// ---------- spectrum interaction (generic) ----------
function bindSpectrumSelection(){
  if(!spectrumCanvas) return; const cvs=spectrumCanvas;
  cvs.addEventListener("mousedown",(e)=>{ if(state.mode!=="generic") return; state.selecting=true; const r=cvs.getBoundingClientRect(); state.selStartX=e.clientX-r.left; state.selEndX=state.selStartX; redrawSpectrum(); });
  cvs.addEventListener("mousemove",(e)=>{ if(!state.selecting) return; const r=cvs.getBoundingClientRect(); state.selEndX=e.clientX-r.left; redrawSpectrum(); });
  window.addEventListener("mouseup", async ()=>{
    if(!state.selecting) return;
    state.selecting=false;
    redrawSpectrum();
    const band=await promptBandFromSelection();
    if(band){
      state.subbands.push(band);
      renderEqSliders();
      await applyEqualizer();
    }
  });
}

function redrawSpectrum(){
  if(!state.signalId || !state.spectrumMags) return;
  drawSpectrum(state.spectrumMags, state.fmax, spectrumCanvas, spectrumCtx, state.scale);
}

function pxToFreq(x,W,fmax){ const frac=Math.min(1,Math.max(0,x/W)); return frac*fmax; }
async function promptBandFromSelection(){
  const W = spectrumCanvas.width;
  const x1=Math.min(state.selStartX,state.selEndX), x2=Math.max(state.selStartX,state.selEndX);
  let fmin=+pxToFreq(x1,W,state.fmax).toFixed(1), fmax=+pxToFreq(x2,W,state.fmax).toFixed(1);
  const resp = window.prompt(`Sub-band:\nMin Hz, Max Hz, Gain (0..2)\n`, `${fmin}, ${fmax}, 1.0`); if(!resp) return null;
  const p = resp.split(",").map(s=>+s.trim()); if(p.length<3||p.some(Number.isNaN)) return null;
  const [mn,mx,g]=p; return {id:`sb${Date.now()}`, fmin:Math.min(mn,mx), fmax:Math.max(mn,mx), gain:Math.max(0,Math.min(2,g))};
}

// ---------- equalizer UI ----------
function renderEqSliders(){
  if(state.mode === 'generic'){
    renderGenericSubbands();
    $('#generic-tools').style.display = 'flex';
  } else {
    renderCustomizedSliders();
    $('#generic-tools').style.display = 'none';
  }
}

function renderGenericSubbands(){
  if(!eqPanel) return;
  eqPanel.innerHTML=""; // Clear panel
  state.subbands.forEach((b,idx)=>{
    const row=document.createElement("div"); row.className="sb-row";
    row.innerHTML = `
      <div class="sb-title">SubBand ${idx+1} [${b.fmin.toFixed(1)}–${b.fmax.toFixed(1)} Hz]</div>
      <input type="range" min="0" max="2" step="0.01" value="${b.gain}" data-id="${b.id}"/>
      <span class="sb-gain">${b.gain.toFixed(2)}x</span>
      <button data-act="edit" data-id="${b.id}">Edit</button>
      <button data-act="del"  data-id="${b.id}" class="btn-danger">Delete</button>`;
    eqPanel.appendChild(row);
  });
  eqPanel.oninput = async (e)=>{ const r=e.target; if(r.tagName==="INPUT"&&r.type==="range"){ const id=r.dataset.id; const sb=state.subbands.find(s=>s.id===id); if(sb){ sb.gain=+r.value; r.parentElement.querySelector(".sb-gain").textContent=`${sb.gain.toFixed(2)}x`; await applyEqualizerDebounced(); }}};
  eqPanel.onclick  = async (e)=>{ const b=e.target.closest("button"); if(!b) return; const id=b.dataset.id; const sb=state.subbands.find(s=>s.id===id); if(!sb) return;
    if(b.dataset.act==="del"){ state.subbands=state.subbands.filter(s=>s.id!==id); renderEqSliders(); await applyEqualizer(); }
    else { const resp=window.prompt(`Edit [min,max,gain]`, `${sb.fmin}, ${sb.fmax}, ${sb.gain}`); if(!resp) return; const p=resp.split(",").map(s=>+s.trim()); if(p.length<3) return; sb.fmin=Math.min(p[0],p[1]); sb.fmax=Math.max(p[0],p[1]); sb.gain=Math.max(0,Math.min(2,p[2])); renderEqSliders(); await applyEqualizer(); }
  };
}

async function renderCustomizedSliders(){
  if(!eqPanel || !state.signalId) return;
  eqPanel.innerHTML = "<p>Loading sliders...</p>"; // Clear panel

  try {
    const modeName = state.mode === 'music' ? 'musical instruments' : (state.mode === 'animals' ? 'animal sounds' : 'human voices');
    const resp = await apiGet(`/api/custom_conf/${state.signalId}/?mode=${modeName}`);
    state.customSliders = resp.sliders || [];

    eqPanel.innerHTML = "";

    if(state.customSliders.length === 0){
      eqPanel.innerHTML = "<p>No sliders defined for this mode.</p>";
      return;
    }

    state.customSliders.forEach((slider, idx) => {
      const row = document.createElement("div"); row.className = "sb-row";
      slider.id = `custom${idx}`;
      row.innerHTML = `
        <div class="sb-title">${slider.name}</div>
        <input type="range" min="0" max="2" step="0.01" value="${slider.gain}" data-id="${slider.id}"/>
        <span class="sb-gain">${slider.gain.toFixed(2)}x</span>`;
      eqPanel.appendChild(row);
    });

    eqPanel.oninput = async (e) => {
      const r = e.target;
      if (r.tagName === "INPUT" && r.type === "range") {
        const id = r.dataset.id;
        const slider = state.customSliders.find(s => s.id === id);
        if (slider) {
          slider.gain = +r.value;
          r.parentElement.querySelector(".sb-gain").textContent = `${slider.gain.toFixed(2)}x`;
          await applyEqualizerDebounced();
        }
      }
    };
    eqPanel.onclick = null;

  } catch(err) {
    console.error(err);
    eqPanel.innerHTML = `<p style="color:var(--danger);">Error loading sliders.</p>`;
  }
}

// ---------- apply equalizer ----------
let eqTimer=null;
async function applyEqualizerDebounced(){ if(eqTimer) clearTimeout(eqTimer); eqTimer=setTimeout(applyEqualizer,120); }
async function applyEqualizer(){
  if(!state.signalId) return;

  const payload = state.mode==="generic"
    ? {mode:"generic", subbands:state.subbands}
    : {mode:state.mode, sliders:state.customSliders};

  try{
    if(spectrumLoader) spectrumLoader.classList.remove("hidden");
    await apiPost(`/api/equalize/${state.signalId}/`, payload);
    await refreshOutputs();
  }
  catch(err){
    console.error(err);
    setStatus(`Equalize error: ${err.message}`);
    if(spectrumLoader) spectrumLoader.classList.add("hidden");
  }
}

async function refreshOutputs(){
  if(!state.signalId) return;

  if(spectrumLoader) spectrumLoader.classList.remove("hidden");

  const audioReload = audioOut.src = `/api/audio/${state.signalId}/output.wav?t=${Date.now()}`;
  const specPromise = apiGet(`/api/spectrum/${state.signalId}/?scale=${state.scale}`);
  const wavesPromise = apiGet(`/api/wave_previews/${state.signalId}/`);
  const specsPromise = state.showSpectrograms ? apiGet(`/api/spectrograms/${state.signalId}/`) : Promise.resolve(null);

  const [spec, waves, specs] = await Promise.all([specPromise, wavesPromise, specsPromise]);

  const jSpec  = typeof spec  === "object" ? spec  : JSON.parse(new TextDecoder().decode(spec));
  const jWaves = typeof waves === "object" ? waves : JSON.parse(new TextDecoder().decode(waves));

  state.fmax = jSpec.fmax;
  state.spectrumMags = jSpec.mags;
  drawSpectrum(jSpec.mags, jSpec.fmax, spectrumCanvas, spectrumCtx, state.scale);
  drawWavePreview(outputCanvas, outCtx, jWaves.output);

  if(specs){
    const jSpecs = typeof specs === "object" ? specs : JSON.parse(new TextDecoder().decode(specs));
    if(specOutCtx && jSpecs.out_png)drawImageBase64(specOutCanvas, specOutCtx, jSpecs.out_png);
  }

  if(spectrumLoader) spectrumLoader.classList.add("hidden");
}


// ---------- toggles / mode ----------
function bindToggles(){
  if(btnScaleSwitch) btnScaleSwitch.addEventListener("click", async ()=>{
    if(!state.signalId) return;
    state.scale = state.scale==="linear" ? "audiogram" : "linear";
    btnScaleSwitch.textContent = `Audiogram: ${state.scale==="audiogram"?"On":"Off"}`;
    await refreshAll();
  });

  if(chkShowSpec) chkShowSpec.addEventListener("change", async e => {
    if(!state.signalId) return;
    state.showSpectrograms = !!e.target.checked;
    await refreshAll();
  });

  if(btnAddSubBand) btnAddSubBand.addEventListener("click", ()=> {
    if(!state.signalId) return alert("Upload a signal first.");
    alert("Select an interval on the spectrum by dragging with the mouse.");
  });

  if(btnClearSubBand) btnClearSubBand.addEventListener("click", async ()=>{
    if(state.mode !== 'generic' || !state.signalId) return;
    state.subbands = [];
    renderEqSliders();
    await applyEqualizer();
  });

  if(modeSelect) modeSelect.addEventListener("change", async e => {
    if(!state.signalId) { e.target.value = state.mode; return; } // Prevent change if no signal
    state.mode = e.target.value;
    state.subbands=[];
    state.customSliders=[];
    renderEqSliders();
    await applyEqualizer();
  });
}

// ---------- save/load scheme & settings ----------
function bindSaveLoad(){
  if(btnSaveScheme) btnSaveScheme.addEventListener("click", async ()=>{
    if(!state.signalId) return alert("Upload a signal first.");
    const scheme = state.mode==="generic" ? {mode:"generic", subbands:state.subbands} : {mode:state.mode, sliders:state.customSliders};
    const buf = await apiPost(`/api/save_scheme/${state.signalId}/`, scheme);
    const j   = typeof buf==="object" ? buf : JSON.parse(new TextDecoder().decode(buf));
    downloadBlob(new TextEncoder().encode(JSON.stringify(j.data,null,2)), j.filename, "application/json");
  });
  if(btnLoadScheme) btnLoadScheme.addEventListener("click", async ()=>{
    if(!state.signalId) return alert("Upload a signal first.");
    const inp=document.createElement("input"); inp.type="file"; inp.accept=".json,application/json";
    inp.onchange=async()=>{ const f=inp.files?.[0]; if(!f) return; const data=JSON.parse(await f.text());
      await apiPost(`/api/load_scheme/${state.signalId}/`, data);
      state.mode=data.mode||"generic"; state.subbands=data.subbands||[]; state.customSliders=data.sliders||[];
      if(modeSelect) modeSelect.value=state.mode;
      renderEqSliders();
      await applyEqualizer();
    };
    inp.click();
  });
  if(btnSaveSettings) btnSaveSettings.addEventListener("click", async ()=>{
    if(!state.signalId) return alert("Upload a signal first.");
    const full = { scale:state.scale, showSpectrograms:state.showSpectrograms, ...(state.mode==="generic"?{mode:"generic",subbands:state.subbands}:{mode:state.mode,sliders:state.customSliders}) };
    const buf = await apiPost(`/api/save_settings/${state.signalId}/`, full);
    const j   = typeof buf==="object" ? buf : JSON.parse(new TextDecoder().decode(buf));
    downloadBlob(new TextEncoder().encode(JSON.stringify(j.data,null,2)), j.filename, "application/json");
  });
  if(btnLoadSettings) btnLoadSettings.addEventListener("click", async ()=>{
    // if(!state.signalId) return alert("Upload a signal first.");
    const inp=document.createElement("input"); inp.type="file"; inp.accept=".json,application/json";
    inp.onchange=async()=>{ const f=inp.files?.[0]; if(!f) return; const data=JSON.parse(await f.text());
      await apiPost(`/api/load_settings/${state.signalId}/`, data);
      state.scale=data.scale||"linear"; state.showSpectrograms=!!data.showSpectrograms; state.mode=data.mode||"generic"; state.subbands=data.subbands||[]; state.customSliders=data.sliders||[];
      if(chkShowSpec) chkShowSpec.checked=state.showSpectrograms; if(modeSelect) modeSelect.value=state.mode;
      await refreshAll();
    };
    inp.click();
  });
}

// ---------- playback ----------
function bindPlayback(){
  if(!audioIn || !audioOut) return;

  // --- Players are now independent ---

  if(btnPlayInput)  btnPlayInput.addEventListener("click", () => {
    audioOut.pause(); // Pause the other player
    audioIn.play();
  });

  if(btnPlayOutput) btnPlayOutput.addEventListener("click", () => {
    audioIn.pause(); // Pause the other player
    audioOut.play();
  });

  if(btnSyncReset)  btnSyncReset.addEventListener("click", () => {
    audioIn.pause();
    audioOut.pause();
    audioIn.currentTime = 0;
    audioOut.currentTime = 0;
  });
}

// ---------- init ----------
function init(){
  bindUpload();
  bindSpectrumSelection();
  bindPlayback();
  bindToggles();
  bindSaveLoad();
  setStatus("Ready.");
}
document.addEventListener("DOMContentLoaded", init);