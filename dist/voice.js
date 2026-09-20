/* Foundry Guard · Live 음성(P5)
   OpenAI Realtime(주경로) + 브라우저 Web Speech(무키 대체·무대 안전망).
   안전 원칙: 음성은 명령을 '실행'하지 않는다. LiveAPI.command 로 조정창 제안만 →
   사람이 승인해야 반영(오인식 방지). 경보는 음성으로 안내. */
(() => {
  const $ = (s) => document.querySelector(s);

  let engine = null;         // 'realtime' | 'webspeech' | null
  let listening = false;
  let recog = null;                              // Web Speech
  let pc = null, dc = null, micStream = null;    // Realtime WebRTC
  let rtSession = null, rtModel = null;

  // ── 공용 TTS ───────────────────────────────────────────────
  const synth = window.speechSynthesis;
  function speak(text) {
    if (!synth || !text) return;
    try { synth.cancel(); const u = new SpeechSynthesisUtterance(text); u.lang = "ko-KR"; u.rate = 1.05; synth.speak(u); }
    catch {}
  }

  // ── 엔진 공통: 인식된 말 → 명령 제안 ───────────────────────
  async function onTranscript(text, isFinal) {
    if (!text) return;
    $("#voice-heard").textContent = (isFinal ? "🗣 " : "… ") + text;
    if (!isFinal || !window.LiveAPI) return;
    const r = await window.LiveAPI.handle(text, "음성");
    if (r.type === "answer") speak(r.text);                                   // 질문 → 말로 답변
    else if (r.type === "command" && engine !== "realtime") speak("명령을 확인했습니다. 조정 창에서 승인해 주세요.");
    else if (r.type === "none") speak("명령이나 질문을 이해하지 못했습니다. 다시 말씀해 주세요.");
    setTimeout(() => { $("#voice-heard").textContent = ""; }, 4000);
  }

  // 경보 음성 안내(음성 켜져 있을 때만)
  if (window.LiveAPI) window.LiveAPI.onAlert = (a) => {
    if (!listening) return;
    const r = a.reasons && a.reasons[0];
    speak(`경고. 불량 위험 ${a.probability}. ${r ? r.label + " 이탈." : ""}`);
  };

  // ── Web Speech ─────────────────────────────────────────────
  function startWebSpeech() {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) return false;
    recog = new SR();
    recog.lang = "ko-KR"; recog.continuous = true; recog.interimResults = true;
    recog.onresult = (e) => {
      for (let i = e.resultIndex; i < e.results.length; i++) {
        onTranscript(e.results[i][0].transcript.trim(), e.results[i].isFinal);
      }
    };
    recog.onerror = (e) => setStatus("음성 오류: " + e.error);
    recog.onend = () => { if (listening) { try { recog.start(); } catch {} } };
    try { recog.start(); return true; } catch { return false; }
  }
  function stopWebSpeech() {
    if (recog) { recog.onend = null; try { recog.stop(); } catch {} recog = null; }
  }

  // ── OpenAI Realtime (WebRTC) ───────────────────────────────
  async function startRealtime() {
    const token = rtSession && rtSession.client_secret && rtSession.client_secret.value;
    if (!token) return false;
    pc = new RTCPeerConnection();
    const audio = document.createElement("audio"); audio.autoplay = true;
    pc.ontrack = (e) => { audio.srcObject = e.streams[0]; };
    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    micStream.getTracks().forEach((t) => pc.addTrack(t, micStream));
    dc = pc.createDataChannel("oai-events");
    dc.onmessage = (e) => {
      try {
        const ev = JSON.parse(e.data);
        if (ev.type === "conversation.item.input_audio_transcription.completed")
          onTranscript((ev.transcript || "").trim(), true);
        else if (ev.type === "conversation.item.input_audio_transcription.delta")
          $("#voice-heard").textContent = "… " + (ev.delta || "");
      } catch {}
    };
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    const resp = await fetch(`https://api.openai.com/v1/realtime?model=${encodeURIComponent(rtModel)}`, {
      method: "POST", body: offer.sdp,
      headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/sdp", "OpenAI-Beta": "realtime=v1" },
    });
    if (!resp.ok) throw new Error("SDP 교환 실패 " + resp.status);
    await pc.setRemoteDescription({ type: "answer", sdp: await resp.text() });
    return true;
  }
  function stopRealtime() {
    if (dc) { try { dc.close(); } catch {} dc = null; }
    if (pc) { try { pc.close(); } catch {} pc = null; }
    if (micStream) { micStream.getTracks().forEach((t) => t.stop()); micStream = null; }
  }

  // ── 토글 ───────────────────────────────────────────────────
  async function start() {
    if (listening) return;
    listening = true; setBtn(true); $("#voice-heard").textContent = "";
    if (engine === "realtime") {
      try { await startRealtime(); setStatus("음성 켜짐 · OpenAI Realtime"); return; }
      catch (e) { setStatus("Realtime 실패 → 브라우저 인식 전환"); engine = "webspeech"; }
    }
    if (engine === "webspeech" || engine === null) {
      if (startWebSpeech()) setStatus("음성 켜짐 · 브라우저 인식");
      else { setStatus("이 브라우저는 음성인식을 지원하지 않습니다"); listening = false; setBtn(false); }
    }
  }
  function stop() {
    listening = false; setBtn(false);
    stopWebSpeech(); stopRealtime();
    setStatus(engineLabel());
  }
  const toggle = () => (listening ? stop() : start());

  // ── UI ─────────────────────────────────────────────────────
  function engineLabel() {
    return engine === "realtime" ? "음성 준비 · OpenAI Realtime"
      : engine === "webspeech" ? "음성 준비 · 브라우저 인식(무키)"
      : "음성 미지원";
  }
  function setBtn(on) {
    const b = $("#voice-btn");
    if (b) { b.classList.toggle("on", on); b.innerHTML = on ? "🔴 듣는 중…" : "🎤 음성"; }
  }
  function setStatus(t) { const s = $("#voice-status"); if (s) s.textContent = t; }

  function mount() {
    const bar = $(".cmd-bar");
    if (!bar) return;
    const btn = document.createElement("button");
    btn.id = "voice-btn"; btn.className = "ctl voice"; btn.innerHTML = "🎤 음성"; btn.onclick = toggle;
    bar.insertBefore(btn, bar.firstChild);
    const heard = document.createElement("span");
    heard.id = "voice-heard"; heard.className = "voice-heard"; bar.appendChild(heard);
    const status = document.createElement("span");
    status.id = "voice-status"; status.className = "voice-status"; bar.appendChild(status);
  }

  async function init() {
    mount();
    const hasWebSpeech = !!(window.SpeechRecognition || window.webkitSpeechRecognition);
    try {
      const s = await fetch("/live/realtime/session").then((r) => r.json());
      if (s.enabled) { engine = "realtime"; rtSession = s.session; rtModel = s.model; }
      else engine = hasWebSpeech ? "webspeech" : null;
    } catch { engine = hasWebSpeech ? "webspeech" : null; }
    setStatus(engineLabel());
  }

  init();
})();
