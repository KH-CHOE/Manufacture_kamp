/* Foundry Guard · Live — SSE 틱 수신 → 가로 파이프라인 미믹 갱신 + 기본 제어(P2) */
(() => {
  const $ = (s) => document.querySelector(s);
  const esc = (t) => String(t).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const fmt = (v) => (v == null ? "—" : (Math.abs(v) >= 100 ? Math.round(v) : Math.round(v * 100) / 100));

  let built = false;
  const stationEls = {};

  // ── 파이프라인 뼈대 1회 생성 ───────────────────────────────
  function build(stations) {
    const wrap = $("#pipeline");
    wrap.innerHTML = "";
    stations.forEach((s, i) => {
      if (i > 0) {
        const pipe = document.createElement("div");
        pipe.className = "pipe";
        wrap.appendChild(pipe);
      }
      const el = document.createElement("div");
      el.className = "station";
      el.dataset.id = s.id;
      wrap.appendChild(el);
      stationEls[s.id] = el;
    });
    built = true;
  }

  function featRow(f) {
    let barW = 0, cls = "feat";
    if (!f.spec) cls += " nospec";
    if (f.oob) cls += " oob";
    if (f.spec && f.low != null && f.high != null && f.value != null) {
      const frac = (f.value - f.low) / ((f.high - f.low) || 1);
      barW = Math.max(0, Math.min(1, frac)) * 100;
    } else if (f.spec) { barW = 50; }
    const grade = f.grade ? `<span class="feat-grade">${esc(f.grade)}</span>` : "";
    return `<div class="${cls}">
      <div class="feat-top"><span class="feat-label">${esc(f.label)}${grade}</span>
        <span class="feat-val">${fmt(f.value)}</span></div>
      <div class="feat-bar"><i style="width:${barW}%"></i></div></div>`;
  }

  function station(s) {
    const el = stationEls[s.id];
    if (!el) return;
    el.className = `station ${s.status}${s.stage === "post" ? " post" : ""}`;
    const tag = s.stage === "post"
      ? `<span class="st-badge st-post-tag">사후측정</span>`
      : `<span class="st-badge">${s.status === "alert" ? "이상" : s.status === "warn" ? "주의" : "정상"}</span>`;
    el.innerHTML = `<div class="st-head"><span class="st-name">${esc(s.label)}</span>${tag}</div>`
      + s.features.map(featRow).join("");
  }

  // ── 틱 반영 ────────────────────────────────────────────────
  function render(t) {
    if (!built) build(t.stations);

    $("#clock").textContent = t.time || "—";
    const badge = $("#risk-badge");
    badge.className = `risk-badge ${t.risk_level}`;
    $("#risk-level").textContent = t.risk_level;

    const prob = t.probability ?? 0;
    $("#prob-val").textContent = prob.toFixed(3);
    const fill = $("#gauge-fill");
    fill.style.right = `${(1 - prob) * 100}%`;
    fill.style.background = t.risk_level === "위험" ? "var(--red)"
      : t.risk_level === "주의" ? "linear-gradient(90deg,var(--green),var(--amber))" : "var(--green)";
    $("#gauge-thr").style.left = `${t.threshold * 100}%`;
    $("#thr-val").textContent = `임계 ${t.threshold}`;

    $("#c-index").textContent = `${t.index} / ${t.total}`;
    $("#c-actual").textContent = t.actual == null ? "—" : t.actual;
    $("#c-highrisk").textContent = t.high_risk_total ?? "—";

    t.stations.forEach(station);

    document.querySelectorAll(".pipe").forEach((p) => p.classList.toggle("stopped", !t.running));

    const rp = $("#reasons");
    if (t.top_reasons && t.top_reasons.length && t.risk_level !== "정상") {
      rp.hidden = false;
      $("#reason-list").innerHTML = t.top_reasons.map((r) =>
        `<div class="reason"><b>${esc(r.label)}</b> ${fmt(r.value)}
          <span>· 정상 ${fmt(r.normal[0])}~${fmt(r.normal[1])}${r.grade ? " · " + esc(r.grade) + "등급" : ""}</span></div>`).join("");
    } else { rp.hidden = true; }

    // 재생 버튼·주입 태그 동기화
    const play = $("#btn-play");
    play.classList.toggle("playing", t.running);
    play.textContent = t.running ? "⏸ 일시정지" : "▶ 재생";
    $("#inject-tag").hidden = !t.injected;
  }

  // ── 현황 보고 · 이상 이력 ──────────────────────────────────
  function renderBriefing(b) {
    if (!b) return;
    $("#briefing").textContent = b.text || "데이터 수집 중…";
    $("#brief-idx").textContent = b.index != null ? `#${b.index}` : "";
    const s = $("#brief-stats");
    if (b.processed) {
      s.innerHTML = `<span>처리 <b>${b.processed}</b></span>`
        + `<span>고위험 <b>${b.high_risk}</b></span>`
        + (b.defects != null ? `<span>실측불량 <b>${b.defects}</b></span>` : "")
        + `<span>추세 <b>${esc(b.trend)}</b></span>`;
    } else { s.innerHTML = ""; }
  }

  const alerts = [];
  function alertHTML(a) {
    const r = (a.reasons && a.reasons[0]) ? a.reasons[0] : null;
    const reason = r ? `${esc(r.label)} ${fmt(r.value)} (정상 ${fmt(r.normal[0])}~${fmt(r.normal[1])})` : "원인 분석 중";
    return `<div class="alert-item"><div class="ai-top">
        <span class="ai-time">${esc(a.time || "#" + a.index)}</span>
        <span class="ai-prob">위험 ${(a.probability ?? 0).toFixed(3)}</span></div>
      <div class="ai-reason">${reason}</div></div>`;
  }
  function renderAlerts(list) {
    alerts.length = 0; alerts.push(...list);
    const box = $("#alert-list");
    box.innerHTML = list.length ? list.map(alertHTML).join("") : '<p class="empty">경보 없음</p>';
    $("#alert-count").textContent = list.length;
  }
  function addAlert(a) {
    alerts.unshift(a); alerts.splice(20);
    renderAlerts(alerts);
  }

  // ── 제어 ───────────────────────────────────────────────────
  async function control(action, extra = {}) {
    try {
      const res = await fetch("/live/control", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action, ...extra }),
      });
      const data = await res.json();
      if (data.tick) render(data.tick);
    } catch (e) { console.error(e); }
  }

  let playing = false;
  $("#btn-play").onclick = () => { playing = !playing; control(playing ? "start" : "stop"); };
  $("#speed").onchange = (e) => control("speed", { value: Number(e.target.value) });
  $("#btn-defect").onclick = () => control("seek_defect", { lead: 6 });
  $("#btn-inject").onclick = () => control("inject");
  $("#btn-clear").onclick = () => control("clear_inject");
  $("#btn-reset").onclick = () => { playing = false; control("reset"); };

  // ── SSE 연결 ───────────────────────────────────────────────
  function connect() {
    // 스냅샷 모드(?snap): SSE 미연결, 상태 1회만 렌더 (헤드리스 캡처용)
    if (new URLSearchParams(location.search).has("snap")) {
      $("#conn").className = "conn on"; $("#conn").innerHTML = '<i class="dot"></i> 스냅샷';
      fetch("/live/tick").then((r) => r.json()).then(render).catch(() => {});
      fetch("/live/briefing").then((r) => r.json()).then(renderBriefing).catch(() => {});
      fetch("/live/alerts").then((r) => r.json()).then((d) => renderAlerts(d.alerts || [])).catch(() => {});
      return;
    }
    const es = new EventSource("/live/stream");
    es.onopen = () => { $("#conn").className = "conn on"; $("#conn").innerHTML = '<i class="dot"></i> 실시간 연결됨'; };
    es.addEventListener("tick", (ev) => { try { render(JSON.parse(ev.data)); } catch {} });
    es.addEventListener("briefing", (ev) => { try { renderBriefing(JSON.parse(ev.data)); } catch {} });
    es.addEventListener("alert", (ev) => { try { addAlert(JSON.parse(ev.data)); } catch {} });
    es.onerror = () => {
      $("#conn").className = "conn off"; $("#conn").innerHTML = '<i class="dot"></i> 연결 끊김 · 재시도';
    };
  }
  connect();
})();
