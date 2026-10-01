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
    if (window.LiveAPI && window.LiveAPI.onAlert) window.LiveAPI.onAlert(a);
  }

  // ── 감사 로그 ──────────────────────────────────────────────
  const audits = [];
  function auditHTML(a) {
    const cls = /차단|취소/.test(a.result) ? (/차단/.test(a.result) ? "block" : "cancel") : "ok";
    return `<div class="audit-item ${cls}"><span class="au-ts">${esc(a.ts)}</span>
      <span class="au-act"><b>${esc(a.action)}</b> <small>${esc(a.detail || "")} · ${esc(a.actor)}</small></span>
      <span class="au-res">${esc(a.result)}</span></div>`;
  }
  function renderAudit(list) {
    audits.length = 0; audits.push(...list);
    const box = $("#audit-list");
    box.innerHTML = list.length ? list.map(auditHTML).join("") : '<p class="empty">기록 없음</p>';
    $("#audit-count").textContent = list.length;
  }
  function addAudit(a) { audits.unshift(a); audits.splice(50); renderAudit(audits); }

  // ── 조정 창(모달): AI 제안 → 사람 승인 ────────────────────
  let current = null;   // {proposal, actor}
  function openModal(proposal, actor) {
    current = { proposal, actor };
    $("#modal-title").textContent = proposal.title;
    $("#modal-summary").textContent = proposal.summary;
    const isThr = proposal.kind === "set_threshold";
    $("#modal-threshold").hidden = !isThr;
    const ok = $("#modal-ok");
    if (isThr) {
      $("#thr-current").textContent = proposal.current;
      $("#thr-proposed").textContent = proposal.proposed;
      const sl = $("#thr-slider");
      sl.min = proposal.min; sl.max = proposal.max; sl.step = proposal.step; sl.value = proposal.proposed;
      $("#thr-value").textContent = Number(proposal.proposed).toFixed(2);
      sl.oninput = () => { $("#thr-value").textContent = Number(sl.value).toFixed(2); };
      ok.textContent = "적용";
    } else {
      ok.textContent = proposal.kind === "stop_line" ? "정지 승인" : "재시작 승인";
    }
    const needOverride = !!proposal.requires_override;
    $("#modal-override").hidden = !needOverride;
    $("#override-chk").checked = false;
    ok.disabled = needOverride;
    $("#modal").hidden = false;
  }
  function closeModal() { $("#modal").hidden = true; current = null; }

  $("#override-chk").onchange = (e) => { $("#modal-ok").disabled = !e.target.checked; };
  $("#modal-cancel").onclick = () => {
    if (current) fetch("/live/reject", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ kind: current.proposal.kind, title: current.proposal.title, actor: current.actor }) });
    closeModal();
  };
  $("#modal-ok").onclick = async () => {
    if (!current) return;
    const p = current.proposal;
    const body = { kind: p.kind, actor: current.actor };
    if (p.kind === "set_threshold") body.value = Number($("#thr-slider").value);
    if (p.requires_override) body.override = $("#override-chk").checked;
    const res = await fetch("/live/approve", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const data = await res.json();
    if (data.ok) { if (data.tick) render(data.tick); closeModal(); }
    else { $("#modal-summary").textContent = data.error || "적용 실패"; }
  };

  // ── 명령(제안·모달) / 질문(답변) 통합 처리 ────────────────
  const post = (url, body) => fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).then((r) => r.json());

  // 특정 종류의 제안창 직접 열기(수동 버튼용)
  async function proposeKind(kind, actor) {
    $("#cmd-hint").textContent = "";
    const data = await post("/live/command", { kind });
    if (data.ok && data.proposal) openModal(data.proposal, actor);
    else $("#cmd-hint").textContent = data.error || "명령을 이해하지 못했습니다.";
  }

  // 자연어: 먼저 명령으로, 아니면 질문 답변으로. {type:'command'|'answer'|'none', text}
  async function handleText(text, actor) {
    $("#cmd-hint").textContent = "";
    const c = await post("/live/command", { text });
    if (c.ok && c.proposal) { openModal(c.proposal, actor); return { type: "command" }; }
    const a = await post("/live/ask", { text });
    if (a.answer) { $("#cmd-hint").textContent = "💬 " + a.answer; return { type: "answer", text: a.answer }; }
    $("#cmd-hint").textContent = "명령·질문을 이해하지 못했습니다.";
    return { type: "none" };
  }

  $("#cmd-send").onclick = () => {
    const v = $("#cmd-input").value.trim();
    if (v) { handleText(v, "텍스트"); $("#cmd-input").value = ""; }
  };
  $("#cmd-input").addEventListener("keydown", (e) => { if (e.key === "Enter") $("#cmd-send").click(); });
  $("#btn-threshold").onclick = () => proposeKind("set_threshold", "수동");

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
    const params = new URLSearchParams(location.search);
    if (params.has("snap")) {
      $("#conn").className = "conn on"; $("#conn").innerHTML = '<i class="dot"></i> 스냅샷';
      fetch("/live/tick").then((r) => r.json()).then(render).catch(() => {});
      fetch("/live/briefing").then((r) => r.json()).then(renderBriefing).catch(() => {});
      fetch("/live/alerts").then((r) => r.json()).then((d) => renderAlerts(d.alerts || [])).catch(() => {});
      fetch("/live/audit").then((r) => r.json()).then((d) => renderAudit(d.audit || [])).catch(() => {});
      const m = params.get("modal");
      if (m === "threshold") setTimeout(() => proposeKind("set_threshold", "음성"), 250);
      else if (m === "restart") setTimeout(() => proposeKind("start_line", "음성"), 250);
      return;
    }
    const es = new EventSource("/live/stream");
    es.onopen = () => { $("#conn").className = "conn on"; $("#conn").innerHTML = '<i class="dot"></i> 실시간 연결됨'; };
    es.addEventListener("tick", (ev) => { try { render(JSON.parse(ev.data)); } catch {} });
    es.addEventListener("briefing", (ev) => { try { renderBriefing(JSON.parse(ev.data)); } catch {} });
    es.addEventListener("alert", (ev) => { try { addAlert(JSON.parse(ev.data)); } catch {} });
    es.addEventListener("audit", (ev) => { try { addAudit(JSON.parse(ev.data)); } catch {} });
    es.onerror = () => {
      $("#conn").className = "conn off"; $("#conn").innerHTML = '<i class="dot"></i> 연결 끊김 · 재시도';
    };
  }
  // 음성 모듈(voice.js)이 쓰는 공개 API: 음성 명령 → 조정창 제안, 경보 훅
  window.LiveAPI = { handle: (text, actor) => handleText(text, actor || "음성"), onAlert: null };

  connect();
})();
