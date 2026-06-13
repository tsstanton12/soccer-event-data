const video = document.querySelector("#video");
const emptyVideo = document.querySelector("#emptyVideo");
const clock = document.querySelector("#clock");
const message = document.querySelector("#formMessage");
let segments = [];
let predictions = [];
let actions = [];
let clipName = "";
let markIn = null;
let markOut = null;

const PREPARED_REVIEW = {
  clip: "117093_panorama_1st_half.mp4",
  half: 1,
  video: "review-data/117093_panorama_1st_half_review_0000_0060_h264.mp4"
};

const $ = (selector) => document.querySelector(selector);
const value = (selector) => $(selector).value;
const activeHalf = () => Number(value("#reviewHalf"));
const formatTime = (seconds) => {
  const safe = Math.max(0, Number(seconds) || 0);
  const mins = Math.floor(safe / 60).toString().padStart(2, "0");
  const secs = Math.floor(safe % 60).toString().padStart(2, "0");
  const millis = Math.floor((safe % 1) * 1000).toString().padStart(3, "0");
  return `${mins}:${secs}.${millis}`;
};
const download = (name, data, type = "application/json") => {
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([data], { type }));
  link.download = name;
  link.click();
  URL.revokeObjectURL(link.href);
};
const normalizeSegments = (input) => {
  const raw = Array.isArray(input) ? input : input.segments;
  const defaultHalf = Array.isArray(input) ? 1 : Number(input.half || 1);
  if (!Array.isArray(raw)) throw new Error("File must contain a segments array.");
  return raw.map((s, index) => ({
    id: s.id || crypto.randomUUID(),
    start: Number(s.start),
    end: Number(s.end),
    state: String(s.state || "unknown"),
    team: String(s.team || ""),
    player: String(s.player || ""),
    confidence: Number(s.confidence ?? 1),
    notes: String(s.notes || ""),
    half: Number(s.half || defaultHalf),
    source_action_id: String(s.source_action_id || ""),
    sourceIndex: index
  })).filter((s) => Number.isFinite(s.start) && Number.isFinite(s.end) && s.end > s.start)
    .sort((a, b) => a.start - b.start);
};
const labelPayload = () => ({
  schema_version: "1.0",
  clip: clipName,
  half: activeHalf(),
  created_at: new Date().toISOString(),
  segments: segments.filter((segment) => segment.half === activeHalf())
    .map(({ sourceIndex, ...segment }) => segment)
});
const actionPayload = () => ({
  schema_version: "1.0",
  match: clipName,
  source: "soccertrack_v2",
  attribution: "SoccerTrack v2, CC BY 4.0",
  actions
});

const SOCCERTRACK_LABELS = new Set([
  "PASS", "DRIVE", "HEADER", "HIGH PASS", "OUT", "CROSS", "THROW IN", "SHOT",
  "BALL PLAYER BLOCK", "PLAYER SUCCESSFUL TACKLE", "FREE KICK", "GOAL"
]);

function normalizeSoccerTrack(input) {
  const sourceActions = input.actions || input.annotations;
  if (!Array.isArray(sourceActions)) throw new Error("SoccerTrack file must contain an actions or annotations array.");
  const match = String(input.match_id || input.UrlLocal || "soccertrack-match");
  const fps = Number(input.fps || 25);
  clipName = match;
  return sourceActions.map((event, index) => {
    const label = String(event.label || "").toUpperCase();
    if (!SOCCERTRACK_LABELS.has(label)) throw new Error(`Unknown SoccerTrack action: ${event.label}`);
    const half = Number(String(event.gameTime || "").split(" - ")[0]);
    const sourceTime = Number.parseInt(event.position, 10) / 1000;
    if (![1, 2].includes(half) || !Number.isFinite(sourceTime)) throw new Error(`Invalid time at annotation ${index + 1}.`);
    // Current Drive files use a global match timeline while older documented
    // files use half-relative positions. Normalize both to the half video.
    const time = half === 2 && sourceTime >= 45 * 60 ? sourceTime - 45 * 60 : sourceTime;
    return {
      id: crypto.randomUUID(),
      half,
      time,
      frame: Math.round(time * fps),
      type: label.toLowerCase().replaceAll(" ", "_"),
      original_label: label,
      source_time: sourceTime,
      team: event.team || "",
      player: event.player_id === null || event.player_id === undefined ? "" : String(event.player_id),
      visibility: event.visibility || "unknown",
      source: "soccertrack_v2"
    };
  }).sort((a, b) => a.half - b.half || a.time - b.time);
}

function setMark(which) {
  if (!Number.isFinite(video.currentTime)) return;
  if (which === "in") markIn = video.currentTime;
  else markOut = video.currentTime;
  $("#startValue").textContent = markIn === null ? "Not set" : formatTime(markIn);
  $("#endValue").textContent = markOut === null ? "Not set" : formatTime(markOut);
  message.textContent = "";
}

function resetMarks() {
  markIn = markOut = null;
  $("#startValue").textContent = $("#endValue").textContent = "Not set";
}

function addSegment() {
  message.style.color = "var(--danger)";
  if (markIn === null || markOut === null) return message.textContent = "Mark both an in and out time.";
  if (markOut <= markIn) return message.textContent = "Out time must be after in time.";
  const state = value("#state");
  const team = value("#team").trim();
  const player = value("#player").trim();
  if (state === "controlled" && (!team || !player)) {
    return message.textContent = "Controlled possession needs a team and player ID.";
  }
  segments.push({
    id: crypto.randomUUID(), start: markIn, end: markOut, state, team, player,
    confidence: Number(value("#confidence")), notes: value("#notes").trim(),
    half: activeHalf(), source_action_id: ""
  });
  segments.sort((a, b) => a.start - b.start);
  resetMarks();
  $("#notes").value = "";
  message.textContent = "";
  renderAll();
}

function renderSegments() {
  const halfSegments = segments.filter((segment) => segment.half === activeHalf());
  const body = $("#segmentsBody");
  $("#segmentSummary").textContent = `${halfSegments.length} segment${halfSegments.length === 1 ? "" : "s"} · half ${activeHalf()}`;
  if (!halfSegments.length) {
    body.innerHTML = '<tr><td colspan="8" class="empty-row">No labels yet.</td></tr>';
    return;
  }
  body.innerHTML = halfSegments.map((s) => `<tr>
    <td data-jump="${s.start}">${formatTime(s.start)} - ${formatTime(s.end)}</td>
    <td>${(s.end - s.start).toFixed(2)}s</td><td>${s.state}</td><td>${s.team || "-"}</td>
    <td>${s.player || "-"}</td><td>${Math.round(s.confidence * 100)}%</td><td>${s.notes || "-"}</td>
    <td><button class="delete" data-delete="${s.id}">Delete</button></td>
  </tr>`).join("");
}

function renderActions() {
  const halfActions = actions.filter((action) => action.half === activeHalf());
  $("#actionSummary").textContent = `${halfActions.length} action${halfActions.length === 1 ? "" : "s"} · half ${activeHalf()}`;
  const body = $("#actionsBody");
  if (!halfActions.length) {
    body.innerHTML = '<tr><td colspan="7" class="empty-row">Import a SoccerTrack v2 BAS annotation file.</td></tr>';
    return;
  }
  body.innerHTML = halfActions.map((a) => `<tr>
    <td>${a.half}</td><td>${formatTime(a.time)}</td><td>${a.original_label}</td>
    <td>${a.team || "-"}</td><td>${a.player || "-"}</td><td>${a.visibility}</td><td>SoccerTrack v2</td>
  </tr>`).join("");
}

function derivePossessionFromActions() {
  const controlActions = new Set([
    "pass", "drive", "header", "high_pass", "cross", "shot", "throw_in",
    "free_kick", "player_successful_tackle"
  ]);
  const derived = [];
  const halfActions = actions.filter((action) => action.half === activeHalf());
  for (let i = 0; i < halfActions.length; i++) {
    const current = halfActions[i];
    if (!controlActions.has(current.type) || !current.team || !current.player) continue;
    const next = halfActions[i + 1];
    let end = next ? next.time : current.time + 2;
    end = Math.min(end, current.time + 8);
    if (end <= current.time) continue;
    derived.push({
      id: crypto.randomUUID(),
      start: current.time,
      end,
      state: "controlled",
      team: current.team,
      player: current.player,
      confidence: current.visibility === "visible" ? .75 : .5,
      notes: `Derived from ${current.original_label}; review required`,
      half: current.half,
      source_action_id: current.id
    });
  }
  segments = segments.filter((segment) => segment.half !== activeHalf())
    .concat(derived)
    .sort((a, b) => a.half - b.half || a.start - b.start);
  message.textContent = `Derived ${derived.length} conservative possession segments from ${halfActions.length} half ${activeHalf()} actions. Review before using as ground truth.`;
  message.style.color = "var(--green)";
  renderAll();
}

function completedPasses() {
  const maxGap = Number(value("#maxGap"));
  const controlled = segments.filter((s) =>
    s.half === activeHalf() && s.state === "controlled" && s.team && s.player
  );
  const passes = [];
  for (let i = 0; i < controlled.length - 1; i++) {
    const from = controlled[i];
    const to = controlled[i + 1];
    const gap = to.start - from.end;
    if (from.team === to.team && from.player !== to.player && gap >= 0 && gap <= maxGap) {
      passes.push({
        id: crypto.randomUUID(), type: "completed_pass", team: from.team,
        from_player: from.player, to_player: to.player, start: from.end, end: to.start,
        transition_seconds: gap, confidence: Math.min(from.confidence, to.confidence)
      });
    }
  }
  return passes;
}

function renderPasses() {
  const passes = completedPasses();
  $("#passes").innerHTML = passes.length ? passes.map((p) => `<div class="pass">
    <time>${formatTime(p.start)}</time><strong>${p.from_player} → ${p.to_player}</strong>
    <span>${p.team} · ${p.transition_seconds.toFixed(2)}s</span>
  </div>`).join("") : '<p class="empty-row">No completed passes found.</p>';
}

function segmentAt(list, time) {
  return list.find((s) => s.start <= time && s.end > time);
}

function renderMetrics() {
  const halfSegments = segments.filter((segment) => segment.half === activeHalf());
  const halfPredictions = predictions.filter((prediction) => prediction.half === activeHalf());
  const fields = ["state", "team", "player"];
  const values = ["-", "-", "-", "-"];
  if (halfSegments.length && halfPredictions.length) {
    const boundaries = [...new Set([...halfSegments, ...halfPredictions].flatMap((s) => [s.start, s.end]))].sort((a, b) => a - b);
    const scores = Object.fromEntries(fields.map((f) => [f, 0]));
    const totals = { state: 0, team: 0, player: 0 };
    for (let i = 0; i < boundaries.length - 1; i++) {
      const start = boundaries[i], end = boundaries[i + 1], duration = end - start;
      const truth = segmentAt(halfSegments, (start + end) / 2);
      if (!truth) continue;
      const predicted = segmentAt(halfPredictions, (start + end) / 2);
      totals.state += duration;
      if (predicted && predicted.state === truth.state) scores.state += duration;
      if (truth.state === "controlled") {
        totals.team += duration;
        totals.player += duration;
        if (predicted && predicted.team === truth.team) scores.team += duration;
        if (predicted && predicted.player === truth.player) scores.player += duration;
      }
    }
    fields.forEach((field, index) => values[index] = totals[field] ? `${Math.round(scores[field] / totals[field] * 100)}%` : "-");
    values[3] = `${totals.state.toFixed(1)}s`;
  }
  $("#metrics").querySelectorAll("strong").forEach((node, index) => node.textContent = values[index]);
}

function renderAll() {
  renderActions();
  renderSegments();
  renderPasses();
  renderMetrics();
}

$("#videoInput").addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (!file) return;
  clipName = file.name;
  video.src = URL.createObjectURL(file);
  video.style.display = "block";
  emptyVideo.style.display = "none";
});
$("#loadPreparedReview").addEventListener("click", () => {
  clipName = PREPARED_REVIEW.clip;
  $("#reviewHalf").value = String(PREPARED_REVIEW.half);
  video.src = PREPARED_REVIEW.video;
  video.style.display = "block";
  emptyVideo.style.display = "none";
  resetMarks();
  renderAll();
  message.textContent = "Prepared first-half review video loaded. Import the prepared actions file, then review the first 60 seconds.";
  message.style.color = "var(--green)";
});
video.addEventListener("timeupdate", () => clock.textContent = formatTime(video.currentTime));
$("#playPause").addEventListener("click", () => video.paused ? video.play() : video.pause());
document.querySelectorAll("[data-seek]").forEach((button) => button.addEventListener("click", () => {
  video.currentTime = Math.max(0, Math.min(video.duration || Infinity, video.currentTime + Number(button.dataset.seek)));
}));
$("#markStart").addEventListener("click", () => setMark("in"));
$("#markEnd").addEventListener("click", () => setMark("out"));
$("#addSegment").addEventListener("click", addSegment);
$("#maxGap").addEventListener("input", renderPasses);
$("#reviewHalf").addEventListener("change", () => {
  resetMarks();
  renderAll();
  message.textContent = `Reviewing half ${activeHalf()}. Open the matching panorama video.`;
  message.style.color = "var(--green)";
});
$("#segmentsBody").addEventListener("click", (event) => {
  const jump = event.target.closest("[data-jump]");
  const remove = event.target.closest("[data-delete]");
  if (jump) video.currentTime = Number(jump.dataset.jump);
  if (remove) {
    segments = segments.filter((s) => s.id !== remove.dataset.delete);
    renderAll();
  }
});
$("#exportJson").addEventListener("click", () => download(`${clipName || "clip"}.half-${activeHalf()}.possession-labels.json`, JSON.stringify(labelPayload(), null, 2)));
$("#exportPasses").addEventListener("click", () => download(`${clipName || "clip"}.half-${activeHalf()}.completed-passes.json`, JSON.stringify({
  schema_version: "1.0", clip: clipName, half: activeHalf(), events: completedPasses()
}, null, 2)));
$("#exportActions").addEventListener("click", () => download(`${clipName || "match"}.standard-actions.json`, JSON.stringify(actionPayload(), null, 2)));
$("#derivePossession").addEventListener("click", derivePossessionFromActions);
$("#loadDemo").addEventListener("click", () => {
  clipName = "demo-clip.mp4";
  actions = [
    { id: crypto.randomUUID(), half: 1, time: 0, frame: 0, type: "drive", original_label: "Drive", team: "home", player: "home_07", visibility: "visible", source: "demo" },
    { id: crypto.randomUUID(), half: 1, time: 2.5, frame: 63, type: "pass", original_label: "Pass", team: "home", player: "home_07", visibility: "visible", source: "demo" },
    { id: crypto.randomUUID(), half: 1, time: 3.1, frame: 78, type: "drive", original_label: "Drive", team: "home", player: "home_10", visibility: "visible", source: "demo" },
    { id: crypto.randomUUID(), half: 1, time: 7, frame: 175, type: "player_successful_tackle", original_label: "Player Successful Tackle", team: "away", player: "away_04", visibility: "visible", source: "demo" }
  ];
  segments = normalizeSegments([
    { start: 0, end: 2.5, state: "controlled", team: "home", player: "home_07", confidence: 1 },
    { start: 2.5, end: 3.1, state: "airborne", team: "", player: "", confidence: 1 },
    { start: 3.1, end: 6.2, state: "controlled", team: "home", player: "home_10", confidence: 1 },
    { start: 6.2, end: 7.0, state: "contested", team: "", player: "", confidence: .75 },
    { start: 7.0, end: 10.0, state: "controlled", team: "away", player: "away_04", confidence: 1 }
  ]);
  predictions = normalizeSegments([
    { start: 0, end: 2.7, state: "controlled", team: "home", player: "home_07" },
    { start: 2.7, end: 3.0, state: "airborne" },
    { start: 3.0, end: 6.4, state: "controlled", team: "home", player: "home_10" },
    { start: 6.4, end: 7.2, state: "contested" },
    { start: 7.2, end: 10.0, state: "controlled", team: "away", player: "away_04" }
  ]);
  renderAll();
  message.textContent = "Demo loaded: 5 possession segments, prediction scores, and 1 completed pass candidate.";
  message.style.color = "var(--green)";
  document.querySelector("#segmentSummary").scrollIntoView({ behavior: "smooth", block: "center" });
});

async function importFile(event, target) {
  const file = event.target.files[0];
  if (!file) return;
  try {
    const imported = normalizeSegments(JSON.parse(await file.text()));
    if (target === "labels") segments = imported;
    else predictions = imported;
    renderAll();
    message.textContent = target === "labels"
      ? `Imported ${segments.length} possession segments.`
      : `Imported ${predictions.length} prediction segments.`;
    message.style.color = "var(--green)";
  } catch (error) {
    message.textContent = `Could not import file: ${error.message}`;
    message.style.color = "var(--danger)";
  }
}
$("#labelsInput").addEventListener("change", (event) => importFile(event, "labels"));
$("#predictionsInput").addEventListener("change", (event) => importFile(event, "predictions"));
$("#soccerTrackInput").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  try {
    actions = normalizeSoccerTrack(JSON.parse(await file.text()));
    message.textContent = `Imported and standardized ${actions.length} SoccerTrack v2 ball actions.`;
    message.style.color = "var(--green)";
    renderAll();
    document.querySelector("#actionSummary").scrollIntoView({ behavior: "smooth", block: "center" });
  } catch (error) {
    message.textContent = `Could not import SoccerTrack actions: ${error.message}`;
    message.style.color = "var(--danger)";
  }
});
document.addEventListener("keydown", (event) => {
  if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName)) return;
  if (event.code === "Space") { event.preventDefault(); video.paused ? video.play() : video.pause(); }
  if (event.key.toLowerCase() === "i") setMark("in");
  if (event.key.toLowerCase() === "o") setMark("out");
  if (event.key === "Enter") addSegment();
  if (event.key === "ArrowLeft") video.currentTime = Math.max(0, video.currentTime - (event.shiftKey ? 1 : .1));
  if (event.key === "ArrowRight") video.currentTime = Math.min(video.duration || Infinity, video.currentTime + (event.shiftKey ? 1 : .1));
});
renderAll();
