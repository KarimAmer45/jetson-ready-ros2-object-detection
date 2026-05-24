const ui = {
  map: document.querySelector("#safetyMap"),
  connectionPill: document.querySelector("#connectionPill"),
  connectionLabel: document.querySelector("#connectionLabel"),
  safetyPill: document.querySelector("#safetyPill"),
  safetyLabel: document.querySelector("#safetyLabel"),
  gateLabel: document.querySelector("#gateLabel"),
  modeLabel: document.querySelector("#modeLabel"),
  scenarioLabel: document.querySelector("#scenarioLabel"),
  clockLabel: document.querySelector("#clockLabel"),
  desiredReadout: document.querySelector("#desiredReadout"),
  outputReadout: document.querySelector("#outputReadout"),
  speedLimit: document.querySelector("#speedLimit"),
  speedLimitValue: document.querySelector("#speedLimitValue"),
  armButton: document.querySelector("#armButton"),
  deadmanButton: document.querySelector("#deadmanButton"),
  estopButton: document.querySelector("#estopButton"),
  resetButton: document.querySelector("#resetButton"),
  commandPad: document.querySelector("#commandPad"),
  joystick: document.querySelector("#joystick"),
  interlockList: document.querySelector("#interlockList"),
  riskScore: document.querySelector("#riskScore"),
  operationMode: document.querySelector("#operationMode"),
  latencyMetric: document.querySelector("#latencyMetric"),
  batteryMetric: document.querySelector("#batteryMetric"),
  humanMetric: document.querySelector("#humanMetric"),
  speedMetric: document.querySelector("#speedMetric"),
  tiltMetric: document.querySelector("#tiltMetric"),
  tempMetric: document.querySelector("#tempMetric"),
  lastPacket: document.querySelector("#lastPacket"),
  eventLog: document.querySelector("#eventLog"),
  clearLogButton: document.querySelector("#clearLogButton"),
  simSource: document.querySelector("#simSource"),
  sourceStatus: document.querySelector("#sourceStatus"),
  bridgeUrl: document.querySelector("#bridgeUrl"),
  bridgeButton: document.querySelector("#bridgeButton"),
  bridgeStatus: document.querySelector("#bridgeStatus"),
  scenarioButtons: document.querySelectorAll(".scenario-button")
};

const ctx = ui.map.getContext("2d");

const bridge = {
  socket: null,
  status: "offline",
  commandAdvertised: false,
  lastCommandAt: 0,
  manualClose: false
};

const scenarios = {
  nominal: {
    label: "Nominal aisle",
    latency: 54,
    linkQuality: 96,
    battery: 82,
    humanDistance: 3.8,
    tilt: 2.1,
    motorTemp: 38,
    humanAngle: -0.75
  },
  crowded: {
    label: "Crowded aisle",
    latency: 83,
    linkQuality: 89,
    battery: 75,
    humanDistance: 1.55,
    tilt: 3.2,
    motorTemp: 44,
    humanAngle: 0.3
  },
  commsLoss: {
    label: "Weak uplink",
    latency: 245,
    linkQuality: 41,
    battery: 68,
    humanDistance: 3.1,
    tilt: 2.8,
    motorTemp: 43,
    humanAngle: 1.1
  },
  slope: {
    label: "Ramp crossing",
    latency: 71,
    linkQuality: 93,
    battery: 59,
    humanDistance: 2.9,
    tilt: 12.7,
    motorTemp: 51,
    humanAngle: -1.35
  },
  lowBattery: {
    label: "Return-to-base",
    latency: 66,
    linkQuality: 94,
    battery: 14,
    humanDistance: 3.3,
    tilt: 3.8,
    motorTemp: 39,
    humanAngle: 0.95
  }
};

const state = {
  armed: false,
  deadman: false,
  estopped: false,
  speedLimit: 0.55,
  scenario: "nominal",
  simSource: "browser",
  mode: "inspection",
  desired: { x: 0, y: 0 },
  output: { linear: 0, angular: 0 },
  telemetry: { ...scenarios.nominal, speed: 0 },
  lastGate: "",
  lastRisk: 0,
  packetAge: 0,
  obstacles: [
    { x: -2.9, y: -1.2, r: 0.35 },
    { x: 2.5, y: 1.8, r: 0.42 },
    { x: 1.2, y: -2.3, r: 0.3 },
    { x: -1.5, y: 2.2, r: 0.28 }
  ]
};

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function fmt(value, digits = 0) {
  return Number(value).toFixed(digits);
}

function nowLabel() {
  return new Date().toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit"
  });
}

function addEvent(message, level = "info") {
  const item = document.createElement("li");
  item.className = level;
  item.innerHTML = `<time>${nowLabel()}</time><span>${message}</span>`;
  ui.eventLog.prepend(item);

  while (ui.eventLog.children.length > 8) {
    ui.eventLog.lastElementChild.remove();
  }
}

function bridgeStatusText() {
  if (state.simSource === "browser") return "Offline";
  if (bridge.status === "connected") return "Connected";
  if (bridge.status === "connecting") return "Connecting";
  if (bridge.status === "error") return "Connection error";
  return "Offline";
}

function renderBridgeUi() {
  ui.sourceStatus.textContent = state.simSource === "gazebo" ? "Gazebo bridge" : "Browser demo";
  ui.sourceStatus.classList.toggle("connected", state.simSource === "gazebo" && bridge.status === "connected");
  ui.sourceStatus.classList.toggle("warn", state.simSource === "gazebo" && bridge.status === "connecting");
  ui.sourceStatus.classList.toggle("danger", state.simSource === "gazebo" && bridge.status === "error");

  ui.bridgeStatus.textContent = bridgeStatusText();
  ui.bridgeStatus.classList.toggle("connected", bridge.status === "connected");
  ui.bridgeStatus.classList.toggle("warn", bridge.status === "connecting");
  ui.bridgeStatus.classList.toggle("danger", bridge.status === "error");
  ui.bridgeButton.textContent = bridge.status === "connected" || bridge.status === "connecting" ? "Disconnect" : "Connect";
}

function sendRosbridge(payload) {
  if (!bridge.socket || bridge.socket.readyState !== WebSocket.OPEN) return false;
  bridge.socket.send(JSON.stringify(payload));
  return true;
}

function connectRosbridge() {
  disconnectRosbridge(false);
  state.simSource = "gazebo";
  ui.simSource.value = "gazebo";
  bridge.status = "connecting";
  bridge.commandAdvertised = false;
  renderBridgeUi();

  try {
    bridge.socket = new WebSocket(ui.bridgeUrl.value.trim());
  } catch (error) {
    bridge.status = "error";
    addEvent("Gazebo bridge URL is invalid.", "danger");
    renderBridgeUi();
    return;
  }

  bridge.socket.addEventListener("open", () => {
    bridge.status = "connected";
    bridge.commandAdvertised = true;
    sendRosbridge({ op: "advertise", topic: "/cmd_vel", type: "geometry_msgs/Twist" });
    sendRosbridge({ op: "subscribe", topic: "/tsi/telemetry", type: "std_msgs/String", throttle_rate: 100 });
    addEvent("Gazebo bridge connected.");
    renderBridgeUi();
    renderUi();
  });

  bridge.socket.addEventListener("message", handleRosbridgeMessage);

  bridge.socket.addEventListener("error", () => {
    bridge.status = "error";
    addEvent("Gazebo bridge connection failed.", "danger");
    renderBridgeUi();
  });

  bridge.socket.addEventListener("close", () => {
    if (state.simSource === "gazebo" && bridge.status !== "error" && !bridge.manualClose) {
      addEvent("Gazebo bridge disconnected.", "warn");
    }
    bridge.socket = null;
    bridge.status = bridge.status === "error" ? "error" : "offline";
    bridge.commandAdvertised = false;
    bridge.manualClose = false;
    renderBridgeUi();
  });
}

function disconnectRosbridge(announce = true) {
  if (!bridge.socket) {
    bridge.status = "offline";
    bridge.commandAdvertised = false;
    renderBridgeUi();
    return;
  }

  if (bridge.socket.readyState === WebSocket.OPEN) {
    sendRosbridge({ op: "unadvertise", topic: "/cmd_vel" });
    sendRosbridge({ op: "unsubscribe", topic: "/tsi/telemetry" });
  }

  bridge.manualClose = true;
  bridge.socket.close();
  bridge.socket = null;
  bridge.status = "offline";
  bridge.commandAdvertised = false;
  if (announce) addEvent("Gazebo bridge disconnected.");
  renderBridgeUi();
}

function readNumber(payload, keys, fallback) {
  for (const key of keys) {
    if (Number.isFinite(Number(payload[key]))) return Number(payload[key]);
  }
  return fallback;
}

function applyBridgeTelemetry(payload) {
  state.telemetry.latency = readNumber(payload, ["latencyMs", "latency"], state.telemetry.latency);
  state.telemetry.linkQuality = readNumber(payload, ["linkQuality", "link"], state.telemetry.linkQuality);
  state.telemetry.battery = readNumber(payload, ["battery", "batteryPercent"], state.telemetry.battery);
  state.telemetry.humanDistance = readNumber(payload, ["humanDistance", "nearestHuman", "range"], state.telemetry.humanDistance);
  state.telemetry.tilt = readNumber(payload, ["tilt", "tiltDeg"], state.telemetry.tilt);
  state.telemetry.motorTemp = readNumber(payload, ["motorTemp", "motorTemperature"], state.telemetry.motorTemp);
  state.telemetry.humanAngle = readNumber(payload, ["humanAngle"], state.telemetry.humanAngle);
  state.telemetry.speed = readNumber(payload, ["speed", "linearSpeed"], state.telemetry.speed);
  state.packetAge = 0;
}

function handleRosbridgeMessage(event) {
  let packet;
  try {
    packet = JSON.parse(event.data);
  } catch {
    return;
  }

  if (packet.op !== "publish" || packet.topic !== "/tsi/telemetry" || !packet.msg) return;

  try {
    const payload = typeof packet.msg.data === "string" ? JSON.parse(packet.msg.data) : packet.msg;
    applyBridgeTelemetry(payload);
    renderUi();
  } catch {
    addEvent("Gazebo telemetry packet was ignored.", "warn");
  }
}

function publishCommandOutput(force = false) {
  if (state.simSource !== "gazebo" || bridge.status !== "connected") return;

  const now = performance.now();
  if (!force && now - bridge.lastCommandAt < 100) return;
  bridge.lastCommandAt = now;

  sendRosbridge({
    op: "publish",
    topic: "/cmd_vel",
    msg: {
      linear: { x: state.output.linear, y: 0, z: 0 },
      angular: { x: 0, y: 0, z: state.output.angular }
    }
  });
}

function drift(current, target, step, noise = 0) {
  const next = current + (target - current) * step;
  return next + (Math.random() - 0.5) * noise;
}

function updateTelemetry() {
  if (state.simSource === "gazebo") return;

  const target = scenarios[state.scenario];
  state.telemetry.latency = clamp(drift(state.telemetry.latency, target.latency, 0.32, 8), 25, 320);
  state.telemetry.linkQuality = clamp(drift(state.telemetry.linkQuality, target.linkQuality, 0.28, 2.5), 10, 100);
  state.telemetry.battery = clamp(drift(state.telemetry.battery, target.battery, 0.2, 0.35), 0, 100);
  state.telemetry.humanDistance = clamp(drift(state.telemetry.humanDistance, target.humanDistance, 0.24, 0.16), 0.55, 6.5);
  state.telemetry.tilt = clamp(drift(state.telemetry.tilt, target.tilt, 0.22, 0.45), 0, 18);
  state.telemetry.motorTemp = clamp(drift(state.telemetry.motorTemp, target.motorTemp, 0.2, 0.55), 24, 86);
  state.telemetry.humanAngle = drift(state.telemetry.humanAngle, target.humanAngle, 0.18, 0.08);
  state.telemetry.speed = Math.abs(state.output.linear);
  state.packetAge = 0;
}

function getInterlocks() {
  const t = state.telemetry;
  const bridgeOffline = state.simSource === "gazebo" && bridge.status !== "connected";
  const telemetryStale = state.simSource === "gazebo" && state.packetAge > 3;

  return [
    {
      name: "Control link",
      detail: bridgeOffline ? "Rosbridge offline" : telemetryStale ? `${fmt(state.packetAge)}s since packet` : `${fmt(t.latency)} ms / ${fmt(t.linkQuality)}% quality`,
      level: bridgeOffline || telemetryStale || t.latency > 220 || t.linkQuality < 50 ? "fail" : t.latency > 150 || t.linkQuality < 70 ? "warn" : "ok"
    },
    {
      name: "Human proximity",
      detail: `${fmt(t.humanDistance, 1)} m nearest track`,
      level: t.humanDistance < 1.2 ? "fail" : t.humanDistance < 2 ? "warn" : "ok"
    },
    {
      name: "Platform attitude",
      detail: `${fmt(t.tilt, 1)} deg chassis tilt`,
      level: t.tilt > 14 ? "fail" : t.tilt > 9 ? "warn" : "ok"
    },
    {
      name: "Battery reserve",
      detail: `${fmt(t.battery)}% available`,
      level: t.battery < 10 ? "fail" : t.battery < 20 ? "warn" : "ok"
    },
    {
      name: "Motor thermal",
      detail: `${fmt(t.motorTemp)} C drive stage`,
      level: t.motorTemp > 76 ? "fail" : t.motorTemp > 62 ? "warn" : "ok"
    },
    {
      name: "Geofence",
      detail: state.mode === "handover" ? "Reduced envelope" : "Nominal envelope",
      level: state.mode === "handover" && Math.hypot(state.desired.x, state.desired.y) > 0.55 ? "warn" : "ok"
    }
  ];
}

function computeRisk(interlocks) {
  const base = interlocks.reduce((sum, lock) => {
    if (lock.level === "fail") return sum + 26;
    if (lock.level === "warn") return sum + 12;
    return sum;
  }, 0);
  const commandLoad = Math.hypot(state.desired.x, state.desired.y) * 18;
  return clamp(Math.round(base + commandLoad), 0, 99);
}

function computeGate(interlocks, risk) {
  const failCount = interlocks.filter((lock) => lock.level === "fail").length;
  const warnCount = interlocks.filter((lock) => lock.level === "warn").length;

  let label = "Live";
  let level = "ok";
  let maxSpeed = state.speedLimit;

  if (state.estopped) {
    label = "Emergency stop";
    level = "danger";
    maxSpeed = 0;
  } else if (failCount > 0) {
    label = "Safety hold";
    level = "danger";
    maxSpeed = 0;
  } else if (!state.armed) {
    label = "Standby";
    level = "neutral";
    maxSpeed = 0;
  } else if (!state.deadman) {
    label = "Output hold";
    level = "neutral";
    maxSpeed = 0;
  } else if (warnCount > 0 || risk > 45) {
    label = "Limited";
    level = "warn";
    maxSpeed = Math.min(state.speedLimit, 0.28);
  }

  const desiredMagnitude = Math.hypot(state.desired.x, state.desired.y);
  const limitedMagnitude = Math.min(desiredMagnitude * state.speedLimit, maxSpeed);
  const scale = desiredMagnitude > 0 ? limitedMagnitude / desiredMagnitude : 0;

  state.output.linear = -state.desired.y * scale;
  state.output.angular = state.desired.x * scale * 1.35;

  if (label !== state.lastGate) {
    const eventLevel = level === "danger" ? "danger" : level === "warn" ? "warn" : "info";
    addEvent(`Gate changed to ${label.toLowerCase()}.`, eventLevel);
    state.lastGate = label;
  }

  return { label, level, maxSpeed };
}

function renderInterlocks(interlocks) {
  ui.interlockList.innerHTML = "";
  interlocks.forEach((lock) => {
    const row = document.createElement("div");
    row.className = `interlock ${lock.level === "ok" ? "" : lock.level}`;
    row.innerHTML = `
      <span class="interlock-marker" aria-hidden="true"></span>
      <div>
        <strong>${lock.name}</strong>
        <span>${lock.detail}</span>
      </div>
      <small>${lock.level}</small>
    `;
    ui.interlockList.append(row);
  });
}

function setPill(pill, level) {
  pill.classList.remove("warn", "danger");
  if (level === "warn") pill.classList.add("warn");
  if (level === "danger") pill.classList.add("danger");
}

function renderTelemetry() {
  const t = state.telemetry;
  ui.latencyMetric.textContent = `${fmt(t.latency)} ms`;
  ui.batteryMetric.textContent = `${fmt(t.battery)}%`;
  ui.humanMetric.textContent = `${fmt(t.humanDistance, 1)} m`;
  ui.speedMetric.textContent = `${fmt(t.speed, 2)} m/s`;
  ui.tiltMetric.textContent = `${fmt(t.tilt, 1)} deg`;
  ui.tempMetric.textContent = `${fmt(t.motorTemp)} C`;
  ui.lastPacket.textContent = state.packetAge < 2 ? "Last packet now" : `Last packet ${fmt(state.packetAge)}s`;
}

function renderUi() {
  const interlocks = getInterlocks();
  const risk = computeRisk(interlocks);
  const gate = computeGate(interlocks, risk);
  const bridgeUnavailable = state.simSource === "gazebo" && (bridge.status !== "connected" || state.packetAge > 3);
  const connectionLevel = bridgeUnavailable || state.telemetry.latency > 220 || state.telemetry.linkQuality < 50 ? "danger" : state.telemetry.latency > 150 || state.telemetry.linkQuality < 70 ? "warn" : "ok";

  renderInterlocks(interlocks);
  renderTelemetry();
  renderBridgeUi();

  ui.connectionLabel.textContent = bridgeUnavailable ? bridge.status === "connected" ? "Telemetry stale" : "Bridge offline" : connectionLevel === "danger" ? "Link critical" : connectionLevel === "warn" ? "Link degraded" : "Connected";
  setPill(ui.connectionPill, connectionLevel);

  ui.safetyLabel.textContent = gate.label;
  ui.gateLabel.textContent = gate.label;
  setPill(ui.safetyPill, gate.level);

  ui.riskScore.textContent = risk;
  ui.riskScore.classList.toggle("warn", risk >= 35 && risk < 65);
  ui.riskScore.classList.toggle("danger", risk >= 65);

  ui.armButton.textContent = state.armed ? "Disarm" : "Arm";
  ui.armButton.classList.toggle("armed", state.armed);
  ui.deadmanButton.classList.toggle("active", state.deadman);
  ui.deadmanButton.textContent = state.deadman ? "Deadman Held" : "Hold Deadman";
  ui.speedLimitValue.textContent = `${fmt(state.speedLimit, 2)} m/s`;
  ui.modeLabel.textContent = state.mode.charAt(0).toUpperCase() + state.mode.slice(1);
  ui.scenarioLabel.textContent = scenarios[state.scenario].label;
  ui.desiredReadout.textContent = `${fmt(Math.hypot(state.desired.x, state.desired.y) * state.speedLimit, 2)} m/s`;
  ui.outputReadout.textContent = `${fmt(state.output.linear, 2)} m/s / ${fmt(state.output.angular, 2)} rad/s`;
  ui.clockLabel.textContent = nowLabel();
  publishCommandOutput();

  ui.scenarioButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.scenario === state.scenario);
  });
}

function resizeCanvas() {
  const rect = ui.map.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  ui.map.width = Math.max(1, Math.floor(rect.width * dpr));
  ui.map.height = Math.max(1, Math.floor(rect.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function drawRoundedRect(x, y, width, height, radius) {
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.lineTo(x + width - radius, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
  ctx.lineTo(x + width, y + height - radius);
  ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
  ctx.lineTo(x + radius, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
  ctx.lineTo(x, y + radius);
  ctx.quadraticCurveTo(x, y, x + radius, y);
  ctx.closePath();
}

function drawMap(time) {
  const rect = ui.map.getBoundingClientRect();
  const width = rect.width;
  const height = rect.height;
  const centerX = width / 2;
  const centerY = height / 2;
  const scale = Math.min(width, height) / 7.4;

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#eef4f0";
  ctx.fillRect(0, 0, width, height);

  ctx.strokeStyle = "rgba(31, 39, 36, 0.08)";
  ctx.lineWidth = 1;
  for (let x = centerX % scale; x < width; x += scale) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }
  for (let y = centerY % scale; y < height; y += scale) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }

  const toScreen = (point) => ({
    x: centerX + point.x * scale,
    y: centerY - point.y * scale
  });

  const fenceWidth = state.mode === "handover" ? 4.3 : 5.9;
  const fenceHeight = state.mode === "docking" ? 4.1 : 5.3;
  const fence = {
    x: centerX - (fenceWidth * scale) / 2,
    y: centerY - (fenceHeight * scale) / 2,
    width: fenceWidth * scale,
    height: fenceHeight * scale
  };

  ctx.strokeStyle = "#087f83";
  ctx.lineWidth = 2;
  ctx.setLineDash([10, 8]);
  drawRoundedRect(fence.x, fence.y, fence.width, fence.height, 14);
  ctx.stroke();
  ctx.setLineDash([]);

  state.obstacles.forEach((obstacle) => {
    const pos = toScreen(obstacle);
    ctx.fillStyle = "#7a6a56";
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, obstacle.r * scale, 0, Math.PI * 2);
    ctx.fill();
  });

  const human = {
    x: Math.cos(state.telemetry.humanAngle) * state.telemetry.humanDistance,
    y: Math.sin(state.telemetry.humanAngle) * state.telemetry.humanDistance
  };
  const humanPos = toScreen(human);
  ctx.fillStyle = state.telemetry.humanDistance < 1.2 ? "#c43b32" : state.telemetry.humanDistance < 2 ? "#b66a00" : "#315c9b";
  ctx.beginPath();
  ctx.arc(humanPos.x, humanPos.y, 13, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "white";
  ctx.fillRect(humanPos.x - 5, humanPos.y - 2, 10, 4);

  const pulse = 0.5 + Math.sin(time / 450) * 0.5;
  ctx.strokeStyle = `rgba(196, 59, 50, ${0.22 + pulse * 0.12})`;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(centerX, centerY, 1.2 * scale, 0, Math.PI * 2);
  ctx.stroke();

  ctx.strokeStyle = "rgba(182, 106, 0, 0.38)";
  ctx.beginPath();
  ctx.arc(centerX, centerY, 2.0 * scale, 0, Math.PI * 2);
  ctx.stroke();

  const desiredEnd = {
    x: centerX + state.desired.x * scale * 1.5,
    y: centerY + state.desired.y * scale * 1.5
  };
  ctx.strokeStyle = "#b66a00";
  ctx.lineWidth = 5;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(centerX, centerY);
  ctx.lineTo(desiredEnd.x, desiredEnd.y);
  ctx.stroke();
  ctx.lineCap = "butt";

  const robotWidth = 52;
  const robotHeight = 70;
  ctx.save();
  ctx.translate(centerX, centerY);
  ctx.rotate(state.output.angular * 0.28);
  ctx.fillStyle = "#1f2724";
  drawRoundedRect(-robotWidth / 2, -robotHeight / 2, robotWidth, robotHeight, 10);
  ctx.fill();
  ctx.fillStyle = "#dff2f0";
  ctx.fillRect(-18, -26, 36, 12);
  ctx.fillStyle = "#087f83";
  ctx.fillRect(-22, 18, 44, 10);
  ctx.restore();

  requestAnimationFrame(drawMap);
}

function setDesiredFromEvent(event) {
  const rect = ui.commandPad.getBoundingClientRect();
  const centerX = rect.left + rect.width / 2;
  const centerY = rect.top + rect.height / 2;
  const radius = rect.width / 2 - 18;
  const dx = event.clientX - centerX;
  const dy = event.clientY - centerY;
  const distance = Math.hypot(dx, dy);
  const limited = Math.min(distance, radius);
  const angle = Math.atan2(dy, dx);
  const x = Math.cos(angle) * limited;
  const y = Math.sin(angle) * limited;

  state.desired.x = clamp(x / radius, -1, 1);
  state.desired.y = clamp(y / radius, -1, 1);
  ui.joystick.style.left = `${50 + state.desired.x * 42}%`;
  ui.joystick.style.top = `${50 + state.desired.y * 42}%`;
  renderUi();
}

function resetDesired() {
  state.desired.x = 0;
  state.desired.y = 0;
  ui.joystick.style.left = "50%";
  ui.joystick.style.top = "50%";
  renderUi();
}

function bindEvents() {
  let padActive = false;

  ui.commandPad.addEventListener("pointerdown", (event) => {
    padActive = true;
    ui.commandPad.setPointerCapture(event.pointerId);
    setDesiredFromEvent(event);
  });

  ui.commandPad.addEventListener("pointermove", (event) => {
    if (padActive) setDesiredFromEvent(event);
  });

  ui.commandPad.addEventListener("pointerup", () => {
    padActive = false;
    resetDesired();
  });

  ui.commandPad.addEventListener("pointercancel", () => {
    padActive = false;
    resetDesired();
  });

  ui.speedLimit.addEventListener("input", () => {
    state.speedLimit = Number(ui.speedLimit.value);
    renderUi();
  });

  ui.armButton.addEventListener("click", () => {
    if (state.estopped) {
      addEvent("Arm rejected while emergency stop is active.", "danger");
      return;
    }
    state.armed = !state.armed;
    addEvent(state.armed ? "Operator armed command output." : "Operator disarmed command output.");
    renderUi();
  });

  ui.deadmanButton.addEventListener("pointerdown", (event) => {
    ui.deadmanButton.setPointerCapture(event.pointerId);
    state.deadman = true;
    renderUi();
  });

  ["pointerup", "pointerleave", "pointercancel"].forEach((name) => {
    ui.deadmanButton.addEventListener(name, () => {
      if (!state.deadman) return;
      state.deadman = false;
      renderUi();
    });
  });

  ui.estopButton.addEventListener("click", () => {
    state.estopped = true;
    state.armed = false;
    state.deadman = false;
    resetDesired();
    addEvent("Emergency stop engaged.", "danger");
    renderUi();
    publishCommandOutput(true);
  });

  ui.resetButton.addEventListener("click", () => {
    state.estopped = false;
    addEvent("Emergency stop reset.");
    renderUi();
  });

  ui.operationMode.addEventListener("change", () => {
    state.mode = ui.operationMode.value;
    addEvent(`Mode set to ${state.mode}.`);
    renderUi();
  });

  ui.simSource.addEventListener("change", () => {
    state.simSource = ui.simSource.value;
    if (state.simSource === "browser") {
      disconnectRosbridge(false);
      addEvent("Simulation source set to browser demo.");
      updateTelemetry();
    } else {
      addEvent("Simulation source set to Gazebo bridge.", "warn");
    }
    renderUi();
  });

  ui.bridgeButton.addEventListener("click", () => {
    if (bridge.status === "connected" || bridge.status === "connecting") {
      disconnectRosbridge();
      return;
    }
    connectRosbridge();
  });

  ui.scenarioButtons.forEach((button) => {
    button.addEventListener("click", () => {
      state.scenario = button.dataset.scenario;
      addEvent(`Scenario set to ${scenarios[state.scenario].label}.`);
      renderUi();
    });
  });

  ui.clearLogButton.addEventListener("click", () => {
    ui.eventLog.innerHTML = "";
  });

  window.addEventListener("resize", resizeCanvas);
}

function tick() {
  state.packetAge += 1;
  updateTelemetry();
  renderUi();
}

resizeCanvas();
bindEvents();
addEvent("Console initialized.");
renderUi();
setInterval(tick, 1000);
requestAnimationFrame(drawMap);
