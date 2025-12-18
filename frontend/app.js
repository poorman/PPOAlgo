function getDefaultDates() {
  const today = new Date();
  const end = today.toISOString().slice(0, 10);
  const back = new Date(today);
  back.setFullYear(today.getFullYear() - 1);
  const start = back.toISOString().slice(0, 10);
  return { start, end };
}

const defaults = getDefaultDates();

const state = {
  config: {
    model: "ppo",
    policy: "mlp",
    checkpoint: "",
    stock: "AAPL",
    useLstm: true,
    useSentiment: false,
    steps: 200000,
    gamma: 0.99,
    clip: 0.2,
    lr: 0.0003,
    start: defaults.start,
    end: defaults.end,
    capital: 100000,
    fees: 5,
    slippage: 0.05,
    momentum: {
      buyTriggerPct: 5,
      buyTime: "09:00",
      sellTriggerPct: 9,
      buyAmount: 1000,
      compound: false,
    },
  },
  series: {
    price: [],
    positions: [],
    portfolio: [],
    drawdown: [],
    actions: [],
  },
  trades: [],
  logs: [],
  showLogs: true,
  showDesc: true,
  tradeSort: {
    key: "timestamp",
    direction: "asc",
  },
};

const logEl = document.getElementById("log");
const tradeBody = document.getElementById("trade-body");
const logsPanel = document.getElementById("logs-panel");
const descBox = document.getElementById("algo-desc");
const toggleDescBtn = document.getElementById("toggle-desc");
const chartStart = document.getElementById("chart-start");
const chartEnd = document.getElementById("chart-end");
const chartPriceStart = document.getElementById("chart-price-start");
const chartPriceEnd = document.getElementById("chart-price-end");
const API_BASE = `${window.location.protocol}//${window.location.hostname}:8010`;
const STORAGE_KEY = "ppoalgo_config_v1";
const MAX_LOOKBACK_YEARS = 5;

const metricIds = {
  return: "m-return",
  sharpe: "m-sharpe",
  sortino: "m-sortino",
  dd: "m-dd",
  winrate: "m-winrate",
  avg: "m-avg",
  comm: "m-comm",
  slip: "m-slip",
  entropy: "m-entropy",
  vloss: "m-vloss",
  ploss: "m-ploss",
  clip: "m-clip",
  status: "m-status",
};

const algorithmDescriptions = {
  ppo:
    "Proximal Policy Optimization: actor-critic RL with clipped objective to prevent destructive policy updates; balances stability and performance.",
  lstm:
    "LSTM Predictor: sequence model to forecast returns/signals; captures temporal patterns and can feed downstream decision layers.",
  baseline:
    "Baseline: heuristic/benchmark strategy used as control to compare RL or ML-driven policies.",
  momentum:
    "Momentum intraday: buys after a 5% pop before 9am, takes profit at 9% intraday; if 9% not reached, exits at end of day.",
};

function log(message) {
  const timestamp = new Date().toLocaleTimeString();
  const entry = `[${timestamp}] ${message}`;
  state.logs.push(entry);
  const div = document.createElement("div");
  div.className = "log-entry";
  div.textContent = entry;
  logEl.appendChild(div);
  if (document.getElementById("auto-scroll").checked) {
    logEl.scrollTop = logEl.scrollHeight;
  }
}

function fmtPct(value) {
  return `${value.toFixed(2)}%`;
}

function fmtMoney(value) {
  return `$${value.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
}

async function sendEvent(event_type, payload) {
  try {
    await fetch(`${API_BASE}/api/event`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ event_type, payload }),
    });
  } catch (err) {
    log(`Event send failed: ${err}`);
  }
}

async function fetchPrices(symbol, start, end, timeframe = "1Day") {
  const params = new URLSearchParams({ symbol, start, end, timeframe });
  const resp = await fetch(`${API_BASE}/api/prices?${params.toString()}`);
  if (!resp.ok) {
    throw new Error(`Price fetch failed (${resp.status})`);
  }
  return resp.json();
}

/**
 * Resample minute bars to daily bars for cleaner chart display.
 * Each daily bar uses: open from first bar, high/low from all bars, close from last bar.
 */
function resampleToDaily(bars) {
  if (!bars || bars.length === 0) return bars;

  const dailyBars = [];
  const grouped = {};

  bars.forEach((bar) => {
    const d = new Date(bar.t);
    if (isNaN(d.getTime())) return;
    // Create daily key: YYYY-MM-DD
    const dayKey = `${d.getUTCFullYear()}-${String(d.getUTCMonth() + 1).padStart(2, '0')}-${String(d.getUTCDate()).padStart(2, '0')}`;

    if (!grouped[dayKey]) {
      grouped[dayKey] = [];
    }
    grouped[dayKey].push(bar);
  });

  // Sort keys chronologically and build daily bars
  const sortedKeys = Object.keys(grouped).sort();
  sortedKeys.forEach((key) => {
    const dayBars = grouped[key];
    if (dayBars.length === 0) return;

    const open = dayBars[0].o ?? dayBars[0].c;
    const close = dayBars[dayBars.length - 1].c;
    const high = Math.max(...dayBars.map((b) => b.h ?? b.c));
    const low = Math.min(...dayBars.map((b) => b.l ?? b.c));
    const volume = dayBars.reduce((sum, b) => sum + (b.v ?? 0), 0);

    dailyBars.push({
      t: dayBars[0].t, // Use first bar's timestamp
      o: open,
      h: high,
      l: low,
      c: close,
      v: volume,
    });
  });

  return dailyBars;
}

function persistConfig() {
  try {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({ config: state.config, showDesc: state.showDesc })
    );
  } catch (err) {
    log(`Persist failed: ${err}`);
  }
}

function loadPersistedConfig() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    const data = JSON.parse(raw);
    if (data.config) {
      state.config = { ...state.config, ...data.config };
    }
    if (typeof data.showDesc === "boolean") {
      state.showDesc = data.showDesc;
    }
  } catch {
    /* ignore */
  }
}

function enforceFreshDates() {
  const today = new Date();
  const fiveYearsAgo = new Date();
  fiveYearsAgo.setFullYear(today.getFullYear() - 5);
  const parsedStart = new Date(state.config.start);
  const parsedEnd = new Date(state.config.end);
  const defaults = getDefaultDates();

  const invalidStart =
    isNaN(parsedStart.getTime()) ||
    parsedStart < fiveYearsAgo ||
    parsedStart > today;
  const invalidEnd =
    isNaN(parsedEnd.getTime()) ||
    parsedEnd < parsedStart ||
    parsedEnd > today;

  if (invalidStart) state.config.start = defaults.start;
  if (invalidEnd) state.config.end = defaults.end;
}

function normalizeDate(input, fallback) {
  const d = new Date(input);
  if (isNaN(d.getTime())) return fallback;
  return d.toISOString().slice(0, 10);
}

function clampDate(dateStr, minIso, maxIso) {
  if (!dateStr) return minIso;
  if (dateStr < minIso) return minIso;
  if (dateStr > maxIso) return maxIso;
  return dateStr;
}

function updateMomentumVisibility() {
  const isMomentum = state.config.model === "momentum";
  const mom = document.getElementById("training-momentum");
  const gen = document.getElementById("training-generic");
  mom?.classList.toggle("hidden", !isMomentum);
  gen?.classList.toggle("hidden", isMomentum);
}

function renderSparkline(targetId, values, options = {}) {
  const el = document.getElementById(targetId);
  if (!el) return;
  const width = el.clientWidth || 320;
  const height = options.height || 160;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const points = values
    .map((v, i) => {
      const x = (i / (values.length - 1 || 1)) * width;
      const y = height - ((v - min) / span) * (height - 10) - 5;
      return `${x},${y}`;
    })
    .join(" ");

  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("width", width);
  svg.setAttribute("height", height);
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);

  if (options.fill) {
    const fillPath = `0,${height} ${points} ${width},${height}`;
    const polygon = document.createElementNS(
      "http://www.w3.org/2000/svg",
      "polygon"
    );
    polygon.setAttribute("points", fillPath);
    polygon.setAttribute("fill", options.fill);
    svg.appendChild(polygon);
  }

  const path = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
  path.setAttribute("fill", "none");
  path.setAttribute("stroke", options.color || "var(--primary)");
  path.setAttribute("stroke-width", options.strokeWidth || 2);
  path.setAttribute("points", points);
  svg.appendChild(path);

  if (options.markers) {
    options.markers.forEach((marker) => {
      const idx = Math.min(marker.index, values.length - 1);
      const x = (idx / (values.length - 1 || 1)) * width;
      const y = height - ((values[idx] - min) / span) * (height - 10) - 5;
      const circle = document.createElementNS(
        "http://www.w3.org/2000/svg",
        "circle"
      );
      circle.setAttribute("cx", x);
      circle.setAttribute("cy", y);
      circle.setAttribute("r", 5);
      circle.setAttribute("fill", marker.color);
      circle.setAttribute("stroke", "#0b1222");
      circle.setAttribute("stroke-width", 1.5);
      if (marker.label) {
        const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
        title.textContent = marker.label;
        circle.appendChild(title);
      }
      svg.appendChild(circle);
    });
  }

  el.innerHTML = "";
  el.appendChild(svg);
}

function setSeriesFromBars(bars) {
  if (!bars || bars.length === 0) return;
  const price = bars.map((b) => b.c);

  // Simple markers based on price slope
  const actions = price.map((p, i) => {
    if (i === 0) return { index: i, type: "flat" };
    if (p > price[i - 1] * 1.01) return { index: i, type: "buy" };
    if (p < price[i - 1] * 0.99) return { index: i, type: "sell" };
    return { index: i, type: "reduce" };
  });

  const returns = [];
  for (let i = 1; i < price.length; i++) {
    returns.push(price[i] / price[i - 1] - 1);
  }
  let nav = state.config.capital || 100000;
  let peak = nav;
  const portfolio = [nav];
  const drawdown = [0];
  returns.forEach((r) => {
    nav = nav * (1 + r);
    peak = Math.max(peak, nav);
    portfolio.push(nav);
    drawdown.push(((nav - peak) / peak) * 100);
  });

  // Placeholder position sizing: scaled momentum of price change
  const positions = price.map((p, i) => {
    if (i === 0) return 0;
    const delta = (p - price[i - 1]) / price[i - 1];
    return Math.max(-100, Math.min(100, delta * 5000));
  });

  const times = bars.map((b) => b.t);
  state.series = { price, portfolio, positions, drawdown, actions, times };
}

function generateMomentumSeries() {
  const len = 30; // intraday slices
  const price = [];
  const positions = [];
  const portfolio = [];
  const drawdown = [];
  const actions = [];

  let p = 100;
  let nav = state.config.capital;
  let peak = nav;
  let bought = false;
  let sold = false;
  let sellIndex = null;

  for (let i = 0; i < len; i++) {
    const drift = (Math.random() - 0.3) * 1.2;
    p = Math.max(70, p * (1 + drift / 100));
    price.push(p);

    if (!bought && i < 6 && p >= price[0] * 1.05) {
      bought = true;
      actions.push({ index: i, type: "buy" });
    } else if (bought && !sold && p >= price[0] * 1.09) {
      sold = true;
      sellIndex = i;
      actions.push({ index: i, type: "sell" });
    } else if (i === len - 1 && bought && !sold) {
      sold = true;
      sellIndex = i;
      actions.push({ index: i, type: "sell" });
    } else if (bought && !sold) {
      actions.push({ index: i, type: "reduce" });
    } else {
      actions.push({ index: i, type: "flat" });
    }

    if (!bought) {
      positions.push(0);
    } else if (sold && i >= sellIndex) {
      positions.push(0);
    } else {
      positions.push(100);
    }

    const exposure = positions[i] / 100;
    nav = nav * (1 + exposure * (drift / 100));
    peak = Math.max(peak, nav);
    portfolio.push(nav);
    drawdown.push(((nav - peak) / peak) * 100);
  }

  state.series = { price, positions, portfolio, drawdown, actions };
}

function renderCharts() {
  const markerColors = {
    buy: "#22c55e",
    sell: "#ef4444",
    reduce: "#fbbf24",
    flat: "#f3f4f6",
  };

  const times = state.series.times || [];
  const prices = state.series.price || [];

  renderSparkline("price-chart", prices, {
    color: "var(--primary)",
    fill: "rgba(59,130,246,0.15)",
    markers: (state.series.actions || []).map((a) => {
      const ts = times[a.index];
      const price = prices[a.index];
      let label = "";
      if (ts && typeof price === "number") {
        const d = new Date(ts);
        const ds = isNaN(d.getTime()) ? ts : d.toISOString().slice(0, 10);
        label = `${ds} – $${price.toFixed(2)}`;
      } else if (typeof price === "number") {
        label = `$${price.toFixed(2)}`;
      }
      return {
        index: a.index,
        color: markerColors[a.type],
        label,
      };
    }),
  });

  renderSparkline("position-chart", state.series.positions, {
    color: "var(--accent)",
    fill: "rgba(34,197,94,0.12)",
  });

  renderSparkline("portfolio-chart", state.series.portfolio, {
    color: "#f97316",
    fill: "rgba(249,115,22,0.12)",
  });

  renderSparkline("dd-chart", state.series.drawdown, {
    color: "#eab308",
    fill: "rgba(234,179,8,0.12)",
  });
}

function updateMetrics(metrics) {
  const setText = (key, value) => {
    const el = document.getElementById(metricIds[key]);
    if (el) el.textContent = value;
  };

  setText("return", fmtPct(metrics.totalReturn));
  setText("sharpe", metrics.sharpe.toFixed(2));
  setText("sortino", metrics.sortino.toFixed(2));
  setText("dd", fmtPct(metrics.maxDrawdown));
  setText("winrate", fmtPct(metrics.winRate));
  setText("avg", `${fmtMoney(metrics.avgWin)} / ${fmtMoney(metrics.avgLoss)}`);
  setText("comm", fmtMoney(metrics.commission));
  setText("slip", fmtMoney(metrics.slippage));
  setText("entropy", metrics.entropy.toFixed(2));
  setText("vloss", metrics.valueLoss.toFixed(2));
  setText("ploss", metrics.policyLoss.toFixed(2));
  setText("clip", fmtPct(metrics.clipFraction * 100));

  const statusEl = document.getElementById(metricIds.status);
  if (statusEl) {
    const status = metrics.stability;
    statusEl.textContent = status;
    statusEl.className =
      status === "stable"
        ? "ok"
        : status === "improving"
          ? "ok"
          : status === "stuck"
            ? "warn"
            : "warn";
  }
}

function renderTrades() {
  tradeBody.innerHTML = "";
  const trades = [...state.trades];

  // Apply sorting if configured
  if (state.tradeSort && state.tradeSort.key) {
    const { key, direction } = state.tradeSort;
    const dir = direction === "desc" ? -1 : 1;

    const toNumber = (v) => {
      if (v === null || v === undefined || Number.isNaN(v)) return 0;
      const n = Number(v);
      return Number.isNaN(n) ? 0 : n;
    };

    trades.sort((a, b) => {
      let va = a[key];
      let vb = b[key];

      // Special handling for timestamp
      if (key === "timestamp") {
        const ta = new Date(va).getTime() || 0;
        const tb = new Date(vb).getTime() || 0;
        return (ta - tb) * dir;
      }

      // Strings sort lexicographically
      if (typeof va === "string" || typeof vb === "string") {
        va = va ?? "";
        vb = vb ?? "";
        return va.toString().localeCompare(vb.toString()) * dir;
      }

      // Numbers fallback
      return (toNumber(va) - toNumber(vb)) * dir;
    });
  }

  trades.slice(-80).forEach((t) => {
    const isSell = t.action === "sell";
    const madeProfit = t.realized > 0 || t.reward > 0;
    const madeLoss = t.realized < 0 || t.reward < 0;
    let actionClass = "";
    if (isSell && madeProfit) actionClass = "trade-sell-profit";
    if (isSell && madeLoss) actionClass = "trade-sell-loss";
    const pnlClass = t.realized > 0 ? "pnl-profit" : t.realized < 0 ? "pnl-loss" : "";

    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${t.timestamp}</td>
      <td>${t.open !== undefined ? t.open.toFixed(2) : ""}</td>
      <td>${t.price9am !== undefined ? t.price9am.toFixed(2) : ""}</td>
      <td>${t.price.toFixed(2)}</td>
      <td>${(t.pctChange ?? 0).toFixed(2)}%</td>
      <td>${t.notional !== undefined ? t.notional.toFixed(2) : ""}</td>
      <td class="${actionClass}">${t.action}</td>
      <td>${t.position}%</td>
      <td>${t.reward.toFixed(2)}</td>
      <td>${t.unreal.toFixed(2)}</td>
      <td class="${pnlClass}">${t.realized.toFixed(2)}</td>
      <td>${t.wallet !== undefined ? t.wallet.toFixed(2) : ""}</td>
    `;
    tradeBody.appendChild(tr);
  });
}

function generateTradesFromActions() {
  const trades = [];
  const times = state.series.times || [];
  const prices = state.series.price || [];
  const positions = state.series.positions || [];
  (state.series.actions || []).forEach((a) => {
    if (a.type === "flat") return;
    const idx = a.index; // Use the action's actual index, not the loop index
    const ts = times[idx] || new Date().toISOString();
    const price = prices[idx] ?? 0;
    const pos = Math.round(positions[idx] ?? 0);
    // Synthetic reward based on price change since prior point
    const prev = prices[Math.max(idx - 1, 0)] ?? price;
    const pnl = price - prev;
    const pctChange = prev ? ((price - prev) / prev) * 100 : 0;
    trades.push({
      timestamp: ts,
      price,
      pctChange: +pctChange.toFixed(2),
      notional: null,
      wallet: null,
      action: a.type,
      position: pos,
      reward: +(pnl.toFixed(2)),
      unreal: +(pnl.toFixed(2)),
      realized: pos === 0 ? +(pnl.toFixed(2)) : 0,
    });
  });
  state.trades = trades;
  updateChartPrices();
}

function formatTs(ts) {
  const d = new Date(ts);
  if (isNaN(d.getTime())) return ts;
  return d.toLocaleString("en-US", {
    timeZone: "America/Chicago",
    hour12: false,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function combineDateWithTimeCt(ts, hhmm = "09:00") {
  // Build a timestamp at given hh:mm Central on the same date, accounting roughly for DST.
  const base = new Date(ts);
  if (isNaN(base.getTime())) return ts;
  const [hh, mm] = hhmm.split(":").map((v) => parseInt(v, 10));
  const year = base.getUTCFullYear();
  const monthNum = base.getUTCMonth() + 1;
  const month = monthNum.toString().padStart(2, "0");
  const day = base.getUTCDate().toString().padStart(2, "0");
  const h = (isNaN(hh) ? 9 : hh).toString().padStart(2, "0");
  const m = (isNaN(mm) ? 0 : mm).toString().padStart(2, "0");
  // Rough DST handling: Mar-Oct use -05:00, otherwise -06:00
  const offset = monthNum >= 3 && monthNum <= 10 ? "-05:00" : "-06:00";
  const iso = `${year}-${month}-${day}T${h}:${m}:00${offset}`;
  return new Date(iso).toISOString();
}

function generateMomentumTrades(bars, actions) {
  const trades = [];
  let entryPrice = null;
  let qty = 0;
  const buyAmount = state.config.momentum.buyAmount || 0;

  // Map actions by index for quick lookup
  const actionMap = {};
  actions.forEach((a) => {
    actionMap[a.index] = a.type;
  });

  for (let i = 0; i < bars.length; i++) {
    const bar = bars[i];
    const action = actionMap[i];
    if (!action || action === "flat") continue;
    const price = bar.c;
    const prev = i > 0 ? bars[i - 1].c : price;
    const pctChange = prev ? ((price - prev) / prev) * 100 : 0;
    if (action === "buy") {
      entryPrice = price;
      qty = buyAmount > 0 ? buyAmount / price : 0;
      trades.push({
        timestamp: formatTs(bar.t),
        price,
        pctChange: +pctChange.toFixed(2),
        action,
        position: 100,
        reward: +(price - prev).toFixed(2),
        unreal: 0,
        realized: 0,
      });
    } else if (action === "sell" && entryPrice != null) {
      const realized = qty ? (price - entryPrice) * qty : price - entryPrice;
      trades.push({
        timestamp: formatTs(bar.t),
        price,
        pctChange: +pctChange.toFixed(2),
        action,
        position: 0,
        reward: +(price - prev).toFixed(2),
        unreal: 0,
        realized: +realized.toFixed(2),
      });
      entryPrice = null;
      qty = 0;
    }
  }

  return trades;
}

function applySettings() {
  const todayIso = new Date().toISOString().slice(0, 10);
  const minDate = new Date();
  minDate.setFullYear(minDate.getFullYear() - MAX_LOOKBACK_YEARS);
  const minIso = minDate.toISOString().slice(0, 10);
  let startInput = document.getElementById("start-date").value;
  let endInput = document.getElementById("end-date").value;
  let start = normalizeDate(startInput, state.config.start);
  let end = normalizeDate(endInput, state.config.end);
  start = clampDate(start, minIso, todayIso);
  end = clampDate(end, minIso, todayIso);
  if (end < start) {
    const tmp = start;
    start = end;
    end = tmp;
  }
  document.getElementById("start-date").value = start;
  document.getElementById("end-date").value = end;

  state.config.model = document.getElementById("model-select").value;
  state.config.policy = document.getElementById("ppo-policy").value;
  state.config.checkpoint = document.getElementById("ppo-checkpoint").value;
  state.config.stock = document.getElementById("stock-symbol").value || "AAPL";
  state.config.useLstm = document.getElementById("toggle-lstm").checked;
  state.config.useSentiment = document.getElementById("toggle-sentiment").checked;
  state.config.steps = Number(document.getElementById("steps-range").value);
  state.config.gamma = Number(document.getElementById("gamma-range").value);
  state.config.clip = Number(document.getElementById("clip-range").value);
  state.config.lr = Number(document.getElementById("lr-range").value);
  state.config.start = start;
  state.config.end = end;
  state.config.capital = Number(document.getElementById("capital").value);
  state.config.fees = Number(document.getElementById("fees").value);
  state.config.slippage = Number(document.getElementById("slippage").value);
  state.config.momentum.buyTriggerPct = Number(
    document.getElementById("mom-buy-trigger").value || 5
  );
  state.config.momentum.buyTime =
    document.getElementById("mom-buy-time").value || "09:00";
  state.config.momentum.sellTriggerPct = Number(
    document.getElementById("mom-sell-trigger").value || 9
  );
  state.config.momentum.buyAmount = Number(
    document.getElementById("mom-buy-amount").value || 1000
  );
  state.config.momentum.compound =
    document.getElementById("mom-compound").checked;

  const pill = document.getElementById("config-status");
  pill.classList.remove("muted");
  pill.classList.add("ok");
  pill.textContent = "Saved";
  log(
    `Settings saved (${state.config.model.toUpperCase()} | steps=${state.config.steps}, gamma=${state.config.gamma}, clip=${state.config.clip}, lr=${state.config.lr})`
  );
  updateChartDates();
  persistConfig();
  sendEvent("settings_saved", { config: state.config });
}

function updateDescription() {
  const current = document.getElementById("model-select").value;
  descBox.textContent =
    algorithmDescriptions[current] || "No description available.";
  descBox.classList.toggle("hidden", !state.showDesc);
  toggleDescBtn.textContent = state.showDesc
    ? "Hide description"
    : "Show description";
}

function updateChartDates() {
  if (chartStart && chartEnd) {
    chartStart.textContent = state.config.start;
    chartEnd.textContent = state.config.end;
  }
  updateChartPrices();
}

function updateChartPrices() {
  const prices = state.series.price || [];
  if (!prices.length) {
    chartPriceStart && (chartPriceStart.textContent = "");
    chartPriceEnd && (chartPriceEnd.textContent = "");
    return;
  }
  const startPrice = prices[0];
  const endPrice = prices[prices.length - 1];
  if (chartPriceStart && typeof startPrice === "number") {
    chartPriceStart.textContent = `$${startPrice.toFixed(2)}`;
  }
  if (chartPriceEnd && typeof endPrice === "number") {
    chartPriceEnd.textContent = `$${endPrice.toFixed(2)}`;
  }
}

function toggleDescription() {
  state.showDesc = !state.showDesc;
  updateDescription();
  persistConfig();
}

function buildSeriesFromBars(bars, withSignals) {
  if (!bars || !bars.length) {
    state.series = {
      price: [],
      positions: [],
      portfolio: [],
      drawdown: [],
      actions: [],
      times: [],
      bars: [],
    };
    return;
  }

  const price = bars.map((b) => b.c);
  const times = bars.map((b) => b.t);

  // Momentum-specific signals
  if (withSignals && state.config.model === "momentum") {
    const momentumSeries = buildMomentumSeries(bars);
    state.series = { ...momentumSeries, bars };
    return;
  }

  // Generic signals
  const actions = withSignals
    ? price.map((p, i) => {
      if (i === 0) return { index: i, type: "flat" };
      if (p > price[i - 1] * 1.01) return { index: i, type: "buy" };
      if (p < price[i - 1] * 0.99) return { index: i, type: "sell" };
      return { index: i, type: "reduce" };
    })
    : price.map((_, i) => ({ index: i, type: "flat" }));

  const returns = [];
  for (let i = 1; i < price.length; i++) {
    returns.push(price[i] / price[i - 1] - 1);
  }
  let nav = state.config.capital || 100000;
  let peak = nav;
  const portfolio = [nav];
  const drawdown = [0];
  returns.forEach((r) => {
    nav = nav * (1 + r);
    peak = Math.max(peak, nav);
    portfolio.push(nav);
    drawdown.push(((nav - peak) / peak) * 100);
  });

  // Placeholder position sizing
  const positions = price.map((p, i) => {
    if (i === 0) return 0;
    const delta = (p - price[i - 1]) / price[i - 1];
    return Math.max(-100, Math.min(100, delta * 5000));
  });

  state.series = { price, portfolio, positions, drawdown, actions, times, bars };
}

// Aggregate intraday minute bars into per-day bars with:
// - previous day's close
// - true 9:00 AM Central (10:00 AM Eastern) price for entry decisions
function aggregateMinutesToMomentumBars(minuteBars, buyTimeCt = "09:00") {
  if (!minuteBars || !minuteBars.length) return [];

  const pad = (n) => String(n).padStart(2, "0");
  const dayMap = new Map();

  minuteBars.forEach((bar) => {
    const tUtc = new Date(bar.t);
    if (Number.isNaN(tUtc.getTime())) return;

    // Convert to America/Chicago for accurate 9AM CT handling (DST-aware).
    const ctString = tUtc.toLocaleString("en-US", { timeZone: "America/Chicago" });
    const ct = new Date(ctString);
    if (Number.isNaN(ct.getTime())) return;

    const dayKey = `${ct.getFullYear()}-${pad(ct.getMonth() + 1)}-${pad(ct.getDate())}`;
    const hh = pad(ct.getHours());
    const mm = pad(ct.getMinutes());
    const hhmm = `${hh}:${mm}`;

    let rec = dayMap.get(dayKey);
    if (!rec) {
      const basePrice = bar.o ?? bar.c;
      rec = {
        dayKey,
        t: bar.t, // first timestamp of the day (UTC)
        o: basePrice,
        h: bar.h ?? bar.c,
        l: bar.l ?? bar.c,
        c: bar.c,
        v: bar.v ?? 0,
        price_9am: null,
        t_9am: null,
      };
      dayMap.set(dayKey, rec);
    } else {
      const high = bar.h ?? bar.c;
      const low = bar.l ?? bar.c;
      rec.h = Math.max(rec.h, high);
      rec.l = Math.min(rec.l, low);
      rec.c = bar.c;
      rec.v += bar.v ?? 0;
      rec.t = rec.t || bar.t;
    }

    // Capture exact 9:00 AM Central bar if present
    if (hhmm === buyTimeCt && rec.price_9am == null) {
      rec.price_9am = bar.c;
      rec.t_9am = bar.t;
    }
  });

  // Sort days chronologically and compute previous close
  const days = Array.from(dayMap.values()).sort((a, b) =>
    a.dayKey.localeCompare(b.dayKey)
  );

  let prevClose = null;
  const result = [];

  for (const day of days) {
    const bar = {
      t: day.t_9am || day.t,
      o: day.o,
      h: day.h,
      l: day.l,
      c: day.c,
      v: day.v,
      prevClose: prevClose != null ? prevClose : day.o,
      price_9am: day.price_9am, // may be null if no bar exactly at 9:00
    };
    result.push(bar);
    prevClose = day.c;
  }

  return result;
}

function buildMomentumSeries(bars) {
  const price = bars.map((b) => b.c);
  const times = bars.map((b) => b.t);
  const buyPct = state.config.momentum.buyTriggerPct / 100;
  const sellPct = state.config.momentum.sellTriggerPct / 100;
  const buyAmount = state.config.momentum.buyAmount || 0;
  const compound = state.config.momentum.compound || false;
  const capital = state.config.capital || 100000;

  const actions = [];
  const positions = Array(price.length).fill(0);
  const trades = [];

  let inPos = false;
  let entryPrice = null;
  let qty = 0;
  let cash = capital;
  let nav = capital;
  let peak = nav;
  // Track compounded amount: starts with buyAmount, grows with each profitable sell
  let compoundedAmount = buyAmount;
  let lastPrevClose = null; // Store prev close for consistent % change calculation
  const portfolio = [];
  const drawdown = [];

  for (let i = 0; i < price.length; i++) {
    const bar = bars[i];
    const open = bar.o ?? bar.c;
    const high = bar.h ?? bar.c;
    const close = bar.c;

    // Previous day's close for momentum calculation, allow pre-computed prevClose
    const prevClose =
      typeof bar.prevClose === "number" && bar.prevClose > 0
        ? bar.prevClose
        : i > 0
          ? bars[i - 1].c
          : open;

    // Reference price at configured buy time (e.g., 09:00 CT).
    // If we have an explicit 9AM price from intraday aggregation, use it.
    const buyRefPrice =
      typeof bar.price_9am === "number" && bar.price_9am > 0
        ? bar.price_9am
        : close;

    // Calculate percentage change from previous close to the price at buy time.
    const pctChangeFromPrevClose =
      prevClose > 0 ? (buyRefPrice - prevClose) / prevClose : 0;

    // Buy only if:
    // 1. Not already in position
    // 2. The price at configured buy time has moved at least buyTrigger% above PREVIOUS CLOSE
    if (!inPos && pctChangeFromPrevClose >= buyPct) {
      // Entry at the price corresponding to configured buy time (e.g., true 9AM from intraday aggregation)
      entryPrice = buyRefPrice;

      // Determine invest amount based on compound setting:
      // - Compound ON: use compoundedAmount (starts with buyAmount, grows with profits)
      // - Compound OFF: use fixed buyAmount
      // - buyAmount = 0: use all available cash
      let investAmount;
      let actionLabel;
      if (buyAmount === 0) {
        investAmount = cash;
        actionLabel = "buy (ALL)";
      } else if (compound) {
        investAmount = Math.min(compoundedAmount, cash); // Can't invest more than available
        actionLabel = "buy (C)"; // C = Compound
      } else {
        investAmount = buyAmount;
        actionLabel = "buy";
      }

      qty = investAmount / entryPrice;
      const notional = qty * entryPrice;
      cash -= notional;
      const buyTs = combineDateWithTimeCt(bar.t, state.config.momentum.buyTime || "09:00");
      inPos = true;
      lastPrevClose = prevClose; // Save for sell calculation
      actions.push({ index: i, type: "buy" });
      trades.push({
        timestamp: formatTs(buyTs),
        open: +open.toFixed(4),
        price9am: +buyRefPrice.toFixed(4),
        price: +entryPrice.toFixed(4), // More precision for display
        // Actual % move from previous close at time of entry
        pctChange:
          prevClose && prevClose > 0
            ? +(((entryPrice - prevClose) / prevClose) * 100).toFixed(2)
            : 0,
        notional: +notional.toFixed(2),
        wallet: +(cash + notional).toFixed(2),
        action: actionLabel,
        position: 100,
        // Trade P/L % at entry should be 0 – we only realize P/L on sell
        reward: 0,
        unreal: 0,
        realized: 0,
      });
    }

    // Same-day sell: target or close
    if (inPos) {
      let exitPrice = null;
      let sellTime = "15:59"; // Default: end of day
      let hitTarget = false;

      // Check if the 9% (or configured sell trigger) target was hit
      // Use entryPrice directly (not rounded) for accurate calculation
      const targetPrice = entryPrice * (1 + sellPct);
      if (high >= targetPrice) {
        exitPrice = targetPrice;
        sellTime = "10:30"; // Target hit during trading hours
        hitTarget = true;
      } else {
        // Target not hit, exit at close
        exitPrice = close;
        sellTime = "15:59";
      }

      actions.push({ index: i, type: "sell" });
      const notional = qty ? qty * exitPrice : 0;
      const realized = qty ? (exitPrice - entryPrice) * qty : exitPrice - entryPrice;
      cash += notional;

      // Update compounded amount with the proceeds from this sell
      // This becomes the next buy amount when compound is enabled
      if (compound) {
        compoundedAmount = notional;
      }

      const sellTs = combineDateWithTimeCt(bar.t, sellTime);
      // % change from PREVIOUS CLOSE (same day) - consistent with buy
      const sellPctFromPrevClose =
        lastPrevClose && lastPrevClose > 0
          ? ((exitPrice - lastPrevClose) / lastPrevClose) * 100
          : 0;
      // Trade P/L % - profit from entry to exit
      const tradePnL =
        entryPrice && entryPrice > 0
          ? ((exitPrice - entryPrice) / entryPrice) * 100
          : 0;
      trades.push({
        timestamp: formatTs(sellTs),
        open: +open.toFixed(4),
        price9am: +entryPrice.toFixed(4),
        price: +exitPrice.toFixed(4), // More precision for display
        pctChange: +sellPctFromPrevClose.toFixed(2), // % above prev close at exit
        notional: +notional.toFixed(2),
        wallet: +cash.toFixed(2),
        action: hitTarget ? "sell" : "sell (EOD)",
        position: 0,
        reward: +tradePnL.toFixed(2), // Trade P/L % (profit from entry)
        unreal: 0,
        realized: +realized.toFixed(2),
      });
      inPos = false;
      entryPrice = null;
      qty = 0;
    }

    positions[i] = inPos ? 100 : 0;
    const positionValue = inPos && qty ? qty * close : 0;
    const total = cash + positionValue;
    nav = total;
    peak = Math.max(peak, nav);
    portfolio.push(nav);
    drawdown.push(((nav - peak) / peak) * 100);
  }

  return { price, positions, portfolio, drawdown, actions, times, trades };
}

async function loadHistoricalData(withSignals = false) {
  try {
    const symbol = (state.config.stock || "AAPL").toUpperCase();
    const useMomentumIntraday =
      withSignals && state.config.model === "momentum";

    // Use intraday minute bars for momentum so we can get an accurate 9AM CT price.
    const timeframe = useMomentumIntraday ? "1Min" : "1Day";
    const rawBars = await fetchPrices(
      symbol,
      state.config.start,
      state.config.end,
      timeframe
    );
    let barCount = 0;

    if (useMomentumIntraday) {
      const momentumBars = aggregateMinutesToMomentumBars(
        rawBars,
        state.config.momentum.buyTime || "09:00"
      );
      const momentumSeries = buildMomentumSeries(momentumBars);
      state.series = momentumSeries;
      state.trades = momentumSeries.trades || [];
      barCount = momentumBars.length;
    } else {
      const bars = timeframe === "1Day" ? rawBars : resampleToDaily(rawBars);
      buildSeriesFromBars(bars, withSignals);
      if (withSignals) {
        generateTradesFromActions();
      } else {
        state.trades = [];
      }
      barCount = bars.length;
    }
    renderCharts();
    renderTrades();
    if (!barCount) {
      log(`No bars returned for ${symbol} in range ${state.config.start} -> ${state.config.end}`);
    } else {
      log(`Loaded ${barCount} bars for ${symbol}.`);
    }
  } catch (err) {
    log(`Failed to load prices: ${err}`);
    // Fallback to synthetic data
    buildSeriesFromBars([], withSignals);
    if (withSignals) {
      generateTradesFromActions();
    } else {
      state.trades = [];
    }
    renderCharts();
    renderTrades();
  }
}

function randomMetrics() {
  return {
    totalReturn: (Math.random() * 30 + 5) * (Math.random() > 0.3 ? 1 : -1),
    sharpe: Math.random() * 2 + 0.5,
    sortino: Math.random() * 2 + 0.5,
    maxDrawdown: -(Math.random() * 25),
    winRate: Math.random() * 40 + 40,
    avgWin: Math.random() * 600 + 200,
    avgLoss: Math.random() * 400 + 150,
    commission: Math.random() * 400 + 150,
    slippage: Math.random() * 250 + 80,
    entropy: Math.random() * 1.5 + 0.2,
    valueLoss: Math.random() * 1.5,
    policyLoss: Math.random() * 1.5,
    clipFraction: Math.random() * 0.25,
    stability: ["stable", "overfitting", "stuck", "improving"][
      Math.floor(Math.random() * 4)
    ],
  };
}

// Calculate real metrics from momentum trades
function calculateMomentumMetrics() {
  const trades = state.trades || [];
  const capital = state.config.capital || 100000;
  const feesBps = state.config.fees || 5;
  const slippagePct = state.config.slippage || 0.05;

  // Use SELL trades as the source of realized PnL.
  // We recompute PnL directly from notional and Trade P/L % so the
  // metrics stay correct even if the 'realized' field in the table
  // is rounded or missing.
  const sellTrades = trades.filter((t) => t.action && t.action.startsWith("sell"));

  const roundTripPnLs = sellTrades.map((sell) => {
    const notional = sell.notional || 0;
    const pct = sell.reward || 0; // Trade P/L % stored in 'reward'
    return (notional * pct) / 100;
  });

  // Win/Loss analysis
  const wins = roundTripPnLs.filter(p => p > 0);
  const losses = roundTripPnLs.filter(p => p < 0);
  const winRate = roundTripPnLs.length > 0 ? (wins.length / roundTripPnLs.length) * 100 : 0;
  const avgWin = wins.length > 0 ? wins.reduce((a, b) => a + b, 0) / wins.length : 0;
  const avgLoss = losses.length > 0 ? Math.abs(losses.reduce((a, b) => a + b, 0) / losses.length) : 0;

  // Total return
  const totalPnL = roundTripPnLs.reduce((a, b) => a + b, 0);
  const totalReturn = capital > 0 ? (totalPnL / capital) * 100 : 0;

  // Commission and slippage (estimated)
  const totalNotional = trades.reduce((sum, t) => sum + (t.notional || 0), 0);
  const commission = (totalNotional * feesBps) / 10000;
  const slippage = (totalNotional * slippagePct) / 100;

  // Calculate daily returns for Sharpe/Sortino from portfolio series
  const portfolio = state.series.portfolio || [];
  const dailyReturns = [];
  for (let i = 1; i < portfolio.length; i++) {
    if (portfolio[i - 1] !== 0) {
      dailyReturns.push((portfolio[i] - portfolio[i - 1]) / portfolio[i - 1]);
    }
  }

  // Sharpe ratio (annualized, assuming 252 trading days)
  const meanReturn = dailyReturns.length > 0 ? dailyReturns.reduce((a, b) => a + b, 0) / dailyReturns.length : 0;
  const variance = dailyReturns.length > 0
    ? dailyReturns.reduce((sum, r) => sum + Math.pow(r - meanReturn, 2), 0) / dailyReturns.length
    : 0;
  const stdDev = Math.sqrt(variance);
  const sharpe = stdDev > 0 ? (meanReturn / stdDev) * Math.sqrt(252) : 0;

  // Sortino ratio (only downside deviation)
  const negativeReturns = dailyReturns.filter(r => r < 0);
  const downsideVariance = negativeReturns.length > 0
    ? negativeReturns.reduce((sum, r) => sum + Math.pow(r, 2), 0) / negativeReturns.length
    : 0;
  const downsideStdDev = Math.sqrt(downsideVariance);
  const sortino = downsideStdDev > 0 ? (meanReturn / downsideStdDev) * Math.sqrt(252) : 0;

  // Max drawdown from drawdown series
  const drawdowns = state.series.drawdown || [];
  const maxDrawdown = drawdowns.length > 0 ? Math.min(...drawdowns) : 0;

  return {
    totalReturn,
    sharpe: Math.max(0, sharpe),
    sortino: Math.max(0, sortino),
    maxDrawdown,
    winRate,
    avgWin,
    avgLoss,
    commission,
    slippage,
    entropy: 0, // Not applicable for momentum
    valueLoss: 0,
    policyLoss: 0,
    clipFraction: 0,
    stability: winRate >= 50 ? "profitable" : "losing",
  };
}

async function runBacktest() {
  log(`Backtest started on ${state.config.stock} with ${state.config.model.toUpperCase()}...`);
  try {
    await loadHistoricalData(true);

    // Use real metrics for momentum, random for other models
    const metrics = state.config.model === "momentum"
      ? calculateMomentumMetrics()
      : randomMetrics();

    updateMetrics(metrics);
    log(
      `Backtest done on ${state.config.stock}. Return ${fmtPct(metrics.totalReturn)}, Sharpe ${metrics.sharpe.toFixed(
        2
      )}, Max DD ${fmtPct(metrics.maxDrawdown)}`
    );
    sendEvent("backtest_completed", {
      config: state.config,
      metrics,
      trades: state.trades,
    });
  } catch (err) {
    log(`Backtest failed: ${err}`);
  }
}

function exportResults() {
  const metrics = Object.entries(metricIds)
    .map(([key, id]) => `${key},${document.getElementById(id).textContent}`)
    .join("\n");
  const trades = state.trades
    .map(
      (t) =>
        `${t.timestamp},${t.price.toFixed(2)},${t.action},${t.position},${t.reward.toFixed(
          2
        )},${t.unreal.toFixed(2)},${t.realized.toFixed(2)}`
    )
    .join("\n");
  const header = "timestamp,price,action,position,reward,unrealized,realized";
  const csv = `metrics,value\n${metrics}\n\n${header}\n${trades}`;
  const blob = new Blob([csv], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "ppoalgo-results.csv";
  a.click();
  URL.revokeObjectURL(url);
  log("Results exported to CSV (simulated).");
}

function toggleLogs() {
  state.showLogs = !state.showLogs;
  logsPanel.classList.toggle("hidden", !state.showLogs);
  document.getElementById("btn-toggle-logs").textContent = state.showLogs
    ? "Hide logs"
    : "Show logs";
}

function clearLogs() {
  state.logs = [];
  logEl.innerHTML = "";
}

function clearTrades() {
  state.trades = [];
  renderTrades();
}

function updateSliderLabels() {
  document.getElementById("steps-value").textContent = `${Math.round(
    state.config.steps / 1000
  )}k`;
  document.getElementById("gamma-value").textContent = state.config.gamma.toFixed(3);
  document.getElementById("clip-value").textContent = state.config.clip.toFixed(2);
  document.getElementById("lr-value").textContent = state.config.lr.toFixed(5);
}

function attachEvents() {
  document
    .getElementById("steps-range")
    .addEventListener("input", (e) => {
      state.config.steps = Number(e.target.value);
      updateSliderLabels();
    });
  document
    .getElementById("gamma-range")
    .addEventListener("input", (e) => {
      state.config.gamma = Number(e.target.value);
      updateSliderLabels();
    });
  document
    .getElementById("clip-range")
    .addEventListener("input", (e) => {
      state.config.clip = Number(e.target.value);
      updateSliderLabels();
    });
  document.getElementById("lr-range").addEventListener("input", (e) => {
    state.config.lr = Number(e.target.value);
    updateSliderLabels();
  });

  document
    .getElementById("btn-apply-settings")
    .addEventListener("click", applySettings);
  document.getElementById("model-select").addEventListener("change", () => {
    applySettings();
    updateDescription();
    updateChartDates();
    updateMomentumVisibility();
  });
  toggleDescBtn.addEventListener("click", toggleDescription);
  document.getElementById("btn-load-data").addEventListener("click", () => {
    applySettings();
    loadHistoricalData(false);
  });
  document.getElementById("btn-run-backtest").addEventListener("click", () => {
    applySettings();
    runBacktest();
  });
  document.getElementById("btn-export").addEventListener("click", exportResults);
  document.getElementById("btn-toggle-logs").addEventListener("click", toggleLogs);
  document
    .getElementById("btn-clear-logs")
    .addEventListener("click", () => {
      clearLogs();
      log("Activity log cleared.");
    });
  document
    .getElementById("btn-clear-trades")
    .addEventListener("click", () => {
      clearTrades();
      log("Trade table cleared.");
    });
  document.getElementById("btn-focus-model").addEventListener("click", () => {
    document.getElementById("model-select").focus();
  });

  // Trade table sorting
  const tradeTable = document.querySelector("table.trade-table");
  if (tradeTable) {
    const headers = tradeTable.querySelectorAll("thead th[data-sort-key]");
    headers.forEach((th) => {
      th.classList.add("sortable");
      th.addEventListener("click", () => {
        const key = th.getAttribute("data-sort-key");
        if (!key) return;
        // Toggle sort direction if same column, otherwise default to descending (most recent / largest first)
        if (state.tradeSort.key === key) {
          state.tradeSort.direction =
            state.tradeSort.direction === "asc" ? "desc" : "asc";
        } else {
          state.tradeSort.key = key;
          state.tradeSort.direction = "desc";
        }
        renderTrades();
      });
    });
  }
}

function init() {
  loadPersistedConfig();
  enforceFreshDates();
  // Sync form fields from persisted config
  document.getElementById("model-select").value = state.config.model;
  document.getElementById("ppo-policy").value = state.config.policy;
  document.getElementById("ppo-checkpoint").value = state.config.checkpoint;
  document.getElementById("stock-symbol").value = state.config.stock;
  document.getElementById("toggle-lstm").checked = state.config.useLstm;
  document.getElementById("toggle-sentiment").checked = state.config.useSentiment;
  document.getElementById("steps-range").value = state.config.steps;
  document.getElementById("gamma-range").value = state.config.gamma;
  document.getElementById("clip-range").value = state.config.clip;
  document.getElementById("lr-range").value = state.config.lr;
  document.getElementById("start-date").value = state.config.start;
  document.getElementById("end-date").value = state.config.end;
  document.getElementById("capital").value = state.config.capital;
  document.getElementById("fees").value = state.config.fees;
  document.getElementById("slippage").value = state.config.slippage;
  document.getElementById("mom-buy-trigger").value =
    state.config.momentum.buyTriggerPct;
  document.getElementById("mom-buy-time").value = state.config.momentum.buyTime;
  document.getElementById("mom-sell-trigger").value =
    state.config.momentum.sellTriggerPct;
  document.getElementById("mom-buy-amount").value =
    state.config.momentum.buyAmount;
  document.getElementById("mom-compound").checked =
    state.config.momentum.compound || false;
  persistConfig();

  attachEvents();
  updateSliderLabels();
  updateDescription();
  updateChartDates();
  updateMomentumVisibility();
  loadHistoricalData();
  updateMetrics({
    totalReturn: 0,
    sharpe: 0,
    sortino: 0,
    maxDrawdown: 0,
    winRate: 0,
    avgWin: 0,
    avgLoss: 0,
    commission: 0,
    slippage: 0,
    entropy: 0,
    valueLoss: 0,
    policyLoss: 0,
    clipFraction: 0,
    stability: "--",
  });
  log("UI ready. Charts and trades are simulated placeholders.");
  document.getElementById("btn-toggle-logs").textContent = "Hide logs";

  // Paper trading is now on a separate page (paper.html)
}

// Add current stock to paper trading
async function addToPaperTrading() {
  const btn = document.getElementById("btn-trade-it");
  const symbol = state.config.stock.toUpperCase();
  const buyTrigger = state.config.momentum.buyTriggerPct;
  const sellTrigger = state.config.momentum.sellTriggerPct;

  // Get credentials from paper trading settings
  let credentials = null;
  try {
    const settingsResp = await fetch(`${API_BASE}/api/paper/settings`);
    if (settingsResp.ok) {
      credentials = await settingsResp.json();
    }
  } catch (err) {
    log(`Failed to get credentials: ${err.message}`);
  }

  if (!credentials || !credentials.key_id || !credentials.secret_key) {
    log("No Alpaca credentials found. Please set them up in Paper Trading first.");
    alert("Please go to Paper Trading page and set up your Alpaca credentials first.");
    return;
  }

  // Default config based on user's screenshot
  const config = {
    key_id: credentials.key_id,
    secret_key: credentials.secret_key,
    algorithm: "momentum",
    symbol: symbol,
    buy_trigger_pct: buyTrigger,
    sell_trigger_pct: sellTrigger,
    buy_amount: 10000,
    compound: true,
    entry_time: "09:00",
    exit_time: "15:55",
    duration_days: 21, // 1 month
    max_daily_loss: 0,
    max_position_size: 0,
    stop_on_loss: true,
  };

  btn.disabled = true;
  btn.textContent = "Adding...";

  try {
    const resp = await fetch(`${API_BASE}/api/paper/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    });

    if (resp.ok) {
      const data = await resp.json();
      btn.textContent = "✓ Added!";
      btn.classList.add("success");
      log(`Added ${symbol} to paper trading (Buy: ${buyTrigger}%, Sell: ${sellTrigger}%)`);

      // Reset button after 2 seconds
      setTimeout(() => {
        btn.textContent = "🚀 Trade It";
        btn.classList.remove("success");
        btn.disabled = false;
      }, 2000);
    } else {
      const err = await resp.json().catch(() => ({}));
      const msg = err.detail || err.error || "Failed to add";
      btn.textContent = "Failed";
      log(`Failed to add ${symbol}: ${msg}`);
      setTimeout(() => {
        btn.textContent = "🚀 Trade It";
        btn.disabled = false;
      }, 2000);
    }
  } catch (err) {
    btn.textContent = "Error";
    log(`Error adding ${symbol}: ${err.message}`);
    setTimeout(() => {
      btn.textContent = "🚀 Trade It";
      btn.disabled = false;
    }, 2000);
  }
}

init();
