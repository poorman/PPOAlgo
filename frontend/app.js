// PPOAlgo VWAP Momentum Backtester
// Matches the optimizer's Rust engine 4-condition entry signal exactly.
// Data source: Massive.com API (Polygon-compatible)

const MASSIVE_API_KEY = 'vQtz66lpyexhpplKWhLL7rOXdfnClQsh';
const MASSIVE_API_URL = 'https://api.massive.com';
const API_BASE = ''; // Same-origin proxy for paper trading etc.
const STORAGE_KEY = 'ppoalgo_vwap_v2';
const ALPHA = 0.50; // Stretch threshold multiplier (hardcoded in Rust)
const GAMMA = 0.25; // Momentum threshold multiplier (hardcoded in Rust)

// ── State ──────────────────────────────────────────────────────────────────────
function getDefaultDates() {
  const today = new Date();
  const end = today.toISOString().slice(0, 10);
  const back = new Date(today);
  back.setMonth(today.getMonth() - 6);
  const start = back.toISOString().slice(0, 10);
  return { start, end };
}

const defaults = getDefaultDates();

const state = {
  config: {
    stock: 'AAPL',
    buyTriggerPct: 0.1,
    buyTime: '10:00',
    sellTriggerPct: 5,
    buyAmount: 100000,
    compound: true,
    sellEOD: true,
    start: defaults.start,
    end: defaults.end,
    capital: 100000,
  },
  trades: [],
  series: { dates: [], prices: [], portfolio: [], returns: [] },
  skipStats: { belowVwap: 0, slope: 0, stretch: 0, momentum: 0, noData: 0 },
  logs: [],
  showLogs: true,
  tradeSort: { key: 'date', direction: 'asc' },
};

let priceChart = null;
let portfolioChart = null;
let returnsChart = null;

// ── Utilities ──────────────────────────────────────────────────────────────────
const logEl = document.getElementById('log');
const tradeBody = document.getElementById('trade-body');

function log(msg) {
  const ts = new Date().toLocaleTimeString();
  const entry = `[${ts}] ${msg}`;
  state.logs.push(entry);
  const div = document.createElement('div');
  div.className = 'log-entry';
  div.textContent = entry;
  logEl.appendChild(div);
  if (document.getElementById('auto-scroll')?.checked) {
    logEl.scrollTop = logEl.scrollHeight;
  }
}

function fmtPct(v) { return `${v.toFixed(2)}%`; }
function fmtMoney(v) { return `$${v.toLocaleString(undefined, { maximumFractionDigits: 0 })}`; }
function round2(v) { return v != null ? Math.round(v * 100) / 100 : null; }

function persistConfig() {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(state.config)); } catch {}
}

function loadConfig() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) state.config = { ...state.config, ...JSON.parse(raw) };
  } catch {}
  const today = new Date().toISOString().slice(0, 10);
  if (!state.config.start || state.config.start > today) state.config.start = defaults.start;
  if (!state.config.end || state.config.end > today) state.config.end = defaults.end;
  if (state.config.end < state.config.start) {
    state.config.start = defaults.start;
    state.config.end = defaults.end;
  }
}

function readConfig() {
  state.config.stock = (document.getElementById('stock-symbol').value || 'AAPL').toUpperCase();
  state.config.buyTriggerPct = Number(document.getElementById('buy-trigger').value) || 0.1;
  state.config.buyTime = document.getElementById('buy-time').value || '10:00';
  state.config.sellTriggerPct = Number(document.getElementById('sell-trigger').value) || 5;
  state.config.buyAmount = Number(document.getElementById('buy-amount').value) || 100000;
  state.config.compound = document.getElementById('compound').checked;
  state.config.sellEOD = document.getElementById('sell-eod').checked;
  state.config.start = document.getElementById('start-date').value || defaults.start;
  state.config.end = document.getElementById('end-date').value || defaults.end;
  state.config.capital = Number(document.getElementById('capital').value) || 100000;
  persistConfig();
}

function syncForm() {
  document.getElementById('stock-symbol').value = state.config.stock;
  document.getElementById('buy-trigger').value = state.config.buyTriggerPct;
  document.getElementById('buy-time').value = state.config.buyTime;
  document.getElementById('sell-trigger').value = state.config.sellTriggerPct;
  document.getElementById('buy-amount').value = state.config.buyAmount;
  document.getElementById('compound').checked = state.config.compound;
  document.getElementById('sell-eod').checked = state.config.sellEOD;
  document.getElementById('start-date').value = state.config.start;
  document.getElementById('end-date').value = state.config.end;
  document.getElementById('capital').value = state.config.capital;
}

// ── Massive.com API ────────────────────────────────────────────────────────────
async function fetchMassive(symbol, startDate, endDate, timespan = 'day') {
  const url = `${MASSIVE_API_URL}/v2/aggs/ticker/${symbol}/range/1/${timespan}/${startDate}/${endDate}`;
  const params = new URLSearchParams({
    apiKey: MASSIVE_API_KEY,
    limit: '50000',
    sort: 'asc',
    adjusted: 'true',
  });
  const resp = await fetch(`${url}?${params}`);
  if (!resp.ok) throw new Error(`Massive API ${resp.status}: ${await resp.text().catch(() => '')}`);
  const data = await resp.json();
  return data.results || [];
}

// Fetch minute bars in monthly chunks to stay under 50K limit per request
async function fetchMinuteBarsChunked(symbol, startDate, endDate) {
  const chunks = [];
  let current = new Date(startDate + 'T00:00:00');
  const end = new Date(endDate + 'T00:00:00');

  while (current <= end) {
    const chunkStart = current.toISOString().slice(0, 10);
    const chunkEnd = new Date(Math.min(
      new Date(current.getFullYear(), current.getMonth() + 1, 0).getTime(), // End of month
      end.getTime()
    )).toISOString().slice(0, 10);

    chunks.push({ start: chunkStart, end: chunkEnd });
    current = new Date(current.getFullYear(), current.getMonth() + 1, 1);
  }

  let allBars = [];
  for (let i = 0; i < chunks.length; i++) {
    const { start, end } = chunks[i];
    log(`Fetching minute bars: ${start} to ${end} (${i + 1}/${chunks.length})...`);
    const bars = await fetchMassive(symbol, start, end, 'minute');
    allBars = allBars.concat(bars);
  }
  return allBars;
}

// ── VWAP Computation from 1-minute bars ────────────────────────────────────────
// buyTimeStr: "HH:MM" in ET (e.g. "10:00", "09:45")
function computeVWAPMetrics(minuteBars, buyTimeStr = '10:00') {
  const [buyHH, buyMM] = buyTimeStr.split(':').map(Number);
  const buyMinute = buyHH * 60 + buyMM; // e.g. 600 for 10:00
  // VWAP window: 9:30 to buyTime; early VWAP: 9:30 to 9:40
  const vwapStart = 570; // 9:30
  const earlyEnd = 580;  // 9:40

  const dayMap = new Map();

  for (const bar of minuteBars) {
    const t = new Date(bar.t);
    if (isNaN(t.getTime())) continue;

    const etStr = t.toLocaleString('en-US', { timeZone: 'America/New_York' });
    const et = new Date(etStr);
    const hh = et.getHours();
    const mm = et.getMinutes();
    const etMinute = hh * 60 + mm;

    // Only market hours 9:30-16:00 ET (570-960)
    if (etMinute < vwapStart || etMinute >= 960) continue;

    const dayKey = `${et.getFullYear()}-${String(et.getMonth() + 1).padStart(2, '0')}-${String(et.getDate()).padStart(2, '0')}`;

    let day = dayMap.get(dayKey);
    if (!day) {
      day = { bars_930_buy: [], bars_930_940: [], priceBuyTime: null };
      dayMap.set(dayKey, day);
    }

    // 9:30 to buyTime (for VWAP and opening range)
    if (etMinute >= vwapStart && etMinute < buyMinute) {
      day.bars_930_buy.push(bar);
    }
    // 9:30-9:40 ET (for early VWAP / slope)
    if (etMinute >= vwapStart && etMinute < earlyEnd) {
      day.bars_930_940.push(bar);
    }
    // Bar at buy time
    if (etMinute === buyMinute) {
      day.priceBuyTime = bar.c;
    }
  }

  const result = new Map();

  for (const [dayKey, day] of dayMap) {
    let priceBuyTime = day.priceBuyTime;
    if (priceBuyTime == null && day.bars_930_buy.length > 0) {
      priceBuyTime = day.bars_930_buy[day.bars_930_buy.length - 1].c;
    }
    if (priceBuyTime == null) continue;

    const vwap = calcVWAP(day.bars_930_buy);
    const vwap_940 = calcVWAP(day.bars_930_940);

    let or_high = -Infinity, or_low = Infinity;
    for (const b of day.bars_930_buy) {
      or_high = Math.max(or_high, b.h ?? b.c);
      or_low = Math.min(or_low, b.l ?? b.c);
    }
    if (!isFinite(or_high)) or_high = priceBuyTime;
    if (!isFinite(or_low)) or_low = priceBuyTime;

    result.set(dayKey, { priceBuyTime, vwap, vwap_940, or_high, or_low });
  }

  return result;
}

function calcVWAP(bars) {
  let tpvSum = 0, volSum = 0;
  for (const b of bars) {
    const tp = ((b.h ?? b.c) + (b.l ?? b.c) + b.c) / 3;
    const vol = b.v ?? 0;
    tpvSum += tp * vol;
    volSum += vol;
  }
  return volSum > 0 ? tpvSum / volSum : 0;
}

// ── VWAP 4-Condition Entry Signal ──────────────────────────────────────────────
function vwapEntrySignal(price_10am, vwap, vwap_940, or_high, or_low, open) {
  if (!vwap || vwap <= 0) return { ok: false, reason: 'No VWAP' };

  if (price_10am <= vwap) return { ok: false, reason: '$ < VWAP' };

  const vwap_slope = vwap_940 > 0 ? (vwap - vwap_940) / vwap_940 : 0;
  if (vwap_slope <= 0) return { ok: false, reason: `slope down (${(vwap_slope * 100).toFixed(2)}%)` };

  const or_vol = Math.max(or_high - vwap, vwap - or_low) / vwap;

  const stretch = (price_10am - vwap) / vwap;
  if (stretch >= ALPHA * or_vol) return { ok: false, reason: `stretch (${(stretch * 100).toFixed(2)}%)` };

  const momentum = open > 0 ? (price_10am - open) / open : 0;
  if (momentum <= GAMMA * or_vol) return { ok: false, reason: `low mom (${(momentum * 100).toFixed(2)}%)` };

  return { ok: true, reason: 'VWAP breakout' };
}

// ── Run Backtest ───────────────────────────────────────────────────────────────
async function runBacktest() {
  readConfig();
  const { stock, start, end, buyTriggerPct, buyTime, sellTriggerPct, buyAmount, compound, sellEOD, capital } = state.config;
  const sellPct = sellTriggerPct / 100;
  const buyTrigPct = buyTriggerPct / 100;

  const pill = document.getElementById('config-status');
  pill.textContent = 'Loading...';
  pill.className = 'status-pill warn';
  log(`Backtest: ${stock} (${start} to ${end}), buy trigger=${buyTriggerPct}%, buy@${buyTime} ET, TP=${sellTriggerPct}%, sellEOD=${sellEOD}, capital=${fmtMoney(capital)}`);

  try {
    // Fetch daily bars from Massive
    log('Fetching daily bars from Massive.com...');
    const dailyBarsRaw = await fetchMassive(stock, start, end, 'day');
    log(`Got ${dailyBarsRaw.length} daily bars.`);

    // Fetch minute bars in chunks
    const minuteBars = await fetchMinuteBarsChunked(stock, start, end);
    log(`Got ${minuteBars.length} total minute bars.`);

    if (!dailyBarsRaw.length) {
      log('No daily bars. Check symbol and date range.');
      pill.textContent = 'No data';
      pill.className = 'status-pill warn';
      return;
    }

    // Compute VWAP metrics from minute bars using configured buy time
    const vwapMap = computeVWAPMetrics(minuteBars, buyTime);
    log(`VWAP metrics computed for ${vwapMap.size} trading days (buy time: ${buyTime} ET).`);

    // Build daily bar map (Massive timestamps are in ms)
    const dailyMap = new Map();
    for (const bar of dailyBarsRaw) {
      const d = new Date(bar.t);
      if (isNaN(d.getTime())) continue;
      // Convert to ET for correct date key
      const etStr = d.toLocaleString('en-US', { timeZone: 'America/New_York' });
      const et = new Date(etStr);
      const dayKey = `${et.getFullYear()}-${String(et.getMonth() + 1).padStart(2, '0')}-${String(et.getDate()).padStart(2, '0')}`;
      dailyMap.set(dayKey, bar);
    }

    // Run VWAP backtest
    const trades = [];
    const dates = [];
    const prices = [];
    const portfolioSeries = [];
    const returnsSeries = [];
    const skipStats = { belowVwap: 0, slope: 0, stretch: 0, momentum: 0, noData: 0 };

    let equity = capital;
    let peak = equity;
    let compoundedAmount = buyAmount;

    // For carry-over positions when sellEOD is unchecked
    let openPosition = null; // { buyPrice, shares, dayKey }

    const sortedDays = Array.from(dailyMap.keys()).sort();

    for (let di = 0; di < sortedDays.length; di++) {
      const dayKey = sortedDays[di];
      const daily = dailyMap.get(dayKey);
      const vwapData = vwapMap.get(dayKey);

      const open = daily.o ?? daily.c;
      const high = daily.h ?? daily.c;
      const low = daily.l ?? daily.c;
      const close = daily.c;
      // Previous close for buy trigger check
      const prevClose = di > 0 ? (dailyMap.get(sortedDays[di - 1])?.c ?? open) : open;

      dates.push(dayKey);
      prices.push(close);

      // If we're carrying an open position from a previous day
      if (openPosition) {
        const tpPrice = openPosition.buyPrice * (1 + sellPct);
        if (high >= tpPrice) {
          // TP hit today
          const profit = openPosition.shares * (tpPrice - openPosition.buyPrice);
          const notional = openPosition.shares * tpPrice;
          equity += profit;
          peak = Math.max(peak, equity);
          if (compound) compoundedAmount = notional;

          trades.push({
            date: dayKey, open: round2(open), price_10am: vwapData ? round2(vwapData.priceBuyTime) : null,
            vwap: vwapData ? round2(vwapData.vwap) : null, high: round2(high), close: round2(close),
            action: 'sell (TP)', skip_reason: `held from ${openPosition.dayKey}`,
            buy_price: round2(openPosition.buyPrice), sell_price: round2(tpPrice),
            profit: Math.round(profit), equity: Math.round(equity),
          });
          portfolioSeries.push(equity);
          returnsSeries.push(profit);
          openPosition = null;
          continue;
        } else if (sellEOD || di === sortedDays.length - 1) {
          // Force EOD exit on last day or if sellEOD re-enabled
          const profit = openPosition.shares * (close - openPosition.buyPrice);
          const notional = openPosition.shares * close;
          equity += profit;
          peak = Math.max(peak, equity);
          if (compound) compoundedAmount = notional;

          trades.push({
            date: dayKey, open: round2(open), price_10am: vwapData ? round2(vwapData.priceBuyTime) : null,
            vwap: vwapData ? round2(vwapData.vwap) : null, high: round2(high), close: round2(close),
            action: 'sell (EOD)', skip_reason: `held from ${openPosition.dayKey}`,
            buy_price: round2(openPosition.buyPrice), sell_price: round2(close),
            profit: Math.round(profit), equity: Math.round(equity),
          });
          portfolioSeries.push(equity);
          returnsSeries.push(profit);
          openPosition = null;
          continue;
        } else {
          // Still holding — mark-to-market (unrealized)
          const unrealized = openPosition.shares * (close - openPosition.buyPrice);
          trades.push({
            date: dayKey, open: round2(open), price_10am: vwapData ? round2(vwapData.priceBuyTime) : null,
            vwap: vwapData ? round2(vwapData.vwap) : null, high: round2(high), close: round2(close),
            action: 'hold', skip_reason: `from ${openPosition.dayKey}, TP not hit`,
            buy_price: round2(openPosition.buyPrice), sell_price: null,
            profit: 0, equity: Math.round(equity + unrealized),
          });
          portfolioSeries.push(equity + unrealized);
          returnsSeries.push(0);
          continue;
        }
      }

      if (!vwapData) {
        skipStats.noData++;
        trades.push({
          date: dayKey, open: round2(open), price_10am: null, vwap: null,
          high: round2(high), close: round2(close),
          action: 'skip', skip_reason: 'No intraday data',
          buy_price: null, sell_price: null, profit: 0, equity: Math.round(equity),
        });
        portfolioSeries.push(equity);
        returnsSeries.push(0);
        continue;
      }

      const { priceBuyTime, vwap, vwap_940, or_high, or_low } = vwapData;

      // Buy trigger check: price at buy time must be >= buyTriggerPct% above previous close
      if (buyTrigPct > 0 && priceBuyTime < prevClose * (1 + buyTrigPct)) {
        skipStats.noData++; // count as filtered
        trades.push({
          date: dayKey, open: round2(open), price_10am: round2(priceBuyTime),
          vwap: round2(vwap), high: round2(high), close: round2(close),
          action: 'skip', skip_reason: `buy trigger (${fmtPct(buyTriggerPct)} above prev close)`,
          buy_price: null, sell_price: null, profit: 0, equity: Math.round(equity),
        });
        portfolioSeries.push(equity);
        returnsSeries.push(0);
        continue;
      }

      const signal = vwapEntrySignal(priceBuyTime, vwap, vwap_940, or_high, or_low, open);

      if (!signal.ok) {
        if (signal.reason.includes('VWAP') || signal.reason.includes('No')) skipStats.belowVwap++;
        else if (signal.reason.includes('slope')) skipStats.slope++;
        else if (signal.reason.includes('stretch')) skipStats.stretch++;
        else if (signal.reason.includes('mom')) skipStats.momentum++;

        trades.push({
          date: dayKey, open: round2(open), price_10am: round2(priceBuyTime),
          vwap: round2(vwap), high: round2(high), close: round2(close),
          action: 'skip', skip_reason: signal.reason,
          buy_price: null, sell_price: null, profit: 0, equity: Math.round(equity),
        });
        portfolioSeries.push(equity);
        returnsSeries.push(0);
        continue;
      }

      // Entry at buy time price
      const buyPrice = priceBuyTime;
      const investAmount = compound ? Math.min(compoundedAmount, equity) : Math.min(buyAmount, equity);
      const shares = Math.floor(investAmount / buyPrice);
      if (shares <= 0) {
        trades.push({
          date: dayKey, open: round2(open), price_10am: round2(priceBuyTime),
          vwap: round2(vwap), high: round2(high), close: round2(close),
          action: 'skip', skip_reason: 'Insufficient funds',
          buy_price: null, sell_price: null, profit: 0, equity: Math.round(equity),
        });
        portfolioSeries.push(equity);
        returnsSeries.push(0);
        continue;
      }

      // Exit logic
      const tpPrice = buyPrice * (1 + sellPct);
      if (high >= tpPrice) {
        // TP hit same day
        const profit = shares * (tpPrice - buyPrice);
        const notional = shares * tpPrice;
        equity += profit;
        peak = Math.max(peak, equity);
        if (compound) compoundedAmount = notional;

        trades.push({
          date: dayKey, open: round2(open), price_10am: round2(priceBuyTime),
          vwap: round2(vwap), high: round2(high), close: round2(close),
          action: 'buy+sell (TP)', skip_reason: '',
          buy_price: round2(buyPrice), sell_price: round2(tpPrice),
          profit: Math.round(profit), equity: Math.round(equity),
        });
        portfolioSeries.push(equity);
        returnsSeries.push(profit);
      } else if (sellEOD) {
        // EOD exit
        const profit = shares * (close - buyPrice);
        const notional = shares * close;
        equity += profit;
        peak = Math.max(peak, equity);
        if (compound) compoundedAmount = notional;

        trades.push({
          date: dayKey, open: round2(open), price_10am: round2(priceBuyTime),
          vwap: round2(vwap), high: round2(high), close: round2(close),
          action: 'buy+sell (EOD)', skip_reason: '',
          buy_price: round2(buyPrice), sell_price: round2(close),
          profit: Math.round(profit), equity: Math.round(equity),
        });
        portfolioSeries.push(equity);
        returnsSeries.push(profit);
      } else {
        // Hold overnight — carry position to next day(s)
        openPosition = { buyPrice, shares, dayKey };
        const unrealized = shares * (close - buyPrice);
        trades.push({
          date: dayKey, open: round2(open), price_10am: round2(priceBuyTime),
          vwap: round2(vwap), high: round2(high), close: round2(close),
          action: 'buy (hold)', skip_reason: 'TP not hit, holding overnight',
          buy_price: round2(buyPrice), sell_price: null,
          profit: 0, equity: Math.round(equity + unrealized),
        });
        portfolioSeries.push(equity + unrealized);
        returnsSeries.push(0);
      }
    }

    state.trades = trades;
    state.series = { dates, prices, portfolio: portfolioSeries, returns: returnsSeries };
    state.skipStats = skipStats;

    const metrics = calculateMetrics(trades, capital, portfolioSeries);
    updateMetrics(metrics, skipStats, trades);
    renderCharts();
    renderTrades();

    pill.textContent = 'Done';
    pill.className = 'status-pill ok';
    log(`Done: ${fmtPct(metrics.totalReturn)} return, ${metrics.totalTrades} trades, ${fmtPct(metrics.winRate)} win rate`);

  } catch (err) {
    log(`Backtest failed: ${err.message}`);
    pill.textContent = 'Error';
    pill.className = 'status-pill warn';
  }
}

// ── Metrics ────────────────────────────────────────────────────────────────────
function calculateMetrics(trades, capital, portfolio) {
  const executed = trades.filter(t => t.action.startsWith('buy'));
  const profits = executed.map(t => t.profit);
  const wins = profits.filter(p => p > 0);
  const losses = profits.filter(p => p < 0);

  const totalPnL = profits.reduce((a, b) => a + b, 0);
  const totalReturn = capital > 0 ? (totalPnL / capital) * 100 : 0;
  const winRate = profits.length > 0 ? (wins.length / profits.length) * 100 : 0;
  const avgWin = wins.length > 0 ? wins.reduce((a, b) => a + b, 0) / wins.length : 0;
  const avgLoss = losses.length > 0 ? Math.abs(losses.reduce((a, b) => a + b, 0) / losses.length) : 0;

  const dailyReturns = [];
  for (let i = 1; i < portfolio.length; i++) {
    if (portfolio[i - 1] > 0) dailyReturns.push((portfolio[i] - portfolio[i - 1]) / portfolio[i - 1]);
  }
  const mean = dailyReturns.length > 0 ? dailyReturns.reduce((a, b) => a + b, 0) / dailyReturns.length : 0;
  const variance = dailyReturns.length > 0 ? dailyReturns.reduce((s, r) => s + (r - mean) ** 2, 0) / dailyReturns.length : 0;
  const std = Math.sqrt(variance);
  const sharpe = std > 0 ? (mean / std) * Math.sqrt(252) : 0;

  const negReturns = dailyReturns.filter(r => r < 0);
  const downVar = negReturns.length > 0 ? negReturns.reduce((s, r) => s + r ** 2, 0) / negReturns.length : 0;
  const sortino = Math.sqrt(downVar) > 0 ? (mean / Math.sqrt(downVar)) * Math.sqrt(252) : 0;

  let maxDD = 0, pk = capital;
  for (const v of portfolio) { pk = Math.max(pk, v); maxDD = Math.min(maxDD, (v - pk) / pk * 100); }

  const finalEquity = portfolio.length > 0 ? portfolio[portfolio.length - 1] : capital;
  return { totalReturn, sharpe, sortino, maxDrawdown: maxDD, winRate, avgWin, avgLoss, totalTrades: profits.length, finalEquity };
}

function updateMetrics(m, skipStats, trades) {
  const set = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
  set('m-return', fmtPct(m.totalReturn));
  set('m-sharpe', m.sharpe.toFixed(2));
  set('m-sortino', m.sortino.toFixed(2));
  set('m-dd', fmtPct(m.maxDrawdown));
  set('m-winrate', fmtPct(m.winRate));
  set('m-avg', `${fmtMoney(m.avgWin)} / ${fmtMoney(m.avgLoss)}`);
  set('m-trades', String(m.totalTrades));
  set('m-equity', fmtMoney(m.finalEquity));

  const totalDays = trades.length;
  const entries = trades.filter(t => t.action.startsWith('buy')).length;
  set('m-days', String(totalDays));
  set('m-entries', String(entries));
  set('m-skip-vwap', String(skipStats.belowVwap));
  set('m-skip-slope', String(skipStats.slope));
  set('m-skip-stretch', String(skipStats.stretch));
  set('m-skip-momentum', String(skipStats.momentum));

  const statusEl = document.getElementById('m-status');
  if (statusEl) {
    statusEl.textContent = m.totalReturn >= 0 ? 'Profitable' : 'Losing';
    statusEl.className = m.totalReturn >= 0 ? 'ok' : 'warn';
  }
}

// ── Chart.js ───────────────────────────────────────────────────────────────────
const chartDefaults = {
  responsive: true,
  maintainAspectRatio: false,
  animation: { duration: 300 },
  plugins: {
    legend: { display: false },
    tooltip: {
      backgroundColor: '#1e293b',
      titleColor: '#e5e7eb',
      bodyColor: '#e5e7eb',
      borderColor: '#334155',
      borderWidth: 1,
      padding: 10,
      cornerRadius: 8,
    },
  },
  scales: {
    x: {
      ticks: { color: '#6b7280', maxTicksLimit: 10, font: { size: 11 } },
      grid: { color: 'rgba(255,255,255,0.05)' },
    },
    y: {
      ticks: { color: '#6b7280', font: { size: 11 } },
      grid: { color: 'rgba(255,255,255,0.05)' },
    },
  },
};

function renderCharts() {
  const { dates, prices, portfolio, returns } = state.series;
  if (!dates.length) return;

  const labels = dates.map(d => d.slice(5));

  // Collect buy/sell markers
  const buyPoints = [], sellTPPoints = [], sellEODPoints = [];
  state.trades.forEach((t, i) => {
    if (t.action.includes('TP')) {
      buyPoints.push({ x: labels[i], y: t.buy_price });
      sellTPPoints.push({ x: labels[i], y: t.sell_price });
    } else if (t.action.includes('EOD')) {
      buyPoints.push({ x: labels[i], y: t.buy_price });
      sellEODPoints.push({ x: labels[i], y: t.sell_price });
    }
  });

  // 1. Price chart
  if (priceChart) priceChart.destroy();
  priceChart = new Chart(document.getElementById('price-chart').getContext('2d'), {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Close',
          data: prices,
          borderColor: '#3b82f6',
          backgroundColor: 'rgba(59,130,246,0.08)',
          fill: true,
          borderWidth: 1.5,
          pointRadius: 0,
          tension: 0.1,
        },
        {
          label: 'Buy (entry)',
          data: buyPoints.map(p => ({ x: p.x, y: p.y })),
          type: 'scatter',
          pointBackgroundColor: '#22c55e',
          pointBorderColor: '#0b1222',
          pointBorderWidth: 1.5,
          pointRadius: 5,
          pointStyle: 'triangle',
        },
        {
          label: 'Sell (TP)',
          data: sellTPPoints.map(p => ({ x: p.x, y: p.y })),
          type: 'scatter',
          pointBackgroundColor: '#ef4444',
          pointBorderColor: '#0b1222',
          pointBorderWidth: 1.5,
          pointRadius: 5,
          pointStyle: 'rectRot',
        },
        {
          label: 'Sell (EOD)',
          data: sellEODPoints.map(p => ({ x: p.x, y: p.y })),
          type: 'scatter',
          pointBackgroundColor: '#fbbf24',
          pointBorderColor: '#0b1222',
          pointBorderWidth: 1.5,
          pointRadius: 4,
          pointStyle: 'rectRot',
        },
      ],
    },
    options: {
      ...chartDefaults,
      plugins: {
        ...chartDefaults.plugins,
        legend: {
          display: true,
          position: 'top',
          labels: { color: '#9ca3af', usePointStyle: true, padding: 12, font: { size: 11 } },
        },
      },
      scales: {
        ...chartDefaults.scales,
        y: { ...chartDefaults.scales.y, ticks: { ...chartDefaults.scales.y.ticks, callback: v => `$${v.toFixed(0)}` } },
      },
    },
  });

  // 2. Portfolio chart
  if (portfolioChart) portfolioChart.destroy();
  portfolioChart = new Chart(document.getElementById('portfolio-chart').getContext('2d'), {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'Equity',
        data: portfolio,
        borderColor: '#f97316',
        backgroundColor: 'rgba(249,115,22,0.1)',
        fill: true,
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.1,
      }],
    },
    options: {
      ...chartDefaults,
      scales: {
        ...chartDefaults.scales,
        y: { ...chartDefaults.scales.y, ticks: { ...chartDefaults.scales.y.ticks, callback: v => `$${(v / 1000).toFixed(0)}k` } },
      },
    },
  });

  // 3. Returns bar chart
  if (returnsChart) returnsChart.destroy();
  const retColors = returns.map(r => r > 0 ? 'rgba(34,197,94,0.7)' : r < 0 ? 'rgba(239,68,68,0.7)' : 'rgba(100,100,100,0.15)');
  returnsChart = new Chart(document.getElementById('returns-chart').getContext('2d'), {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'P/L',
        data: returns,
        backgroundColor: retColors,
        borderRadius: 2,
        barPercentage: 0.8,
      }],
    },
    options: {
      ...chartDefaults,
      scales: {
        ...chartDefaults.scales,
        y: { ...chartDefaults.scales.y, ticks: { ...chartDefaults.scales.y.ticks, callback: v => `$${v.toFixed(0)}` } },
      },
    },
  });
}

// ── Trade Table ────────────────────────────────────────────────────────────────
function renderTrades() {
  tradeBody.innerHTML = '';
  const trades = [...state.trades];

  if (state.tradeSort?.key) {
    const { key, direction } = state.tradeSort;
    const dir = direction === 'desc' ? -1 : 1;
    trades.sort((a, b) => {
      const va = a[key] ?? '';
      const vb = b[key] ?? '';
      if (typeof va === 'string' && typeof vb === 'string') return va.localeCompare(vb) * dir;
      return ((Number(va) || 0) - (Number(vb) || 0)) * dir;
    });
  }

  for (const t of trades) {
    const isEntry = t.action.startsWith('buy');
    const isProfit = t.profit > 0;
    const isLoss = t.profit < 0;

    let actionClass = '';
    if (isEntry && isProfit) actionClass = 'pnl-profit';
    if (isEntry && isLoss) actionClass = 'pnl-loss';

    const profitClass = isProfit ? 'pnl-profit' : isLoss ? 'pnl-loss' : '';
    const reasonClass = t.action === 'skip' ? 'skip-reason' : '';

    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${t.date}</td>
      <td>${t.open ?? ''}</td>
      <td>${t.price_10am ?? ''}</td>
      <td>${t.vwap ?? ''}</td>
      <td>${t.high ?? ''}</td>
      <td>${t.close ?? ''}</td>
      <td class="${actionClass}">${t.action}</td>
      <td class="${reasonClass}">${t.skip_reason || ''}</td>
      <td>${t.buy_price ?? ''}</td>
      <td>${t.sell_price ?? ''}</td>
      <td class="${profitClass}">${t.profit ? t.profit.toLocaleString() : ''}</td>
      <td>${t.equity ? t.equity.toLocaleString() : ''}</td>
    `;
    tradeBody.appendChild(tr);
  }
}

// ── Paper Trading ──────────────────────────────────────────────────────────────
async function addToPaperTrading() {
  const btn = document.getElementById('btn-trade-it');
  btn.disabled = true;
  btn.textContent = 'Adding...';
  try {
    const resp = await fetch(`${API_BASE}/api/paper/settings`);
    let creds = resp.ok ? await resp.json() : null;
    if (!creds?.key_id) {
      alert('Set up Alpaca credentials in Paper Trading first.');
      btn.textContent = 'Trade It'; btn.disabled = false;
      return;
    }
    await fetch(`${API_BASE}/api/paper/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        key_id: creds.key_id, secret_key: creds.secret_key,
        algorithm: 'vwap_momentum', symbol: state.config.stock,
        sell_trigger_pct: state.config.sellTriggerPct,
        buy_amount: state.config.buyAmount, compound: state.config.compound,
      }),
    });
    btn.textContent = 'Added!'; btn.classList.add('success');
    log(`Added ${state.config.stock} to paper trading.`);
  } catch (err) {
    btn.textContent = 'Error'; log(`Error: ${err.message}`);
  }
  setTimeout(() => { btn.textContent = 'Trade It'; btn.classList.remove('success'); btn.disabled = false; }, 2000);
}

// ── Export ──────────────────────────────────────────────────────────────────────
function exportResults() {
  const header = 'date,open,price_10am,vwap,high,close,action,reason,entry,exit,profit,equity';
  const rows = state.trades.map(t =>
    `${t.date},${t.open ?? ''},${t.price_10am ?? ''},${t.vwap ?? ''},${t.high ?? ''},${t.close ?? ''},${t.action},${t.skip_reason || ''},${t.buy_price ?? ''},${t.sell_price ?? ''},${t.profit},${t.equity}`
  );
  const blob = new Blob([[header, ...rows].join('\n')], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `ppoalgo-vwap-${state.config.stock}-${state.config.start}.csv`;
  a.click();
  URL.revokeObjectURL(url);
  log('Exported to CSV.');
}

// ── Init ───────────────────────────────────────────────────────────────────────
function init() {
  loadConfig();
  syncForm();

  document.getElementById('btn-run-backtest').addEventListener('click', runBacktest);
  document.getElementById('btn-run-backtest-side').addEventListener('click', runBacktest);
  document.getElementById('btn-export').addEventListener('click', exportResults);

  document.getElementById('btn-toggle-logs').addEventListener('click', () => {
    state.showLogs = !state.showLogs;
    document.getElementById('logs-panel').classList.toggle('hidden', !state.showLogs);
    document.getElementById('btn-toggle-logs').textContent = state.showLogs ? 'Hide logs' : 'Show logs';
  });

  document.getElementById('btn-clear-trades').addEventListener('click', () => {
    state.trades = []; renderTrades(); log('Table cleared.');
  });
  document.getElementById('btn-clear-logs').addEventListener('click', () => {
    state.logs = []; logEl.innerHTML = ''; log('Log cleared.');
  });

  document.querySelectorAll('table.trade-table thead th[data-sort-key]').forEach(th => {
    th.classList.add('sortable');
    th.addEventListener('click', () => {
      const key = th.getAttribute('data-sort-key');
      if (state.tradeSort.key === key) state.tradeSort.direction = state.tradeSort.direction === 'asc' ? 'desc' : 'asc';
      else { state.tradeSort.key = key; state.tradeSort.direction = 'desc'; }
      renderTrades();
    });
  });

  log('VWAP Backtester ready. Data: Massive.com API. Click Run Backtest.');
}

init();
