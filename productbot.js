'use strict';

const path = require('path');
require('dotenv').config({ path: path.resolve(__dirname, '.env'), quiet: true });

const { Telegraf, session, Markup } = require('telegraf');
const { createClient } = require('@supabase/supabase-js');
const fetch =
  globalThis.fetch ??
  ((...a) => import('node-fetch').then(({ default: f }) => f(...a)));
const { OpenAI } = require('openai');
const { HfInference } = require('@huggingface/inference');

let _sharp = null;
try { _sharp = require('sharp'); } catch { console.warn('[img] `sharp` not installed. Using minimal fallbacks.'); }

/* -------------------- env checks -------------------- */
const REQUIRED_ENV = ['TELEGRAM_BOT_TOKEN','SUPABASE_URL','SUPABASE_KEY'];
for (const k of REQUIRED_ENV) if (!process.env[k]) console.error(`❌ Missing env: ${k}`);

const HF_KEY = process.env.HUGGING_FACE_API_KEY || '';
const DEEPAI_KEY = process.env.DEEPAI_API_KEY || '';
const IMAGE_TEXT_OVERLAY = (process.env.IMAGE_TEXT_OVERLAY || '1') === '1';

// near the other env reads
// Gemini config — SINGLE SOURCE OF TRUTH
const GEMINI_KEYS = (process.env.GEMINI_API_KEYS || process.env.GEMINI_API_KEY || '')
  .split(',')
  .map(s => s.trim())
  .filter(Boolean);

const GEMINI_TEXT_MODEL = process.env.GEMINI_TEXT_MODEL || 'gemini-2.5-pro';
const GEMINI_BASE = process.env.GEMINI_BASE || 'https://generativelanguage.googleapis.com/v1beta';


/* -------------------- config -------------------- */
const ADMIN_IDS = (process.env.ADMIN_IDS || '7057639075')
  .split(',').map(s => Number(s.trim())).filter(n => Number.isFinite(n) && n>0);

const TABLES = { products: 'products', exclusive: 'exclusive_products' };
const CATEGORIES_ALLOWED = ['OTT Accounts', 'IPTV', 'Product Key', 'Download'];

const bot = new Telegraf(process.env.TELEGRAM_BOT_TOKEN);
const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_KEY);

const groq = new OpenAI({
  apiKey: process.env.GROQ_API_KEY,
  baseURL: 'https://api.groq.com/openai/v1',
});

const hf = new HfInference(HF_KEY);
const HF_IMAGE_MODEL = 'stabilityai/stable-diffusion-xl-base-1.0';

/* -------------------- utils -------------------- */
const isAdmin = (ctx) => !!ctx?.from && ADMIN_IDS.includes(ctx.from.id);
const ok = (x) => typeof x !== 'undefined' && x !== null && x !== '';
const toStr = (v) => String(v ?? '');
const escapeMd = (v = '') => toStr(v).replace(/([_\*\[\]\(\)~`>#+\-=|{}\.!])/g, '\\$1');
const replyMD = (ctx, text, extra={}) => ctx.reply(text, { parse_mode: 'Markdown', ...extra });
function sanitizeForFilename(name = 'product') { return name.replace(/[^a-z0-9_.-]/gi, '_').substring(0, 100); }
const parsePrice = (raw) => { if (!raw) return null; const m = String(raw).match(/(\d[\d,\.]*)/); if (!m) return null; const n = parseFloat(m[1].replace(/,/g, '')); return Number.isFinite(n) ? Math.round(n) : null; };
const uniqMerge = (...arrs) => { const set = new Set(); arrs.flat().filter(Boolean).forEach(x => set.add(String(x).trim())); return Array.from(set).filter(Boolean); };
function sanitizeTextForAI(text = '') { return text.replace(/[*_`~]/g, '').replace(/\s{2,}/g, ' ').trim(); }
function safeParseFirstJsonObject(s) {
  if (!s) return null;
  s = String(s).replace(/```(?:json)?\s*([\s\S]*?)\s*```/gi, '$1').trim();
  const m = s.match(/\{[\s\S]*\}/m);
  if (!m) return null;
  try { return JSON.parse(m[0]); } catch { return null; }
}

// --- helpers ---
function hostOf(u){ try{ return new URL(u).hostname.replace(/^www\./,''); }catch{ return null; } }
function stripTLD(host){ return (host||'').split('.').slice(0,-1).join('.') || host; }

function levenshtein(a, b){
  a=String(a||''); b=String(b||'');
  const m=a.length,n=b.length; if(!m) return n; if(!n) return m;
  const dp=Array.from({length:m+1},(_,i)=>[i,...Array(n).fill(0)]);
  for(let j=1;j<=n;j++) dp[0][j]=j;
  for(let i=1;i<=m;i++){
    for(let j=1;j<=n;j++){
      const cost = a[i-1]===b[j-1]?0:1;
      dp[i][j]=Math.min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost);
    }
  }
  return dp[m][n];
}


// --- tiny sleep + retry wrapper ---
const sleep = (ms)=>new Promise(r=>setTimeout(r, ms));
async function tryWithRetries(label, fn, attempts = 2, baseDelay = 800) {
  let lastErr;
  for (let i=0;i<attempts;i++) {
    try { return await fn(i); } catch (e) {
      lastErr = e;
      console.warn(`[retry] ${label} attempt ${i+1} failed: ${e.message}`);
      await sleep(baseDelay * Math.pow(2, i)); // backoff
    }
  }
  if (lastErr) console.warn(`[retry] ${label} giving up after ${attempts} attempts: ${lastErr.message}`);
  return null;
}
const TEXT_RETRIES = Math.max(1, parseInt(process.env.TEXT_RETRIES||'2',10));
const IMAGE_RETRIES = Math.max(1, parseInt(process.env.IMAGE_RETRIES||'2',10));

// Try to guess the official domain from search results
// Try to guess the official domain from search results (uses both DDG modes)
async function pickOfficialDomainFromSearch(brandName){
  const urls = new Set();
  try {
    const a = await ddgSearchHTML(brandName, 10);
    a.forEach(u => urls.add(u));
  } catch {}
  try {
    const b = await ddgSearchLite(brandName, 10);
    b.forEach(u => urls.add(u));
  } catch {}

  const hosts = Array.from(urls).map(hostOf).filter(Boolean);
  if (!hosts.length) return null;

  const brand = String(brandName||'').toLowerCase().replace(/[^a-z0-9]+/g,'');
  let best = null, bestScore = Infinity;
  for (const h of hosts) {
    const base = stripTLD(h).toLowerCase().replace(/[^a-z0-9]+/g,'');
    const d = levenshtein(brand, base);
    if (d < bestScore) { bestScore = d; best = h; }
  }
  return bestScore <= 3 ? best : null;
}






/* ---------------- category helpers ---------------- */
function normalizeCategory(prodLike = {}, aiCategory) {
  if (CATEGORIES_ALLOWED.includes(aiCategory)) return aiCategory;
  const hay = [prodLike.name, prodLike.description, prodLike.category, prodLike.subcategory, Array.isArray(prodLike.tags) ? prodLike.tags.join(' ') : prodLike.tags].filter(Boolean).join(' ');
  const cat = CATEGORIES_ALLOWED.find(c => new RegExp(c.split(' ')[0], 'i').test(hay));
  return cat || 'Download';
}

/* ---------------- Website parsing helpers ---------------- */
const URL_RX = /(https?:\/\/[^\s)]+)|(www\.[^\s)]+)/ig;

async function fetchWebsiteRaw(url) {
  if (!url) return { html: '', text: '' };
  const normalized = url.startsWith('http') ? url : `https://${url}`;

  const toText = (html) =>
    html
      .replace(/<script[^>]*>[\s\S]*?<\/script>/gi, '')
      .replace(/<style[^>]*>[\s\S]*?<\/style>/gi, '')
      .replace(/<[^>]+>/g, ' ')
      .replace(/\s+/g, ' ')
      .trim();

  // 1) try direct
  try {
    const res = await fetch(normalized, {
      headers: {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
      },
      redirect: 'follow',
      signal: typeof AbortSignal?.timeout === 'function' ? AbortSignal.timeout(12000) : undefined,
    });
    if (res.ok) {
      const html = await res.text();
      const text = toText(html);
      if (text.length > 200) return { html, text };
    }
  } catch {}

  // 2) fallback: readable proxy (handles JS-heavy / blocked sites)
  try {
    const proxied = normalized.replace(/^https?:\/\//, '');
    const res2 = await fetch(`https://r.jina.ai/http://${proxied}`, {
      headers: { 'user-agent': 'Mozilla/5.0' },
      signal: typeof AbortSignal?.timeout === 'function' ? AbortSignal.timeout(12000) : undefined,
    });
    if (res2.ok) {
      const txt = await res2.text();
      // r.jina.ai returns already-extracted text
      if (txt && txt.length > 200) return { html: '', text: txt.slice(0, 20000) };
    }
  } catch {}

  return { html: '', text: '' };
}

// provider runner -> returns parsed JSON or null
async function getTextFromProvider(provider, systemPrompt, userPrompt) {
  if (provider === 'pollinations') {
    return await pollinationsTextJSON(systemPrompt, userPrompt, 'searchgpt'); // strongest first
  }
  if (provider === 'groq') {
    try {
      const r = await groq.chat.completions.create({
        messages: [{ role: 'system', content: systemPrompt }, { role: 'user', content: userPrompt }],
        model: 'llama3-70b-8192',
        temperature: 0.1,
        response_format: { type: 'json_object' },
      });
      const s = r.choices[0]?.message?.content || '';
      return safeParseFirstJsonObject(s) ?? (s ? JSON.parse(s) : null);
    } catch (e) {
      console.warn('[text] Groq failed:', e.message);
      return null;
    }
  }
  if (provider === 'gemini') {
    try { return await geminiTextJSON(systemPrompt, userPrompt); }
    catch (e) { console.warn('[text] Gemini failed:', e.message); return null; }
  }
  return null;
}


function extractMetaTags(html = '') {
  const pick = (prop, attr='property') => {
    const re = new RegExp(`<meta[^>]+${attr}=["']${prop}["'][^>]+content=["']([^"']+)["']`, 'i');
    return re.exec(html)?.[1] || null;
  };
  return {
    ogTitle: pick('og:title') || pick('twitter:title','name'),
    ogDesc: pick('og:description') || pick('twitter:description','name'),
    ogImage: pick('og:image:secure_url') || pick('og:image') || pick('twitter:image','name'),
  };
}

function extractJsonLdProduct(html = '') {
  const blocks = [];
  const rx = /<script[^>]+type=["']application\/ld\+json["'][^>]*>([\s\S]*?)<\/script>/gi;
  let m;
  while ((m = rx.exec(html)) !== null) {
    const raw = m[1].trim();
    try { blocks.push(JSON.parse(raw)); }
    catch {
      try {
        const cleaned = raw
          .replace(/\/\*[\s\S]*?\*\//g, '')
          .replace(/\/\/.*$/gm, '')
          .replace(/,(\s*[}\]])/g, '$1');
        blocks.push(JSON.parse(cleaned));
      } catch {}
    }
  }
  const flat = blocks.flatMap(b => Array.isArray(b) ? b : [b]);
  const productNode = flat.find(n => {
    const t = (n['@type'] || n.type || '');
    return (Array.isArray(t) ? t : [t]).some(x => String(x).toLowerCase() === 'product');
  });
  if (!productNode) return null;

  const offers = Array.isArray(productNode.offers) ? productNode.offers[0] : productNode.offers || {};
  const priceNum = parsePrice(offers.price || offers.priceSpecification?.price);
  const validity = offers.availabilityEnds || offers.validThrough || productNode.validThrough || 'unknown';

  const features = [];
  if (Array.isArray(productNode.additionalProperty)) {
    for (const p of productNode.additionalProperty) if (p?.name && p?.value) features.push(`${p.name}: ${p.value}`);
  }
  if (Array.isArray(productNode.featureList)) features.push(...productNode.featureList.filter(Boolean));

  return { name: productNode.name || null, description: productNode.description || null, price: priceNum || null, validity, features };
}

function bestValue(a, b) { return ok(a) ? a : (ok(b) ? b : null); }

/* --------------- Core Helpers (Telegram, Supabase) --------------- */
async function tgFileUrl(fileId) {
  try { const f = await bot.telegram.getFile(fileId); const token = process.env.TELEGRAM_BOT_TOKEN; return `https://api.telegram.org/file/bot${token}/${f.file_path}`; }
  catch { const link = await bot.telegram.getFileLink(fileId); return typeof link === 'string' ? link : link.toString(); }
}

async function rehostToSupabase(fileUrlOrBuffer, filenameHint = 'image.jpg', table) {
  const buf = Buffer.isBuffer(fileUrlOrBuffer)
    ? fileUrlOrBuffer
    : await (async () => {
        const res = await fetch(fileUrlOrBuffer, {
          signal: typeof AbortSignal?.timeout==='function' ? AbortSignal.timeout(15000) : undefined,
        });
        if (!res.ok) throw new Error(`Fetch failed: ${res.status}`);
        const ab = await res.arrayBuffer();
        return Buffer.from(ab);
      })();

  const bucket = table === TABLES.products
    ? (process.env.SUPABASE_BUCKET_PRODUCTS || 'images')
    : (process.env.SUPABASE_BUCKET_EXCLUSIVE || 'exclusiveproduct-images');

  const folder = table === TABLES.products ? 'products' : 'exclusive-products';
  const key = `${folder}/${Date.now()}-${Math.random().toString(36).slice(2)}-${sanitizeForFilename(filenameHint)}`;

  console.log(`[upload] Uploading ${filenameHint} to Supabase bucket ${bucket} at path ${key}`);
  const { error: upErr } = await supabase.storage.from(bucket).upload(key, buf, { upsert: true });
  if (upErr) throw upErr;
  const { data: pub } = supabase.storage.from(bucket).getPublicUrl(key);
  return pub.publicUrl;
}

async function ensureHostedInSupabase(u, table, filenameHint='prod.jpg') {
  if (!u || !/^https?:\/\//i.test(u)) return u;
  return rehostToSupabase(u, filenameHint, table);
}

/* -------------------- AI enrichment -------------------- */
function shortBrandName(prod) {
  const commonWords = ['premium','pro','plus','subscription','subs','account','license','key','activation','fan','mega','plan','tier','access','year','years','month','months','day','days','lifetime','annual','basic','standard','advanced','creator','business','enterprise','personal','family','student','individual'];
  const regex = new RegExp(`\\b(${commonWords.join('|')})\\b`, 'ig');
  let name = String(prod?.name || 'Product').trim().split(/[-–—(]/)[0];
  name = name.replace(regex, '');
  name = name.replace(/\b\d+\b/g, '');
  name = name.replace(/\s+/g, ' ').trim();
  return name || prod?.name || 'Product';
}

async function enrichWithAI(textHints = '', websiteContent = '', providerOrderParam = null) {
  const cleanTextHints = sanitizeTextForAI(textHints);

  const guessedName = cleanTextHints.split('\n')[0].slice(0, 120);
  const planGuess = (cleanTextHints.match(/plan[:\-]?\s*([^\n]+)/i)?.[1] || '').slice(0, 80);

  // gather evidence
  const webBundle = await searchWebForProduct(guessedName, planGuess);
  const combinedSite = sanitizeTextForAI(`${websiteContent}\n\n${webBundle}`).slice(0, 16000);

  const systemPrompt =
    'You MUST output ONLY one JSON object with EXACT keys: {"name":"string","plan":"string|unknown","validity":"string|unknown","price":"number|unknown","description":"string","tags":["string"],"category":"string","subcategory":"string|unknown","features":["string"]}';

  const userPrompt = `User text:
"""${cleanTextHints}"""

Trusted sources (use for description & features; do NOT invent):
"""${combinedSite}"""

Rules:
1) Prefer user's explicit name/plan/validity/price if present.
2) Description: 1–3 factual sentences taken from the sources.
3) Features: 4–6 short factual bullets taken from the sources.
4) Category must be one of: ${CATEGORIES_ALLOWED.join(' | ')}.
5) If some field is unknown, use "unknown". Return JSON only.`;

  // resolve provider order: explicit arg -> session -> env -> default
  const providerOrder =
    (Array.isArray(providerOrderParam) && providerOrderParam.length && providerOrderParam) ||
    (Array.isArray(this?.ctx?.session?.textOrder) && this.ctx.session.textOrder.length && this.ctx.session.textOrder) ||
    (process.env.TEXT_PROVIDER_ORDER
      ? process.env.TEXT_PROVIDER_ORDER.split(',').map(s=>s.trim()).filter(Boolean)
      : ['pollinations','groq','gemini']);
  console.log('[text] providerOrder resolved to:', providerOrder);

  // 🔧 CALL THE PROVIDERS (this was missing)
  const { json: got, provider } = await runTextProvidersWithOrder(providerOrder, systemPrompt, userPrompt);
  let json = got;

  // minimal fallback
  if (!json) {
    console.error('[text] All providers failed. Using minimal extraction.');
    const name = cleanTextHints.split('\n')[0].trim() || 'Product';
    return {
      name,
      plan: 'unknown',
      validity: 'unknown',
      price: parsePrice(cleanTextHints),
      description: name,
      tags: [],
      features: [],
      category: normalizeCategory({ name, description: cleanTextHints }),
    };
  }

  // normalize
  json.name = json.name || guessedName || 'Product';
  json.plan = json.plan || planGuess || 'unknown';
  json.validity = json.validity || 'unknown';
  json.price = parsePrice(json.price || textHints);
  if (!Array.isArray(json.tags)) json.tags = (json.tags ? String(json.tags) : '').split(/[;,]/).map(s=>s.trim()).filter(Boolean);
  if (!Array.isArray(json.features)) json.features = [];

  // second pass if thin
  const needDetail = (!json.description || json.description.length < 150 || json.features.length < 3);
  if (needDetail && combinedSite.length > 400) {
    try {
      const detailPrompt = `From ONLY the following sources, write:
A) A concise, factual 2–3 sentence description of "${json.name}" (${json.plan}).
B) 5 short factual bullet features.

Sources:
"""${combinedSite.slice(0, 8000)}"""`;
      const more = await groq.chat.completions.create({
        messages: [{ role: 'user', content: detailPrompt }],
        model: 'llama3-70b-8192',
        temperature: 0.2
      });
      const txt = more.choices[0]?.message?.content || '';
      const lines = txt.split('\n').map(s=>s.trim()).filter(Boolean);
      const bullets = lines.filter(l=>/^[-•]/.test(l)).map(l=>l.replace(/^[-•]\s?/, '').slice(0,140));
      const desc = lines.filter(l=>!/^[-•]/.test(l)).join(' ').slice(0, 600);
      if ((!json.description || json.description.length < 120) && desc.length) json.description = desc;
      if (json.features.length < 3 && bullets.length) json.features = bullets.slice(0,5);
    } catch (e) { console.warn('[text] detail pass failed:', e.message); }
  }

  json.category = normalizeCategory({ ...json, description: textHints }, json.category);
  json.subcategory = json.subcategory || 'unknown';

  console.log(`[text] filled by ${provider} — evidence ${combinedSite.length} chars — name="${json.name}" plan="${json.plan}" descChars=${(json.description||'').length} feats=${json.features.length}`);
  return json;
}


async function runTextProvidersWithOrder(order, systemPrompt, userPrompt) {
  for (const provider of order) {
    console.log('[text] trying provider:', provider);
    const json = await tryWithRetries(
      `text:${provider}`,
      async () => {
        const j = await getTextFromProvider(provider, systemPrompt, userPrompt);
        if (!j) throw new Error('no-json');
        return j;
      },
      TEXT_RETRIES
    );
    if (json) return { json, provider };
  }
  return { json: null, provider: null };
}



/* ===================== IMAGE GENERATION: Providers + Flow ===================== */

/* ---- 0) simple gradient fallback (never fails) ---- */
function gradientBackgroundSVG(width = 1024, height = 1024) {
  const svg = `
  <svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stop-color="#111827"/>
        <stop offset="50%" stop-color="#1f2937"/>
        <stop offset="100%" stop-color="#374151"/>
      </linearGradient>
    </defs>
    <rect width="100%" height="100%" fill="url(#g)"/>
  </svg>`;
  return Buffer.from(svg);
}

/* ---- compose overlay text on background ---- */
const escXML = (s='') => String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
async function composeTextOverBackground(backgroundBuffer, prod) {
  if (!_sharp) return backgroundBuffer; // no sharp, just return background
  const productName = shortBrandName(prod);
  const planName = prod.plan || '';
  function wrapText(text, maxWidth, maxLines) {
    const words = text.split(' ');
    let lines = [];
    let current = words[0] || '';
    for (let i=1;i<words.length;i++) {
      if (current.length + words[i].length + 1 < maxWidth) current += ' ' + words[i];
      else { lines.push(current); current = words[i]; }
    }
    lines.push(current);
    return lines.slice(0, maxLines);
  }
  const titleLines = wrapText(productName, 18, 2);
  const titleSize = titleLines.length > 1 ? 100 : 120;
const titleSvg = titleLines
  .map((line, idx) => `<tspan x="512" dy="${idx===0?0:'1.2em'}">${escXML(line)}</tspan>`)
  .join('');

const textSvg = `<svg width="1024" height="512" viewBox="0 0 1024 512" xmlns="http://www.w3.org/2000/svg">
  <text x="512" y="256" text-anchor="middle"
        font-family="Arial, Helvetica, DejaVu Sans, sans-serif"
        font-size="${titleSize}" font-weight="700" fill="#FFFFFF">${titleSvg}</text>
  ${planName ? `<text x="512" y="400" text-anchor="middle"
        font-family="Arial, Helvetica, DejaVu Sans, sans-serif"
        font-size="60" font-weight="500" fill="#E5E7EB">${escXML(planName)}</text>` : ''}
</svg>`;

  const textBuffer = await _sharp(Buffer.from(textSvg)).png().toBuffer();
  return _sharp(backgroundBuffer)
    .composite([
      { input: Buffer.from('<svg width="1024" height="1024"><rect width="1024" height="1024" fill="black" opacity="0.4"/></svg>'), blend: 'multiply' },
      { input: textBuffer, gravity: 'center' }
    ])
    .png()
    .toBuffer();
}

/* ---- 1) Pollinations (hosted, free) ---- */
async function generateImageFromPollinations(prompt) {
  try {
    const url = `https://image.pollinations.ai/prompt/${encodeURIComponent(prompt)}`;
    const res = await fetch(url, { signal: typeof AbortSignal?.timeout==='function' ? AbortSignal.timeout(20000) : undefined });
    if (!res.ok) throw new Error(`Pollinations HTTP ${res.status}`);
    const ab = await res.arrayBuffer();
    return Buffer.from(ab);
  } catch (e) {
    console.warn('[img] Pollinations failed:', e.message);
    return null;
  }
}

/* ---- 2) Hugging Face (existing) ---- */
async function generateImageFromHuggingFace(prompt) {
  if (!HF_KEY) { console.warn('[img] HF key missing'); return null; }
  try {
    const result = await hf.textToImage({
      model: HF_IMAGE_MODEL,
      inputs: prompt,
      parameters: {
        negative_prompt: 'blurry, ugly, deformed, noisy, plain, boring, text, watermark, signature',
        guidance_scale: 7.5,
        num_inference_steps: 28,
        width: 1024,
        height: 1024,
      },
    });
    return Buffer.from(await result.arrayBuffer());
  } catch (e) {
    const status = e?.httpResponse?.status;
    const msg = e?.httpResponse?.body?.error || e.message;
    console.warn(`[img] HF failed (${status||'??'}): ${msg}`);
    return null;
  }
}

/* ---- 3) DeepAI (hosted fallback) ---- */
async function generateImageFromDeepAI(prompt) {
  if (!DEEPAI_KEY) { console.warn('[img] DeepAI key missing'); return null; }
  try {
    const res = await fetch('https://api.deepai.org/api/text2img', {
      method: 'POST',
      headers: { 'Api-Key': DEEPAI_KEY, 'Accept':'application/json' },
      body: new URLSearchParams({ text: prompt }),
      signal: typeof AbortSignal?.timeout==='function' ? AbortSignal.timeout(30000) : undefined,
    });
    if (!res.ok) throw new Error(`DeepAI HTTP ${res.status}`);
    const data = await res.json();
    if (!data?.output_url) throw new Error('DeepAI no output_url');
    const imgRes = await fetch(data.output_url);
    if (!imgRes.ok) throw new Error(`DeepAI image fetch ${imgRes.status}`);
    const ab = await imgRes.arrayBuffer();
    return Buffer.from(ab);
  } catch (e) {
    console.warn('[img] DeepAI failed:', e.message);
    return null;
  }
}

/* ---- helpers for brand OG/logo/search before AI ---- */
async function getOG(url) {
  try {
    const res = await fetch(url, { signal: typeof AbortSignal?.timeout==='function' ? AbortSignal.timeout(8000) : undefined });
    if (!res.ok) return {};
    const html = await res.text();
    const pick = (prop) => new RegExp(`<meta[^>]+property=["']${prop}["'][^>]+content=["']([^"']+)["']`, 'i').exec(html)?.[1];
    return { image: pick('og:image:secure_url') || pick('og:image') };
  } catch { return {}; }
}
function resolveBrandDomain(name = '') {
  if (!name) return null;
  const cleanName = shortBrandName({ name });
  const n = cleanName.toLowerCase().trim();
  const map = {
    v0: 'v0.dev', gamma: 'gamma.app', spotify: 'spotify.com', netflix: 'netflix.com', youtube: 'youtube.com',
    crunchyroll: 'crunchyroll.com', elevenlabs: 'elevenlabs.io', coursera: 'coursera.org', scribd: 'scribd.com',
    skillshare: 'skillshare.com', kittl: 'kittl.com', perplexity: 'perplexity.ai'
  };
  for (const k of Object.keys(map)) if (n.includes(k)) return map[k];
  const slug = n.replace(/[^a-z0-9]/g, '');
  return slug.length >= 3 ? `${slug}.com` : null;
}
async function findBestImageWithSearch(query) {
  try {
    console.log(`[img] DuckDuckGo image search for "${query}"`);
    const url = `https://duckduckgo.com/?q=${encodeURIComponent(query)}&t=h_&iax=images&ia=images`;
    const res = await fetch(url, {
      headers: { 'User-Agent': 'Mozilla/5.0' },
      signal: typeof AbortSignal?.timeout==='function' ? AbortSignal.timeout(10000) : undefined,
    });
    if (!res.ok) return null;
    const html = await res.text();
    const regex = /"image":"(https?:\/\/[^"]+)"/g;
    let m; const urls = [];
    while ((m = regex.exec(html)) !== null) urls.push(m[1]);
    const best = urls.find(u => u.length > 50 && !u.includes('data:image'));
    return best || null;
  } catch (e) { console.warn('[img] image search failed:', e.message); return null; }
}

/* ---- non-AI image attempts (OG/logo/search). returns URL or null ---- */
async function tryBrandImages(prod, table) {
  const domain = resolveBrandDomain(prod.name);
  if (domain) {
    const fullUrl = (domain.startsWith('http') ? '' : 'https://') + domain;
    console.log(`[img] Checking OG image from ${fullUrl}`);
    try {
      const og = await getOG(fullUrl);
      if (og.image) {
        console.log(`[img] Found OG image: ${og.image}. Rehosting...`);
        return await rehostToSupabase(og.image, `${prod.name}.jpg`, table);
      }
    } catch (e) {
      console.warn(`[img] OG fetch failed: ${e.message}`);
    }

    console.log(`[img] Trying Brandfetch for ${domain}`);
    try {
      const res = await fetch(`https://api.brandfetch.io/v2/logo/${domain}`, {
        signal: typeof AbortSignal?.timeout==='function' ? AbortSignal.timeout(5000) : undefined,
      });
      if (res.ok) {
        const data = await res.json();
        const logo = data?.formats?.find(f => f.format === 'png') || data?.formats?.find(f => f.format === 'svg');
        if (logo?.src) {
          console.log(`[img] Brandfetch logo: ${logo.src}. Rehosting...`);
          return await rehostToSupabase(logo.src, `${prod.name}_logo.png`, table);
        }
      }
    } catch (e) {
      console.warn(`[img] Brandfetch failed: ${e.message}`);
    }
  }

  const searchQuery = `${shortBrandName(prod)} logo png`;
  const searchImageUrl = await findBestImageWithSearch(searchQuery);
  if (searchImageUrl) {
    console.log(`[img] Using search image: ${searchImageUrl}. Rehosting...`);
    try { return await rehostToSupabase(searchImageUrl, `${prod.name}_search.jpg`, table); }
    catch (e) { console.warn('[img] rehost search failed:', e.message); }
  }
  return null;
}

// DuckDuckGo HTML
async function ddgSearchHTML(query, max = 8) {
  const url = `https://duckduckgo.com/html/?q=${encodeURIComponent(query)}&ia=web`;
  try {
    const res = await fetch(url, {
      headers: { 'user-agent': 'Mozilla/5.0' },
      signal: typeof AbortSignal?.timeout === 'function' ? AbortSignal.timeout(12000) : undefined,
    });
    if (!res.ok) return [];
    const html = await res.text();
    const links = Array.from(html.matchAll(/<a[^>]+class="result__a"[^>]+href="([^"]+)"/gi))
      .map(m => m[1])
      .filter(u => /^https?:\/\//i.test(u));
    return Array.from(new Set(links)).slice(0, max);
  } catch { return []; }
}

// DuckDuckGo LITE (fallback)
async function ddgSearchLite(query, max = 8) {
  const url = `https://duckduckgo.com/lite/?q=${encodeURIComponent(query)}`;
  try {
    const res = await fetch(url, {
      headers: { 'user-agent': 'Mozilla/5.0' },
      signal: typeof AbortSignal?.timeout === 'function' ? AbortSignal.timeout(12000) : undefined,
    });
    if (!res.ok) return [];
    const html = await res.text();
    const links = Array.from(html.matchAll(/<a href="(https?:\/\/[^"]+)"/gi))
      .map(m => m[1])
      .filter(Boolean);
    return Array.from(new Set(links)).slice(0, max);
  } catch { return []; }
}

// Wikipedia opensearch (no key)
async function wikiSearch(phrase) {
  try {
    const url = `https://en.wikipedia.org/w/api.php?action=opensearch&search=${encodeURIComponent(phrase)}&limit=1&namespace=0&format=json`;
    const res = await fetch(url, { signal: typeof AbortSignal?.timeout === 'function' ? AbortSignal.timeout(12000) : undefined });
    if (!res.ok) return null;
    const data = await res.json();
    const link = data?.[3]?.[0];
    return link || null;
  } catch { return null; }
}

// --- Wikipedia deep search (no key) ---
async function wikiBestPage(query) {
  try {
    // broader search to get best matching title
    const url = `https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=${encodeURIComponent(query)}&srlimit=5&format=json`;
    const res = await fetch(url, {
      signal: typeof AbortSignal?.timeout === 'function' ? AbortSignal.timeout(12000) : undefined,
    });
    if (!res.ok) return null;
    const data = await res.json();
    const hits = data?.query?.search || [];
    // prefer exact-ish / top-scoring
    const best = hits[0];
    return best?.title || null;
  } catch { return null; }
}

async function wikiExtractByTitle(title) {
  try {
    const url = `https://en.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&redirects=1&format=json&titles=${encodeURIComponent(title)}`;
    const res = await fetch(url, {
      signal: typeof AbortSignal?.timeout === 'function' ? AbortSignal.timeout(12000) : undefined,
    });
    if (!res.ok) return '';
    const data = await res.json();
    const pages = data?.query?.pages || {};
    const first = Object.values(pages)[0];
    const text = first?.extract || '';
    return text ? text.slice(0, 12000) : '';
  } catch { return ''; }
}


// Use both DDG modes + wikipedia; fetch many pages; try common product paths
async function searchWebForProduct(productName, plan) {
  const q = [productName, plan, 'price features premium plan'].filter(Boolean).join(' ');
  const urls = new Set();

  // DDG html + lite (reuse your ddgSearchHTML / ddgSearchLite functions)
  if (typeof ddgSearchHTML === 'function') (await ddgSearchHTML(q, 8)).forEach(u => urls.add(u));
  if (typeof ddgSearchLite === 'function') (await ddgSearchLite(q, 8)).forEach(u => urls.add(u));

  // pick likely official host from found URLs (levenshtein helpers already present)
  const hosts = Array.from(urls).map(u => {
    try { return new URL(u).hostname.replace(/^www\./,''); } catch { return null; }
  }).filter(Boolean);

  const brand = String(productName||'').toLowerCase().replace(/[^a-z0-9]+/g,'');
  let bestHost = null, bestScore = Infinity;
  for (const h of hosts) {
    const base = (h.split('.').slice(0,-1).join('.') || h).toLowerCase().replace(/[^a-z0-9]+/g,'');
    const d = typeof levenshtein === 'function' ? levenshtein(brand, base) : 999;
    if (d < bestScore) { bestScore = d; bestHost = h; }
  }
  if (bestHost && bestScore <= 3) {
    const base = `https://${bestHost}`;
    ['/','/pricing','/plans','/premium','/subscribe','/membership','/features','/help','/faq']
      .map(p => base.replace(/\/$/,'') + p).forEach(u => urls.add(u));
  }

  // Fetch many pages (direct or via r.jina.ai inside fetchWebsiteRaw)
  const chunks = [];
  let count = 0;
  for (const u of urls) {
    if (count >= 12) break;
    try {
      const { text } = await fetchWebsiteRaw(u);
      if (text && text.length > 200) {
        chunks.push(`SOURCE: ${u}\n${text.slice(0, 4000)}`);
        count++;
      }
    } catch {}
  }

  // Bundle so far
  let bundle = chunks.join('\n\n').slice(0, 20000);

  // 🛟 Deep Wikipedia fallback when evidence is weak
  if (bundle.length < 800) {
    const title = await wikiBestPage(productName);
    if (title) {
      const wikiText = await wikiExtractByTitle(title);
      if (wikiText && wikiText.length > 400) {
        bundle += `\n\nSOURCE: https://en.wikipedia.org/wiki/${encodeURIComponent(title)}\n${wikiText}`;
      }
    }
  }

  // Final log & return
  const pageCount = (bundle.match(/SOURCE:/g) || []).length;
  console.log(`[text] evidence: ${bundle.length} chars from ${pageCount} pages; official=${bestHost || '-'}`);
  return bundle;
}


// ---------- Pollinations Text (OpenAI-compatible, no key) ----------
async function pollinationsTextJSON(systemPrompt, userPrompt, model = 'searchgpt') {
  try {
    const res = await fetch('https://text.pollinations.ai/openai/v1/chat/completions', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        model, temperature: 0.1,
        response_format: { type: 'json_object' },
        messages: [
          { role: 'system', content: systemPrompt + '\nIMPORTANT: Output ONLY raw JSON. No code fences.' },
          { role: 'user', content: userPrompt }
        ]
      }),
      signal: typeof AbortSignal?.timeout === 'function' ? AbortSignal.timeout(20000) : undefined,
    });
    if (!res.ok) throw new Error(`Pollinations text HTTP ${res.status}`);
    const data = await res.json();
    const content = data?.choices?.[0]?.message?.content || '';
    const parsed = safeParseFirstJsonObject(content) ?? JSON.parse(content);
    return parsed;
  } catch (e) {
    console.warn('[text] Pollinations failed:', e.message);
    return null;
  }
}



// ---------- Gemini rotation (optional; check ToS) ----------
// ---------- Gemini rotation (better logging + JSON mode) ----------
let _geminiIndex = 0;

async function geminiTextJSON(systemPrompt, userPrompt) {
  if (!GEMINI_KEYS.length) {
    console.warn('[text] Gemini skipped: no GEMINI_API_KEYS set');
    return null;
  }

  const body = {
    generationConfig: { temperature: 0.1, responseMimeType: 'application/json' },
    systemInstruction: { role: 'user', parts: [{ text: `${systemPrompt}\nReturn only valid JSON.` }]},
    contents: [{ role: 'user', parts: [{ text: userPrompt }]}]
  };

  const tried = new Set();
  for (let i = 0; i < GEMINI_KEYS.length; i++) {
    const key = GEMINI_KEYS[_geminiIndex];
    _geminiIndex = (_geminiIndex + 1) % GEMINI_KEYS.length;
    if (tried.has(key)) continue;
    tried.add(key);

    try {
      const res = await fetch(
        `${GEMINI_BASE}/models/${encodeURIComponent(GEMINI_TEXT_MODEL)}:generateContent?key=${encodeURIComponent(key)}`,
        {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify(body),
          signal: typeof AbortSignal?.timeout === 'function' ? AbortSignal.timeout(20000) : undefined,
        }
      );

      if (!res.ok) {
        const msg = await res.text().catch(()=>'');
        console.warn(`[text] Gemini HTTP ${res.status} — ${msg.slice(0, 300)}`);
        if (res.status === 429) continue; // try next key
        continue;
      }

      const data = await res.json();
      const parts = data?.candidates?.[0]?.content?.parts || [];
      const text = parts.map(p => p.text).filter(Boolean).join('');
      if (!text) {
        const finish = data?.candidates?.[0]?.finishReason || 'unknown';
        console.warn(`[text] Gemini empty content (finishReason=${finish})`);
        continue;
      }

      try {
        const cleaned = text.replace(/```(?:json)?\s*([\s\S]*?)\s*```/gi, '$1').trim();
        return JSON.parse(cleaned);
      } catch (e) {
        console.warn('[text] Gemini JSON parse error:', e.message, 'sample=', text.slice(0, 200));
        continue;
      }
    } catch (e) {
      console.warn('[text] Gemini network/error:', e.message);
    }
  }
  return null;
}


function buildImagePrompt(prod) {
  const name = prod?.name || 'Unnamed Product';
  const plan = prod?.plan || '';
  return `Cinematic, professional product background for "${name}" ${plan}, 4K, modern, clean, vibrant, soft lighting, high detail`;
}


// === Local dynamic card (no external APIs) ===
async function createInitialImage(prod) {
  const title = shortBrandName(prod);
  const plan = prod.plan || '';

  // Background SVG with gradient + soft blobs
  const bgSvg = `
  <svg width="1024" height="1024" viewBox="0 0 1024 1024" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stop-color="#0f172a"/>
        <stop offset="50%" stop-color="#1e293b"/>
        <stop offset="100%" stop-color="#334155"/>
      </linearGradient>
      <radialGradient id="b1" cx="20%" cy="25%" r="40%">
        <stop offset="0%" stop-color="#38bdf8" stop-opacity="0.35"/>
        <stop offset="100%" stop-color="#38bdf8" stop-opacity="0"/>
      </radialGradient>
      <radialGradient id="b2" cx="85%" cy="70%" r="45%">
        <stop offset="0%" stop-color="#a78bfa" stop-opacity="0.35"/>
        <stop offset="100%" stop-color="#a78bfa" stop-opacity="0"/>
      </radialGradient>
    </defs>
    <rect width="1024" height="1024" fill="url(#g)"/>
    <circle cx="220" cy="220" r="280" fill="url(#b1)"/>
    <circle cx="880" cy="760" r="360" fill="url(#b2)"/>
  </svg>`;

  // Title/plan overlay SVG
const esc = (s='') => String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
function wrap(text, maxWidth, maxLines) {
  const words = String(text).trim().split(/\s+/);
  const lines = []; let cur = '';
  for (const w of words) {
    if ((cur + ' ' + w).trim().length <= maxWidth) cur = (cur ? cur + ' ' : '') + w;
    else { lines.push(cur); cur = w; if (lines.length >= maxLines-1) break; }
  }
  if (cur) lines.push(cur);
  return lines.slice(0, maxLines);
}
const titleLines = wrap(title, 18, 2);
const titleSize = titleLines.length > 1 ? 100 : 120;
const tspans = titleLines.map((line,i)=>`<tspan x="512" dy="${i===0?0:'1.2em'}">${esc(line)}</tspan>`).join('');
const overlaySvg = `
<svg width="1024" height="512" viewBox="0 0 1024 512" xmlns="http://www.w3.org/2000/svg">
  <text x="512" y="256" text-anchor="middle"
        font-family="Arial, Helvetica, DejaVu Sans, sans-serif"
        font-size="${titleSize}" font-weight="700" fill="#FFFFFF">${tspans}</text>
  ${plan ? `<text x="512" y="400" text-anchor="middle"
        font-family="Arial, Helvetica, DejaVu Sans, sans-serif"
        font-size="60" font-weight="500" fill="#E5E7EB">${esc(plan)}</text>` : ''}
</svg>`;


  // If sharp is missing, return a single-layer SVG fallback buffer
  if (!_sharp) {
    // merge background + dark overlay + title by simple stacking (best-effort)
    // Telegram/Supabase will accept this as binary content
    return Buffer.from(bgSvg);
  }

  const bgBuf = await _sharp(Buffer.from(bgSvg)).png().toBuffer();
  const overlayBuf = await _sharp(Buffer.from(overlaySvg)).png().toBuffer();

  const composed = await _sharp(bgBuf)
    .composite([
      { input: Buffer.from('<svg width="1024" height="1024"><rect width="1024" height="1024" fill="black" opacity="0.35"/></svg>'), blend: 'over' },
      { input: overlayBuf, gravity: 'center' },
    ])
    .png()
    .toBuffer();

  return composed;
}


/* ---- build prompt used by all generators ---- */
function buildBackgroundPrompt(prod) {
  const theme = `${shortBrandName(prod)}, ${prod.description || prod.category || ''}`.trim();
  return `cinematic, professional product background, abstract, vibrant gradient, soft lighting, modern, clean, themed around "${theme}". 4k, masterpiece.`;
}

/* ---- interactive API selection keyboard ---- */
const kbImageAPIs = Markup.inlineKeyboard([
  [Markup.button.callback('🖼️ Pollinations (Free)', 'imgapi_pollinations')],
  [Markup.button.callback('🤗 Hugging Face', 'imgapi_hf')],
  [Markup.button.callback('🟦 DeepAI', 'imgapi_deepai')],
  [Markup.button.callback('🟣 Local Card (No API)', 'imgapi_local')],
  [Markup.button.callback('🤖 Auto (best effort)', 'imgapi_auto')],
  [Markup.button.callback('❌ Cancel', 'imgapi_cancel')],
]);

const kbTextAPIs = Markup.inlineKeyboard([
  [Markup.button.callback('🦙 Pollinations (SearchGPT)', 'txtapi_pollinations')],
  [Markup.button.callback('🟪 Groq (Llama3-70B)', 'txtapi_groq')],
  [Markup.button.callback('🔷 Gemini (rotation)', 'txtapi_gemini')],
  [Markup.button.callback('🤖 Auto (best → fallback)', 'txtapi_auto')],
  [Markup.button.callback('❌ Cancel', 'txtapi_cancel')],
]);


/* ---- run generation with ordered providers, return hosted URL ---- */
async function generateBackgroundWithOrder(prod, table, order = []) {
  const prompt = buildImagePrompt(prod);
console.log('[img] Using image prompt:', prompt);
  for (const provider of order) {
    let buf = null;

  if (provider === 'pollinations') {
  buf = await tryWithRetries('image:pollinations', () => generateImageFromPollinations(prompt), IMAGE_RETRIES);
} else if (provider === 'hf') {
  buf = await tryWithRetries('image:hf', () => generateImageFromHuggingFace(prompt), IMAGE_RETRIES);
} else if (provider === 'deepai') {
  buf = await tryWithRetries('image:deepai', () => generateImageFromDeepAI(prompt), IMAGE_RETRIES);
} else if (provider === 'local') {
  try {
    const localBuf = await createInitialImage(prod);
    return await rehostToSupabase(localBuf, `${prod.name}_local.png`, table);
  } catch (e) {
    console.warn('[img] local createInitialImage failed:', e.message);
  }
  continue;
}

if (buf && buf.length) {
  const composed = IMAGE_TEXT_OVERLAY
    ? await composeTextOverBackground(buf, prod)
    : buf; // no overlay, just the background
  try {
    return await rehostToSupabase(composed, `${prod.name}_ai.png`, table);
  } catch (e) { console.warn(`[img] rehost after ${provider} failed:`, e.message); }
}
  }

  // last resort: minimal gradient + overlay
 // last resort: minimal gradient (+ optional overlay), then host & return
const fallbackBase = gradientBackgroundSVG();
const fallback = IMAGE_TEXT_OVERLAY
  ? await composeTextOverBackground(fallbackBase, prod)
  : fallbackBase;

try {
  const hosted = await rehostToSupabase(fallback, `${prod.name}_fallback.png`, table);
  return hosted;
} catch (e) {
  console.warn('[img] rehost fallback failed:', e.message);
  return null;
}


/* --------------------- keyboards / messages & Rest --------------------- */
const kbConfirm = Markup.inlineKeyboard([
  [Markup.button.callback('✅ Looks Good & Save', 'save')],
  [Markup.button.callback('✏️ Edit Text', 'edit_text'), Markup.button.callback('🖼️ Change Image', 'change_image')],
  [Markup.button.callback('❌ Cancel', 'cancel')]
]);
const kbChooseTable = Markup.inlineKeyboard([
  [Markup.button.callback('📦 Products', 'set_table_products')],
  [Markup.button.callback('⭐ Exclusive Products', 'set_table_exclusive')]
]);
const kbAfterTask = Markup.inlineKeyboard([
  [Markup.button.callback('➕ Add Another Product', 'again_smartadd')],
  [Markup.button.callback('🏁 Done (Fresh Start)', 'again_done')]
]);

const kbEditWhich = (table) => {
  const textFields = ['name', 'plan', 'validity', 'price', 'description', 'tags'];
  const proFields = ['originalPrice', 'stock', 'category', 'subcategory'];
  const fields = table === TABLES.products ? [...textFields, ...proFields] : textFields;

  const rows = [];
  for (let i=0;i<fields.length;i+=3) rows.push(fields.slice(i,i+3).map(f => Markup.button.callback(f, `edit_field_${f}`)));
  rows.push([Markup.button.callback('⬅️ Back to Review', 'back_review')]);
  return Markup.inlineKeyboard(rows);
};

function reviewMessage(prod, ai, table) {
  const tags = uniqMerge(prod.tags || [], ai.tags || []);
  const parts = [];
  parts.push(`*Review before save*`);
  parts.push(`*Name:* ${escapeMd(prod.name)}`);
  if (ok(prod.plan)) parts.push(`*Plan:* ${escapeMd(prod.plan)}`);
  if (ok(prod.validity)) parts.push(`*Validity:* ${escapeMd(prod.validity)}`);
  parts.push(`*Price:* ${escapeMd(prod.price ? `₹${prod.price}` : '-')}`);
  parts.push(`*Description:* ${ai.description || prod.description}`);
  if (ai.features && ai.features.length > 0) {
    parts.push(`\n*Key Features:*`);
    ai.features.forEach(feature => parts.push(`- ${feature}`));
  }
  parts.push(`\n*Tags:* ${escapeMd(tags.join(', ') || '-')}`);
  if (table === TABLES.products) {
    parts.push(`*MRP:* ${escapeMd(prod.originalPrice ?? '-')}`);
    parts.push(`*Stock:* ${escapeMd(prod.stock ?? '-')}`);
    parts.push(`*Category:* ${escapeMd(ai.category || '-')}`);
    parts.push(`*Subcategory:* ${escapeMd(ai.subcategory || '-')}`);
  }
  parts.push(`\n*Image:* ${prod.image ? `[View Image](${prod.image})` : 'No Image'}`);
  return parts.join('\n');
}

/* --------------------- bot wiring --------------------- */
bot.use(session());
bot.use(async (ctx, next) => { if (!isAdmin(ctx)) return; if (!ctx.session) ctx.session = {}; return next(); });

bot.start(async (ctx) => {
  if (!isAdmin(ctx)) return;
  ctx.session = {};
  // reset any lingering choices
  ctx.session.textOrder = null;
  ctx.session.await = null;
  await ctx.reply('Welcome! Please choose which table you want to work with:', kbChooseTable);
});

bot.command('table', (ctx) => { if (!isAdmin(ctx)) return; ctx.session = {}; replyMD(ctx, 'Type *products* or *exclusive*'); });

bot.command('list', async (ctx) => {
  if (!isAdmin(ctx)) return;
  if (!ctx.session.table) return ctx.reply('Choose table first with /table');
  const { data, error } = await supabase.from(ctx.session.table).select('id,name,price,is_active').order('id', { ascending: false }).limit(12);
  if (error) return ctx.reply(`DB error: ${error.message}`);
  const items = data || [];
  if (!items.length) return ctx.reply('No items yet.');
  const msg = items.map((r, i) => `${i + 1}. ${r.name} — ₹${Number(r.price || 0).toLocaleString('en-IN')} — ${r.is_active ? '✅' : '⛔️'} (id: ${r.id})`).join('\n');
  ctx.reply('Latest:\n' + msg);
});

bot.command('addproduct', (ctx) => {
  if (!isAdmin(ctx)) return;
  if (!ctx.session.table) return replyMD(ctx, 'First choose a table: *products* or *exclusive*');
  ctx.session.mode = 'manual';
  ctx.session.form = { step: 0, prod: {} };
  replyMD(ctx, 'Enter *Name*:');
});

bot.command('smartadd', (ctx) => {
  if (!isAdmin(ctx)) return;
  if (!ctx.session.table) return replyMD(ctx, 'First choose a table: *products* or *exclusive*');
  ctx.session.mode = 'smart';
  ctx.session.await = 'blob';
  ctx.reply('Send the product text (can be messy). You may also attach a photo or include a URL.');
});

bot.command('toggle', async (ctx) => {
  if (!isAdmin(ctx)) return;
  const id = (ctx.message.text.split(' ')[1] || '').trim();
  if (!id) return ctx.reply('Usage: /toggle <id>');
  if (!ctx.session.table) return ctx.reply('Choose table first with /table');

  const { data, error } = await supabase.from(ctx.session.table).select('is_active').eq('id', id).maybeSingle();
  if (error || !data) return ctx.reply('Not found.');
  const { error: upErr } = await supabase.from(ctx.session.table).update({ is_active: !data.is_active }).eq('id', id);
  if (upErr) return ctx.reply(`❌ Toggle failed: ${upErr.message}`);
  ctx.reply(`Toggled id ${id} to ${!data.is_active ? '✅ active' : '⛔️ inactive'}.`);
});

bot.action(/^set_table_(.+)$/, async (ctx) => {
  if (!isAdmin(ctx)) return;
  const tableName = ctx.match[1];
  ctx.session.table = tableName === 'products' ? TABLES.products : TABLES.exclusive;

  // also reset model choice when switching tables
  ctx.session.textOrder = null;
  ctx.session.await = null;

  await ctx.answerCbQuery(`Table set to ${ctx.session.table}`);
  await ctx.deleteMessage().catch(()=>{});
  await ctx.reply(
    escapeMd(`✅ Table set to *${ctx.session.table}*.\n\nCommands:\n• /smartadd (Add a product easily)\n• /list\n• /update <id>\n• /toggle <id>`),
    { parse_mode: 'MarkdownV2' }
  );
});


bot.action('again_smartadd', async (ctx) => {
  if (!isAdmin(ctx)) return;
  await ctx.answerCbQuery();
  await ctx.deleteMessage().catch(()=>{});
  ctx.session.mode = 'smart';
  ctx.session.await = 'blob';
  await ctx.reply('Send the next product text (can be messy). You may also attach a photo or include a URL.');
});

bot.action('again_done', async (ctx) => {
  if (!isAdmin(ctx)) return;
  await ctx.answerCbQuery();
  ctx.session = {};
  await ctx.deleteMessage().catch(()=>{});
  await ctx.reply('🏁 All done! Session cleared. Ready for a new task.');
  await ctx.reply('Please choose a table to begin:', kbChooseTable);
});

/* ----- edit / image change inline flow ----- */
bot.action('edit_text', async (ctx) => {
  if (!ctx.session.review) return ctx.answerCbQuery();
  await ctx.answerCbQuery();
  await ctx.deleteMessage().catch(()=>{});
  await ctx.reply('Which text field would you like to edit?', kbEditWhich(ctx.session.review.table));
});

bot.action('change_image', async (ctx) => {
  if (!ctx.session.review) return ctx.answerCbQuery();
  await ctx.answerCbQuery();
  ctx.session.await = 'image_url';
  await ctx.deleteMessage().catch(()=>{});
  await ctx.reply('Please send a new image URL or upload a photo for the product.');
});

bot.action(/^edit_field_(.+)$/, (ctx) => {
  if (!isAdmin(ctx) || !ctx.session.review) return ctx.answerCbQuery();
  const field = ctx.match[1];
  ctx.session.await = 'edit_field';
  ctx.session.edit = { field };
  ctx.answerCbQuery();
  replyMD(ctx, `Please send the new text for *${field}*:`);
});

function setTextOrder(ctx, order) {
  ctx.session.textOrder = order;
  ctx.session.await = null;
  ctx.answerCbQuery().catch(()=>{});
  ctx.deleteMessage().catch(()=>{});

  if (ctx.session.pendingSmart) {
    // Finish the whole smart-add flow now that a model is chosen
    return resumeSmartAddAfterTextChoice(ctx);
  }
  return ctx.reply('✅ Text provider set.');
}


bot.action('txtapi_pollinations', (ctx)=>setTextOrder(ctx, ['pollinations','groq','gemini']));
bot.action('txtapi_groq',         (ctx)=>setTextOrder(ctx, ['groq','pollinations','gemini']));
bot.action('txtapi_gemini',       (ctx)=>setTextOrder(ctx, ['gemini','groq','pollinations']));
bot.action('txtapi_auto',         (ctx)=>setTextOrder(ctx, ['pollinations','groq','gemini']));
bot.action('txtapi_cancel', async (ctx)=>{
  await ctx.answerCbQuery(); await ctx.deleteMessage().catch(()=>{});
  await ctx.reply('Text model selection cancelled.');
  ctx.session.await = null; ctx.session.pendingText = null;
});


/* ---------------- smart add ---------------- */
// runEnrichment
async function runEnrichment(ctx, text, websiteContent) {
  const aiData = await enrichWithAI(text, websiteContent, ctx.session.textOrder);
  console.log(`[text] filled — name="${aiData.name}" plan="${aiData.plan}" descChars=${(aiData.description||'').length} feats=${aiData.features?.length||0}`);
  return aiData;
}



const smartAddHandler = async (ctx) => {
  if (!ctx.session.table) { await ctx.reply('Please choose a table first.', kbChooseTable); return; }

  const text = ctx.message?.text || '';
  await ctx.reply('🤖 Checking for product URL & fetching site content...');

  try {
    const urlMatch = Array.from(text.matchAll(URL_RX)).map(m => m[0])[0] || null;
const guessedName = text.split('\n')[0].trim();

// 1) user URL > 2) brand slug > 3) search-based official domain (typo fixer)
let domain = urlMatch || resolveBrandDomain(guessedName);
if (!urlMatch && !domain && typeof pickOfficialDomainFromSearch === 'function') {
  try {
    const bestHost = await pickOfficialDomainFromSearch(guessedName);
    if (bestHost) domain = `https://${bestHost}`;
  } catch {}
}

let websiteContent = '';
let structured = null;
let meta = {};
let ogImageFromPage = null;

if (domain) {
  const fullUrl = domain.startsWith('http') ? domain : `https://${domain}`;
  await ctx.reply(`🌐 Reading ${fullUrl} ...`);
  const { html, text: pageText } = await fetchWebsiteRaw(fullUrl);
  websiteContent = pageText;
  if (html) {
    structured = extractJsonLdProduct(html);
    meta = extractMetaTags(html);
    ogImageFromPage = meta.ogImage || null;
  }
} else {
  await ctx.reply('🔎 Couldn’t auto-detect the official site. I’ll rely on web search evidence.');
}


    // AI parse
   // TEXT MODEL CHOICE (once per session or per product)
if (!ctx.session.textOrder) {
  console.log('[text] no textOrder; asking user to choose text model');
 ctx.session.await = 'choose_text_api';
ctx.session.pendingSmart = { text, websiteContent, ogImageFromPage }; // <-- keep OG image too
await ctx.reply('Choose which **text model** to fill product details (I’ll retry and fallback automatically):', kbTextAPIs);
return;
}
console.log('[text] using textOrder =', ctx.session.textOrder);



// AI parse with chosen order
const aiData = await runEnrichment(ctx, text, websiteContent);

 await ctx.reply(
  `📝 Parsed:
• Name: ${aiData.name}
• Plan: ${aiData.plan}
• Validity: ${aiData.validity}
• Price: ${aiData.price ?? '-'}
• Category: ${aiData.category}
(Generating/choosing image next…)`
).catch(e => console.error('[tg] reply Parsed failed:', e));


    const prod = { ...aiData, image: null };

    // Try page OG image first
    if (ogImageFromPage) {
        console.log('[img] starting image step; ogImageFromPage=', !!ogImageFromPage);
      try {
        prod.image = await rehostToSupabase(ogImageFromPage, `${prod.name}.jpg`, ctx.session.table);
      } catch (e) { console.warn('[img] OG rehost failed:', e.message); }
    }
    // Try brand sources if still empty
    if (!prod.image) {
      const hosted = await tryBrandImages(prod, ctx.session.table);
      if (hosted) prod.image = hosted;
    }

    // Prepare session review
    ctx.session.review = { prod, ai: aiData, table: ctx.session.table };
    ctx.session.mode = null;

    if (prod.image) {
      // We have an image, present review immediately
      const caption = reviewMessage(prod, aiData, ctx.session.table);
      return prod.image
        ? ctx.replyWithPhoto({ url: prod.image }, { caption, parse_mode: 'Markdown', ...kbConfirm })
        : replyMD(ctx, caption, kbConfirm);
    } else {
      // No image yet — ask user which API to use
      ctx.session.await = 'choose_image_api';
      ctx.session.pendingImage = { table: ctx.session.table };
      await ctx.reply('Choose which image API to use first (fallbacks will be tried automatically if it fails):', kbImageAPIs);
      return;
    }

  } catch (e) {
    console.error('Smart add flow failed:', e);
    await ctx.reply(`❌ An error occurred: ${e.message}. Please try again.`);
    ctx.session = { table: ctx.session.table };
  }
};

async function resumeSmartAddAfterTextChoice(ctx) {
  const pending = ctx.session.pendingSmart;
  if (!pending) return ctx.reply('No pending product to resume. Send /smartadd again.');

  ctx.session.pendingSmart = null;
  const { text, websiteContent, ogImageFromPage } = pending;

  const aiData = await runEnrichment(ctx, text, websiteContent);

  await ctx.reply(
    `📝 Parsed:
• Name: ${aiData.name}
• Plan: ${aiData.plan}
• Validity: ${aiData.validity}
• Price: ${aiData.price ?? '-'}
• Category: ${aiData.category}
(Generating/choosing image next…)`
  ).catch(e => console.error('[tg] reply Parsed failed:', e));

  const prod = { ...aiData, image: null };

  // try OG first
  if (ogImageFromPage) {
    try {
      prod.image = await rehostToSupabase(ogImageFromPage, `${prod.name}.jpg`, ctx.session.table);
    } catch (e) { console.warn('[img] OG rehost failed:', e.message); }
  }
  if (!prod.image) {
    const hosted = await tryBrandImages(prod, ctx.session.table);
    if (hosted) prod.image = hosted;
  }

  ctx.session.review = { prod, ai: aiData, table: ctx.session.table };
  ctx.session.mode = null;

  if (prod.image) {
    const caption = reviewMessage(prod, aiData, ctx.session.table);
    return ctx.replyWithPhoto({ url: prod.image }, { caption, parse_mode: 'Markdown', ...kbConfirm });
  } else {
    ctx.session.await = 'choose_image_api';
    ctx.session.pendingImage = { table: ctx.session.table };
    return ctx.reply('Choose which image API to use first (fallbacks will be tried automatically if it fails):', kbImageAPIs);
  }
}


/* ----- present review helper ----- */
async function presentReview(ctx) {
  if (!ctx.session.review) return;
  const { prod, ai, table } = ctx.session.review;
  const caption = reviewMessage(prod, ai, table);
  await ctx.deleteMessage().catch(()=>{});
  if (prod.image) {
    return ctx.replyWithPhoto({ url: prod.image }, { caption, parse_mode: 'Markdown', ...kbConfirm });
  } else {
    return replyMD(ctx, caption, kbConfirm);
  }
}

/* ----- inline edit apply ----- */
async function applyInlineEdit(ctx) {
  if (!ctx.session.review || !ctx.session.await || !ctx.session.edit) return;
  const field = ctx.session.edit.field;
  let val = (ctx.message?.text || '').trim();

  // tidy last two messages
  if (ctx.message?.message_id) {
    await ctx.deleteMessage(ctx.message.message_id - 1).catch(()=>{});
    await ctx.deleteMessage().catch(()=>{});
  }

  if (['price','originalPrice','stock'].includes(field)) val = parsePrice(val);
  if (['tags','features'].includes(field)) val = val.split(';').map(x => x.trim()).filter(Boolean);

  if (Object.prototype.hasOwnProperty.call(ctx.session.review.prod, field)) ctx.session.review.prod[field] = val;
  else ctx.session.review.ai[field] = val;

  ctx.session.await = null;
  ctx.session.edit = null;

  await presentReview(ctx);
}

/* ----- text handler ----- */
bot.on('text', async (ctx, next) => {
  if (!isAdmin(ctx)) return;
  const text = ctx.message?.text || '';

   // 🛡 Prevent random text while waiting for text or image model selection
  if (ctx.session.await === 'choose_text_api') {
    // Waiting for the inline button selection for text model
    return;
  }
  if (ctx.session.await === 'choose_image_api') {
    // Waiting for the inline button selection for image API
    return;
  }

  if (!ctx.session.table && !text.startsWith('/')) {
    return ctx.reply('Welcome! To get started, please choose a table.', kbChooseTable);
  }

 
  if (ctx.session.await === 'image_url' && text.startsWith('http')) {
    await ctx.reply('🔗 Got it. Rehosting your image URL...');
    try {
      const imageUrl = await rehostToSupabase(
        text,
        `${ctx.session.review?.prod?.name || 'product'}.jpg`,
        ctx.session.review.table
      );
      ctx.session.review.prod.image = imageUrl;
      ctx.session.await = null;
      await presentReview(ctx);
    } catch (e) {
      await ctx.reply('❌ That URL didn’t work. Please try another one, or upload a photo.');
    }
    return;
  }

  if (ctx.session.await === 'edit_field') {
    return applyInlineEdit(ctx);
  }

  if (
    ctx.session.table &&
    !text.startsWith('/') &&
    (text.includes('\n') || text.length > 40)
  ) {
    return smartAddHandler(ctx);
  }

  if (ctx.session.mode === 'smart' && ctx.session.await === 'blob') {
    return smartAddHandler(ctx);
  }

  return next();
});


/* -------- image API selection handlers -------- */
async function handleImageChoice(ctx, order) {
  if (!ctx.session.review) return ctx.answerCbQuery();
  await ctx.answerCbQuery();
  await ctx.deleteMessage().catch(()=>{});
  await ctx.reply('🎨 Generating product image...');

  const { prod } = ctx.session.review;
  const table = ctx.session.review.table;

  const hosted = await generateBackgroundWithOrder(prod, table, order);
  if (hosted) {
    ctx.session.review.prod.image = hosted;
    ctx.session.await = null;
    ctx.session.pendingImage = null;
    return presentReview(ctx);
  } else {
    await ctx.reply('❌ All image APIs failed. Using fallback.');
    const fb = await rehostToSupabase(await composeTextOverBackground(gradientBackgroundSVG(), prod), `${prod.name}_fallback.png`, table);
    ctx.session.review.prod.image = fb;
    ctx.session.await = null;
    ctx.session.pendingImage = null;
    return presentReview(ctx);
  }
}

bot.action('imgapi_pollinations', (ctx)=>handleImageChoice(ctx, ['pollinations','hf','deepai']));
bot.action('imgapi_hf',          (ctx)=>handleImageChoice(ctx, ['hf','pollinations','deepai']));
bot.action('imgapi_deepai',      (ctx)=>handleImageChoice(ctx, ['deepai','pollinations','hf']));
bot.action('imgapi_local', (ctx) => handleImageChoice(ctx, ['local','pollinations','hf','deepai']));

// tweak Auto to include local as the final step
bot.action('imgapi_auto',  (ctx) => handleImageChoice(ctx, ['pollinations','hf','deepai','local']));

bot.action('imgapi_cancel', async (ctx) => {
  await ctx.answerCbQuery();
  await ctx.deleteMessage().catch(()=>{});
  await ctx.reply('Image generation cancelled. You can tap "🖼️ Change Image" later to provide one.');
  ctx.session.await = null;
  ctx.session.pendingImage = null;
  if (ctx.session.review) await presentReview(ctx);
});

/* ----- media stubs (unchanged/no-op) ----- */
async function processIncomingImage(_ctx, _fileId) { return; }
bot.on('photo', async (_ctx) => { return; });
bot.on('document', async (_ctx) => { return; });

/* ----- cancel / save ----- */
bot.action('cancel', (ctx) => {
  ctx.answerCbQuery();
  ctx.session.review = null;
  ctx.session.mode = null;
  ctx.deleteMessage().catch(()=>{});
  ctx.reply('Cancelled.', kbAfterTask);
});

bot.action('save', async (ctx) => {
  if (!ctx.session.review) return ctx.answerCbQuery('Nothing to save');
  await ctx.answerCbQuery('Saving…');

  const { prod, ai, table, updateId } = ctx.session.review;
  const imgCol = table === TABLES.products ? 'image' : 'image_url';

  const rest = table === TABLES.products
    ? { name: prod.name, plan: prod.plan || null, validity: prod.validity || null, price: prod.price || null, originalPrice: prod.originalPrice || null, description: prod.description || ai.description || null, category: ai.category, subcategory: ai.subcategory || null, stock: prod.stock || null, tags: uniqMerge(prod.tags, ai.tags), features: ai.features || [], is_active: true, }
    : { name: prod.name, description: prod.description || ai.description || null, price: prod.price || null, is_active: true, tags: uniqMerge(prod.tags, ai.tags), };

  let error;
  if (updateId) {
    const { error: imgErr } = await supabase.from(table).update({ [imgCol]: prod.image }).eq('id', updateId);
    if (imgErr) error = imgErr;
    else ({ error } = await supabase.from(table).update(rest).eq('id', updateId));
  } else {
    const { data: dup } = await supabase.from(table).select('id').eq('name', prod.name).eq('price', prod.price).maybeSingle();
    if (dup) {
      await ctx.deleteMessage().catch(()=>{});
      return ctx.reply(`⚠️ Product with same name & price exists (id: ${dup.id}). Cancelled save.`, kbAfterTask);
    }
    ({ error } = await supabase.from(table).insert([{ ...rest, [imgCol]: prod.image }]));
  }

  await ctx.deleteMessage().catch(()=>{});
  if (error) await ctx.reply(`❌ Save failed: ${error.message}`);
  else {
    await ctx.reply(escapeMd(`✅ Saved to ${table}`), { parse_mode: 'MarkdownV2' });
    await ctx.reply('What next?', kbAfterTask);
  }

  ctx.session.review = null;
  ctx.session.mode = null;
});

/* ----- placeholders to preserve your "Unchanged" comments ----- */
bot.action('edit', async (_ctx) => { /* Unchanged */ });
bot.action(/^edit_(.+)$/, (_ctx) => { /* Unchanged */ });
bot.action('back_review', async (_ctx) => { /* Unchanged */ });
bot.command('update', async (_ctx) => { /* Unchanged */ });

/* --------------------- error & lifecycle --------------------- */
bot.catch((err, ctx) => {
  console.error(`Bot error for user ${ctx.from?.id}:`, err);
  try { if (isAdmin(ctx)) ctx.reply('⚠️ Unexpected error. Check console logs.'); } catch {}
});

(async () => {
  try { await bot.telegram.deleteWebhook({ drop_pending_updates: true }); } catch {}
  await bot.launch();
  console.log('🚀 Product Bot running with /smartadd and /update');
  const shutdown = async (signal) => {
    console.log(`\n${signal} received. Stopping bot...`);
    try { await bot.stop(); process.exit(0); } catch (e) { console.error('Error on shutdown:', e); process.exit(1); }
  };
  process.once('SIGINT', () => shutdown('SIGINT'));
  process.once('SIGTERM', () => shutdown('SIGTERM'));
})();
