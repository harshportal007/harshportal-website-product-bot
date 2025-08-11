// --- snip: everything is identical to your last file EXCEPT:
// 1) No OpenAI import / usage
// 2) DEFAULT_IMAGE_MODEL and generateProductImageBytes are updated to Imagen 3
// 3) Image calls use responseMimeType: "image/png"
// I’m including the whole file so you can just replace it.

require('dotenv').config();
const { Telegraf, session, Markup } = require('telegraf');
const { createClient } = require('@supabase/supabase-js');
const fetch = globalThis.fetch ?? ((...a) => import('node-fetch').then(({ default: f }) => f(...a)));
const { GoogleGenerativeAI } = require('@google/generative-ai');

const ADMIN_IDS = (process.env.ADMIN_IDS || '7057639075')
  .split(',').map(s => Number(s.trim())).filter(Boolean);

const TABLES = { products: 'products', exclusive: 'exclusive_products' };

const bot = new Telegraf(process.env.TELEGRAM_BOT_TOKEN);
const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_KEY);
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

const CATEGORIES_ALLOWED = ['OTT Accounts', 'IPTV', 'Product Key', 'Download'];
const categories = CATEGORIES_ALLOWED;

/* ----------------------- utils ----------------------- */
const isAdmin = (ctx) => ADMIN_IDS.includes(ctx.from.id);
const ok = (x) => typeof x !== 'undefined' && x !== null && x !== '';

const toStr = (v) => {
  if (v == null) return '';
  if (typeof v === 'string') return v;
  if (Array.isArray(v)) return v.join(', ');
  if (typeof v === 'object') return JSON.stringify(v);
  return String(v);
};

const TEXT_MODEL = (process.env.GEMINI_TEXT_MODEL || 'gemini-2.5-pro').trim();
// 👇 default to Imagen 3 (works today). You can override with env.
const DEFAULT_IMAGE_MODEL =
  (process.env.GEMINI_IMAGE_MODEL && process.env.GEMINI_IMAGE_MODEL.trim()) ||
  'imagen-3.0';

const escapeMd = (v = '') => toStr(v).replace(/([_\*\[\]\(\)~`>#+\-=|{}\.!])/g, '\\$1');

function safeParseFirstJsonObject(s) {
  if (!s) return null;
  s = String(s).replace(/```(?:json)?\s*([\s\S]*?)\s*```/gi, '$1').trim();
  const m = s.match(/\{[\s\S]*\}/m);
  if (!m) return null;
  try { return JSON.parse(m[0]); } catch { return null; }
}

const parsePrice = (raw) => {
  if (!raw) return null;
  const m = String(raw).match(/(\d[\d,\.]*)/);
  if (!m) return null;
  return Math.round(parseFloat(m[1].replace(/,/g, '')));
};

const uniqMerge = (...arrs) => {
  const set = new Set();
  arrs.flat().filter(Boolean).forEach(x => set.add(String(x).trim()));
  return Array.from(set).filter(Boolean);
};

const CATEGORY_RULES = [
  { rx: /(iptv|live\s*tv)/i, to: 'IPTV' },
  { rx: /(product\s*key|license|licen[cs]e|activation|serial)/i, to: 'Product Key' },
  { rx: /(download|installer|setup|software|vpn)/i, to: 'Download' },
  { rx: /(ott|spotify|netflix|youtube|yt|prime|disney|hbo|hotstar|sonyliv|zee|music|stream)/i, to: 'OTT Accounts' },
];

function normalizeCategoryFromText(text = '') {
  for (const r of CATEGORY_RULES) if (r.rx.test(text)) return r.to;
  return null;
}

function normalizeCategory(prodLike = {}, aiCategory) {
  if (CATEGORIES_ALLOWED.includes(aiCategory)) return aiCategory;

  const hay = [
    prodLike.name, prodLike.description, prodLike.category, prodLike.subcategory,
    Array.isArray(prodLike.tags) ? prodLike.tags.join(' ') : prodLike.tags
  ].filter(Boolean).join(' ');

  const inferred = normalizeCategoryFromText(hay) || normalizeCategoryFromText(aiCategory || '');
  if (inferred) return inferred;

  return 'Download';
}

// Telegram file URL helper
async function tgFileUrl(fileId) {
  try {
    const f = await bot.telegram.getFile(fileId);
    const token = process.env.TELEGRAM_BOT_TOKEN;
    return `https://api.telegram.org/file/bot${token}/${f.file_path}`;
  } catch {
    const link = await bot.telegram.getFileLink(fileId);
    return typeof link === 'string' ? link : link.toString();
  }
}

function guessMime(buf, filenameHint='') {
  const ext = (filenameHint.split('.').pop() || '').toLowerCase();
  const head = buf.subarray(0, 12);

  const isPNG  = head[0]===0x89 && head[1]===0x50 && head[2]===0x4E && head[3]===0x47;
  const isJPG  = head[0]===0xFF && head[1]===0xD8 && head[2]===0xFF;
  const isRIFF = head[0]===0x52 && head[1]===0x49 && head[2]===0x46 && head[3]===0x46 &&
                 head[8]===0x57 && head[9]===0x45 && head[10]===0x42 && head[11]===0x50;

  if (isPNG) return { mime: 'image/png', ext: 'png' };
  if (isJPG) return { mime: 'image/jpeg', ext: 'jpg' };
  if (isRIFF) return { mime: 'image/webp', ext: 'webp' };

  if (ext === 'png') return { mime: 'image/png', ext: 'png' };
  if (ext === 'jpg' || ext === 'jpeg') return { mime: 'image/jpeg', ext: 'jpg' };
  if (ext === 'webp') return { mime: 'image/webp', ext: 'webp' };
  if (ext === 'svg') return { mime: 'image/svg+xml', ext: 'svg' };

  return { mime: 'image/jpeg', ext: 'jpg' };
}

/* --------------- OpenGraph --------------- */
const findUrls = (txt = '') =>
  (txt.match(/https?:\/\/\S+/gi) || [])
    .map(u => u.replace(/[),.\]]+$/,''))
    .slice(0, 5);

async function getOG(url) {
  const ac = new AbortController();
  const t = setTimeout(() => ac.abort(), 10_000);
  try {
    const res = await fetch(url, { signal: ac.signal });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const html = await res.text();
    const pick = (prop) => {
      const r =
        new RegExp(`<meta[^>]+property=["']${prop}["'][^>]+content=["']([^"']+)["']`, 'i').exec(html) ||
        new RegExp(`<meta[^>]+name=["']${prop}["'][^>]+content=["']([^"']+)["']`, 'i').exec(html);
      return r ? r[1] : null;
    };
    return {
      title: pick('og:title') || pick('twitter:title'),
      description: pick('og:description') || pick('twitter:description'),
      image: pick('og:image:secure_url') || pick('og:image') || pick('twitter:image'),
    };
  } catch {
    return {};
  } finally { clearTimeout(t); }
}

function isSupabasePublicUrl(u='') {
  try {
    const url = new URL(u);
    const base = new URL(process.env.SUPABASE_URL);
    return url.hostname === base.hostname && /\/storage\/v1\/object\/public\//.test(url.pathname);
  } catch { return false; }
}

async function ensureHostedInSupabase(u, table, filenameHint='prod.jpg') {
  if (!u || !/^https?:\/\//i.test(u)) return u;
  if (isSupabasePublicUrl(u)) return u;
  return rehostToSupabase(u, filenameHint, table);
}

/* --------------- Supabase Storage --------------- */
async function rehostToSupabase(fileUrl, filenameHint = 'image.jpg', table) {
  try {
    const res = await fetch(fileUrl);
    if (!res.ok) throw new Error(`Fetch failed: ${res.status}`);
    const buf = Buffer.from(await res.arrayBuffer());

    const bucket = table === TABLES.products
      ? (process.env.SUPABASE_BUCKET_PRODUCTS || 'images')
      : (process.env.SUPABASE_BUCKET_EXCLUSIVE || 'exclusiveproduct-images');
    const folder = table === TABLES.products ? 'products' : 'exclusive-products';

    const headerCT = (res.headers.get('content-type') || '').toLowerCase();
    let { mime, ext } = guessMime(buf, filenameHint);
    if (headerCT.startsWith('image/') && headerCT !== 'application/octet-stream') {
      mime = headerCT;
      ext  = headerCT.includes('png') ? 'png'
          : headerCT.includes('webp') ? 'webp'
          : headerCT.includes('svg') ? 'svg' : 'jpg';
    }

    const base = filenameHint.replace(/\.[A-Za-z0-9]+$/, '');
    const safeName = `${base}.${ext}`;
    const key = `${folder}/${Date.now()}-${Math.random().toString(36).slice(2)}-${safeName}`;

    const { error: upErr } = await supabase.storage.from(bucket).upload(key, buf, {
      contentType: mime,
      upsert: true,
    });
    if (upErr) throw upErr;

    const { data: pub } = supabase.storage.from(bucket).getPublicUrl(key);
    return pub.publicUrl;
  } catch (e) {
    console.error('Rehost failed:', e.message);
    throw e;
  }
}

/* ---------------- Brand & Style helpers ---------------- */
const STYLE_THEMES = {
  neo:      'dark glossy, subtle smoke, neon rim light, soft reflections, glassmorphism',
  minimal:  'clean minimal studio light, white/very light background, soft shadow, no gradients',
  gradient: 'bold smooth gradient background, soft glow, depth of field',
  cyber:    'cyberpunk, dark, holographic HUD lines, neon rim light',
  clay:     'clay 3D render, soft pastel background, gentle shadow'
};

const BRAND_STYLES = [
  { match: /\bspotify\b/i, name: 'Spotify', palette: ['#1DB954', '#121212', '#1e1e1e'], keywords: '' },
  { match: /\bnetflix\b/i, name: 'Netflix', palette: ['#E50914', '#0B0B0B'], keywords: '' },
  { match: /\bv0(\.dev)?|\bvercel\b/i, name: 'Vercel v0', palette: ['#0b0b0b', '#111111', '#ffffff'], keywords: '' },
  { match: /\byou ?tube|yt ?premium\b/i, name: 'YouTube', palette: ['#FF0000', '#FFFFFF', '#0f0f0f'], keywords: '' },
  { match: /\bprime|amazon prime\b/i, name: 'Prime Video', palette: ['#00A8E1', '#0B0B0B', '#0a2533'], keywords: '' },
  { match: /\bdisney\b/i, name: 'Disney+', palette: ['#0d3df2', '#0b0b0b', '#1a73e8'], keywords: '' },
  { match: /\biptv\b/i, name: 'IPTV', palette: ['#00E0FF', '#141414', '#00FFA3'], keywords: '' },
  {
    match: /\bspotify|netflix|youtube|yt|prime|amazon|disney|iptv|hbo|hotstar|zee|sonyliv|hulu|paramount\b/i,
    name: 'OTT', palette: ['#00FFC6', '#0B0B0B', '#7C4DFF'], keywords: ''
  },
];

function getBrandStyle(prod) {
  const tags = Array.isArray(prod?.tags) ? prod.tags.join(' ') : (prod?.tags || '');
  const hay = [prod?.name, prod?.description, prod?.category, prod?.subcategory, tags]
    .filter(Boolean).join(' ').toLowerCase();
  return BRAND_STYLES.find(b => b.match.test(hay)) || null;
}

const BRAND_ICON_RECIPES = {
  spotify: { bg: '#121212', fg: '#1DB954', describe: '' },
  netflix: { bg: '#0B0B0B', fg: '#E50914', describe: '' },
  youtube: { bg: '#0f0f0f', fg: '#FF0000', describe: '' },
  'v0':    { bg: '#0b0b0b', fg: '#ffffff', describe: '' },
  raycast: { bg: '#0b0b0b', fg: '#FF6363', describe: '' },
  warp:    { bg: '#0b0b0b', fg: '#7C4DFF', describe: '' }
};

function brandRecipe(prod) {
  const n = String(prod?.name || '').toLowerCase();
  for (const key of Object.keys(BRAND_ICON_RECIPES)) {
    if (n.includes(key)) return BRAND_ICON_RECIPES[key];
  }
  return null;
}

/* --- refs & watermark helpers --- */
async function fetchAsBase64(url) {
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`ref fetch ${res.status}`);
    const buf = Buffer.from(await res.arrayBuffer());
    const mime = res.headers.get('content-type') || 'image/jpeg';
    return { mime, b64: buf.toString('base64') };
  } catch (e) {
    console.warn('ref image fetch failed:', e.message);
    return null;
  }
}

let canSvgText = true;

async function tryCompositeSvg(baseBuf, svgBuf, opts={}) {
  try {
    return await _sharp(baseBuf).composite([{ input: svgBuf, ...opts }]).png().toBuffer();
  } catch (e) {
    if (/Fontconfig/i.test(String(e))) {
      canSvgText = false;
      console.warn('SVG text disabled: fontconfig missing');
      return baseBuf;
    }
    throw e;
  }
}

let _sharp = null;
try { _sharp = require('sharp'); } catch {}

async function addWatermarkToBuffer(buf, text = 'Harshportal') {
  if (!_sharp) return null;
  const svg = Buffer.from(`
  <svg xmlns="http://www.w3.org/2000/svg" width="1200" height="320">
    <text x="98%" y="82%"
      font-family="Inter, Segoe UI, Roboto, Arial"
      font-size="120" font-weight="900"
      text-anchor="end"
      fill="#ffffff" opacity=".38"
      stroke="#000000" stroke-opacity=".28" stroke-width="3"
      transform="rotate(-8 1000 250)">${text}</text>
  </svg>`);
  return await _sharp(buf)
    .composite([{ input: svg, gravity: 'southeast', blend: 'over' }])
    .png()
    .toBuffer();
}

/* ---------------- Gemini caller with retry ---------------- */
async function callGeminiWithRetry(modelId, contentsOrPrompt, genCfg = {}, tries = 3) {
  let lastErr;
  for (let i = 0; i < tries; i++) {
    try {
      const model = genAI.getGenerativeModel({ model: modelId });
      const payload = typeof contentsOrPrompt === 'string'
        ? { contents: [{ role: 'user', parts: [{ text: contentsOrPrompt }]}], generationConfig: genCfg }
        : { contents: [{ role: 'user', parts: contentsOrPrompt }], generationConfig: genCfg };
      return await model.generateContent(payload);
    } catch (e) {
      const msg = String(e?.message || e);
      if (/hard limit|permission|API key|quota/i.test(msg)) throw e;
      lastErr = e;
      await new Promise(r => setTimeout(r, 600 * (i + 1)));
    }
  }
  throw lastErr;
}

/* ---------------- AI enrichment (unchanged) ---------------- */
async function enrichWithAI(prod, textHints = '', ogHints = {}) {
  const brand = getBrandStyle?.(prod);

  const prompt = `
You are a product data normalizer. Fix misspellings, normalize brands, infer likely plan/validity,
and return compact JSON. Do NOT invent prices. If unsure, use "unknown".
Keep "description" <= 220 chars. Also extract 3–6 short "features".

Given:
- Raw product JSON: ${JSON.stringify(prod)}
- Extra text: """${textHints}"""
- OpenGraph hints: ${JSON.stringify(ogHints)}
- Allowed categories: ${categories.join(' | ')}

Return ONLY JSON:
{
  "name": "string",
  "plan": "string|unknown",
  "validity": "string|unknown",
  "price": "number|unknown",
  "description": "string",
  "tags": ["3-8 tags"],
  "category": "one of: ${categories.join(' | ')}",
  "subcategory": "string|unknown",
  "features": ["3-6 concise bullet phrases"],
  "gradient": ["#hex1","#hex2"]
}
`.trim();

  try {
    const out = await callGeminiWithRetry(TEXT_MODEL, prompt);
    const text = out.response.text().trim();
    const json = safeParseFirstJsonObject(text) || {};

    json.name = (json.name || prod.name || '').toString().trim();
    json.description = (json.description || prod.description || '').toString().trim();
    if (json.price !== 'unknown') json.price = parsePrice(json.price);

    json.category = normalizeCategory({ ...prod, ...json }, json.category);
    json.tags = Array.isArray(json.tags) ? json.tags.slice(0, 8) : [];

    let feats = Array.isArray(json.features) ? json.features : [];
    if (!feats.length && textHints) {
      feats = String(textHints)
        .split(/\n|[;•·\-–—]\s+/g).map(s => s.trim())
        .filter(s => s.length > 3 && s.length <= 80).slice(0, 6);
    }
    json.features = feats
      .map(s => s.replace(/^[\-\*\•\•–—]\s*/, '').trim())
      .filter(Boolean).slice(0, 6);

    const palette = (brand?.palette && brand.palette.length >= 2)
      ? brand.palette.slice(0, 2)
      : ['#0ea5e9', '#7c3aed'];
    if (!Array.isArray(json.gradient) || json.gradient.length < 2) {
      json.gradient = palette;
    } else {
      json.gradient = json.gradient.slice(0, 2);
    }

    return json;
  } catch (e) {
    console.error('AI enrich error:', e.message);
    const palette = (getBrandStyle?.(prod)?.palette?.slice(0,2)) || ['#0ea5e9','#7c3aed'];
    return {
      name: prod.name || '',
      plan: prod.plan || 'unknown',
      validity: prod.validity || 'unknown',
      price: parsePrice(prod.price) || null,
      description: prod.description || '',
      tags: [],
      category: normalizeCategory(prod, prod.category || ''),
      subcategory: 'unknown',
      features: [],
      gradient: palette
    };
  }
}

/** extract single product from freeform */
async function extractFromFreeform(text) {
  const prompt = `
Extract a single product from this freeform text (it might be messy).
If quantity/pack/plan/validity are present, keep them.
Return ONLY JSON: {"name":"","plan":"","validity":"","price":"","description":""}

Text:
${text}
`.trim();
  try {
    const out = await callGeminiWithRetry(TEXT_MODEL, prompt);
    const raw = out.response.text();
    const item = safeParseFirstJsonObject(raw) || {};
    item.price = parsePrice(item.price);
    return {
      name: item.name || '',
      plan: item.plan || '',
      validity: item.validity || '',
      price: item.price || null,
      description: item.description || ''
    };
  } catch {
    return { name: '', plan: '', validity: '', price: null, description: '' };
  }
}

/* robust list extractor */
function tryParseJsonArrayAnywhere(raw) {
  if (!raw) return null;
  let s = String(raw).trim();
  s = s.replace(/```(?:json)?\s*([\s\S]*?)\s*```/gi, '$1');
  const start = s.indexOf('[');
  if (start === -1) return null;
  let depth = 0, end = -1;
  for (let i = start; i < s.length; i++) {
    const ch = s[i];
    if (ch === '[') depth++;
    if (ch === ']') { depth--; if (depth === 0) { end = i; break; } }
  }
  if (end === -1) return null;
  try { return JSON.parse(s.slice(start, end + 1)); } catch { return null; }
}

async function extractManyFromFreeform(text) {
  const prompt = `
You will extract a LIST of products from messy text.
Return ONLY JSON array in this exact shape (no prose):

[
  {"name":"","plan":"","validity":"","price":"","description":"","tags":[]}
]

Text:
"""${text}"""
`.trim();
  try {
    const out = await callGeminiWithRetry(TEXT_MODEL, prompt);
    const raw = out.response.text();

    let arr = tryParseJsonArrayAnywhere(raw);
    if (!Array.isArray(arr)) {
      const lines = String(text).split(/\n+/).map(s => s.trim()).filter(Boolean);
      arr = [];
      for (const line of lines) {
        const one = await extractFromFreeform(line);
        if (one?.name) arr.push(one);
      }
    }

    arr = (arr || []).map(x => ({
      name: x.name || '',
      plan: x.plan || '',
      validity: x.validity || '',
      price: parsePrice(x.price),
      description: x.description || '',
      tags: Array.isArray(x.tags) ? x.tags.slice(0,8) : [],
      _txt: text
    }));

    return arr.filter(x => x.name);
  } catch {
    return [];
  }
}

function shortBrandName(prod) {
  const b = getBrandStyle(prod);
  if (b?.name) return b.name;
  const n = String(prod?.name || '')
    .replace(/\b(premium|pro|subscription|subs|account|license|key|activation)\b/ig, '')
    .trim();
  return n || (prod?.name || 'Product');
}

/* ---------------- IMAGE GENERATION ---------------- */
// exact prompt you asked for
async function buildImagePrompt(prod) {
  const name = (prod?.name || '').toString().trim() || 'Product';
  const plan = (prod?.plan || '').toString().trim();
  const desc = (prod?.description || '').toString().trim();
  const title = [name, plan].filter(Boolean).join(' ');
  return `Generate ${title} Image.${desc ? ' ' + desc : ''}`;
}

// small timeout helper
function withTimeout(ms) {
  let timer = null;
  const p = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(`AI call timed out after ${ms}ms`)), ms);
  });
  return {
    race: (promise) => Promise.race([promise, p]),
    clear: () => timer && clearTimeout(timer),
  };
}

// ✅ Gemini-only image generation (Imagen 3). Returns a PNG buffer.
async function generateProductImageBytes({ prompt }) {
  const candidates = Array.from(new Set([
    DEFAULT_IMAGE_MODEL,        // e.g. imagen-3.0 (recommended)
    'imagen-3.0-fast',          // cheap/fast fallback
  ]));

  let lastErr;
  for (const id of candidates) {
    try {
      const t = withTimeout(25000);
      const res = await t.race(
        callGeminiWithRetry(
          id,
          String(prompt || ''),
          { temperature: 0.2, responseMimeType: 'image/png' },
          2
        )
      );
      t.clear();

      const parts = res.response?.candidates?.[0]?.content?.parts ?? [];
      const imagePart = parts.find(p => p.inlineData && p.inlineData.data);
      if (!imagePart) throw new Error('No inline image returned');

      return Buffer.from(imagePart.inlineData.data, 'base64');
    } catch (e) {
      lastErr = e;
      console.warn(`Image model "${id}" failed:`, e?.message || e);
    }
  }
  const err = new Error('All image models failed');
  err.cause = lastErr;
  throw err;
}

async function uploadImageBufferToSupabase(
  buf,
  { table, filename = 'ai.png', contentType = 'image/png' } = {}
) {
  const bucket =
    table === TABLES.products
      ? process.env.SUPABASE_BUCKET_PRODUCTS || 'images'
      : process.env.SUPABASE_BUCKET_EXCLUSIVE || 'exclusiveproduct-images';

  const folder = table === TABLES.products ? 'products' : 'exclusive-products';
  const key = `${folder}/${Date.now()}-${Math.random().toString(36).slice(2)}-${filename}`;

  const { error: upErr } = await supabase.storage
    .from(bucket)
    .upload(key, buf, { contentType, upsert: true });
  if (upErr) throw upErr;

  const { data: pub } = supabase.storage.from(bucket).getPublicUrl(key);
  return pub.publicUrl;
}

/* Title SVG fallback */
function makeTextCardSvg(title = 'Product', subtitle = '') {
  const brand = getBrandStyle?.({ name: title }) || null;
  const [c1, c2] = (brand?.palette?.length >= 2) ? brand.palette.slice(0, 2) : ['#0ea5e9', '#7c3aed'];

  const esc = s => String(s || '').replace(/[<&>]/g, c => ({'<':'&lt;','>':'&gt;','&':'&amp;'}[c]));
  const tRaw = (title || '').trim();
  const t = esc(tRaw).slice(0, 48);
  const sub = esc(subtitle).slice(0, 64);

  const sizeGuess = Math.max(72, Math.min(140, 140 - Math.max(0, tRaw.length - 10) * 4));
  return `
<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="1024">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="${c1}"/><stop offset="100%" stop-color="${c2}"/>
    </linearGradient>
  </defs>
  <rect width="1024" height="1024" fill="url(#g)"/>
  <text x="72" y="560" fill="#fff" font-family="Segoe UI, Roboto, Arial"
        font-size="${sizeGuess}" font-weight="900"
        textLength="880" lengthAdjust="spacingAndGlyphs">${t}</text>
  ${sub ? `<text x="72" y="640" fill="rgba(255,255,255,0.92)" font-family="Segoe UI, Roboto, Arial"
        font-size="48" textLength="880" lengthAdjust="spacingAndGlyphs">${sub}</text>` : ''}
  <text x="96%" y="96%" fill="rgba(255,255,255,0.28)" font-family="Segoe UI, Roboto, Arial"
        font-size="40" text-anchor="end" stroke="#000" stroke-opacity=".25" stroke-width="2"
        transform="rotate(-8 900 900)">Harshportal</text>
</svg>`.trim();
}

/* --- deterministic brand tile fallback using refs (kept) --- */
const BRAND_DOMAIN_MAP = {
  'spotify': 'spotify.com', 'spotify premium': 'spotify.com',
  'netflix': 'netflix.com',
  'youtube': 'youtube.com', 'yt premium': 'youtube.com',
  'disney': 'disneyplus.com',
  'prime video': 'primevideo.com', 'prime': 'primevideo.com',
  'descript': 'descript.com',
  'mobbin': 'mobbin.com',
  'raycast': 'raycast.com',
  'warp': 'warp.dev',
  'bolt.new': 'bolt.new', 'boltnew': 'bolt.new',
  'lovable': 'lovable.dev', 'lovable.dev': 'lovable.dev',
  'gamma': 'gamma.app',
  'n8n': 'n8n.io',
  'magic patterns': 'magicpatterns.design',
  'wispr': 'wispr.cc', 'wispr flow': 'wispr.cc',
  'windows': 'microsoft.com', 'windows 10': 'microsoft.com', 'windows 11': 'microsoft.com',
  'v0': 'v0.dev', 'v0.dev': 'v0.dev', 'vercel v0': 'v0.dev',
  'vercel': 'vercel.com',
  'tradingview': 'tradingview.com',
  'tradingview essential': 'tradingview.com',
};

function resolveBrandDomain(name = '') {
  const n = String(name).toLowerCase().trim();
  for (const k of Object.keys(BRAND_DOMAIN_MAP)) {
    if (n.includes(k)) return BRAND_DOMAIN_MAP[k];
  }
  const slug = n.replace(/[^a-z0-9]/g,'');
  if (slug.length >= 3) return `${slug}.com`;
  return null;
}

async function fetchBrandRefs(name) {
  const refs = [];
  const domain = resolveBrandDomain(name);

  if (domain) {
    refs.push(`https://logo.clearbit.com/${domain}?size=1024`);
    try {
      const og = await getOG(`https://${domain}`);
      if (og?.image) refs.push(og.image);
    } catch {}
    refs.push(`https://icons.duckduckgo.com/ip3/${domain}.ico`);
    refs.push(`https://unavatar.io/${domain}`);
  }

  try {
    const cleaned = String(name).replace(/\b(subscription|premium|pro|account|key)\b/ig,'').trim();
    const q = encodeURIComponent(cleaned);
    const r = await fetch(`https://en.wikipedia.org/api/rest_v1/page/summary/${q}`);
    if (r.ok) {
      const j = await r.json();
      if (j?.thumbnail?.source) refs.push(j.thumbnail.source);
      if (j?.originalimage?.source) refs.push(j.originalimage.source);
    }
  } catch {}

  const uniq = Array.from(new Set(
    refs.filter(Boolean).filter(u => /^https?:\/\//i.test(u)).filter(u => !/\.ico(\?|$)/i.test(u))
  )).slice(0, 3);

  const blobs = [];
  for (const u of uniq) {
    const b = await fetchAsBase64(u).catch(() => null);
    if (b) blobs.push(b);
  }
  return blobs;
}

async function composeTileWithLogo({ bgBuf, logoRef, brandName }) {
  if (!_sharp) throw new Error('sharp not available');
  const logoBuf = Buffer.from(logoRef.b64, 'base64');

  const canvas = await _sharp(bgBuf).resize(1024, 1024, { fit: 'cover' }).toBuffer();
  const logoPng = await _sharp(logoBuf).resize({ width: 560, withoutEnlargement: true }).png().toBuffer();

  const safe = (s='').replace(/[<&>]/g,c=>({ '<':'&lt;','>':'&gt;','&':'&amp;' }[c]));
  const title = safe(brandName || 'Product').slice(0,48);
  const textSvg = Buffer.from(`
    <svg xmlns="http://www.w3.org/2000/svg" width="1024" height="300">
      <style>.t{ font-family: "Inter","Segoe UI",Roboto,Arial; font-weight: 800; }</style>
      <text x="512" y="230" text-anchor="middle" class="t" font-size="96" fill="#fff">${title}</text>
    </svg>
  `);

  let composed = await _sharp(canvas)
    .composite([{ input: logoPng, top: 260, left: Math.round((1024-560)/2) }])
    .png().toBuffer();

  composed = await tryCompositeSvg(composed, textSvg, { top: 724, left: 0 });

  const withWm = await addWatermarkToBuffer(composed, 'Harshportal') || composed;
  return withWm;
}

/* ensure image for product */
async function ensureImageForProduct(prod, table, style = 'neo') {
  if (prod?.image && String(prod.image).trim()) return prod.image;

  // 1) Imagen 3 generation using your prompt
  try {
    const prompt = await buildImagePrompt(prod);
    const buf = await generateProductImageBytes({ prompt });

    let out = buf;
    if (_sharp) {
      const wm = await addWatermarkToBuffer(buf, 'Harshportal');
      if (wm) out = wm;
    }
    return await uploadImageBufferToSupabase(out, { table, filename: 'ai.png', contentType: 'image/png' });
  } catch (e) {
    console.warn('AI image generation failed (Gemini/Imagen). Cause:', e?.message || e);
  }

  // 2) Deterministic brand tile fallback
  try {
    const refs = await fetchBrandRefs(prod?.name || '');
    const ref = refs[0];
    if (ref && _sharp) {
      const base = await _sharp({ create: { width:1024, height:1024, channels:3, background:'#111' } }).png().toBuffer();
      const composed = await composeTileWithLogo({ bgBuf: base, logoRef: ref, brandName: shortBrandName(prod) });
      return await uploadImageBufferToSupabase(composed, { table, filename: 'brand.png', contentType: 'image/png' });
    }
  } catch (e) {
    console.warn('composeBrandTile failed:', e?.message || e);
  }

  // 3) SVG fallback
  const svg = makeTextCardSvg(prod?.name || 'Digital Product', prod?.plan || prod?.subcategory || '');
  const svgBuf = Buffer.from(svg, 'utf8');
  return uploadImageBufferToSupabase(svgBuf, { table, filename: 'ai.svg', contentType: 'image/svg+xml' });
}

/* --------------------- /style command --------------------- */
const STYLE_KEYS = Object.keys(STYLE_THEMES);
bot.command('style', (ctx) => {
  if (!isAdmin(ctx)) return;
  const current = ctx.session.style || 'neo';
  const rows = STYLE_KEYS.map(k => [Markup.button.callback(
    (k === current ? '✅ ' : '') + k, `style_set_${k}`
  )]);
  ctx.reply('Choose image style theme:', Markup.inlineKeyboard(rows));
});
bot.action(/^style_set_(.+)$/, (ctx) => {
  if (!isAdmin(ctx)) return ctx.answerCbQuery();
  const pick = ctx.match[1];
  if (!STYLE_THEMES[pick]) return ctx.answerCbQuery('Unknown style');
  ctx.session.style = pick;
  ctx.answerCbQuery('Style updated');
  ctx.reply(escapeMd(`🎨 Style set to ${pick}`), { parse_mode: 'MarkdownV2' });
});

/* --------------- keyboards / messages --------------- */
const kbConfirm = Markup.inlineKeyboard([
  [Markup.button.callback('✅ Save', 'save'), Markup.button.callback('✏️ Edit', 'edit')],
  [Markup.button.callback('❌ Cancel', 'cancel')],
]);

const kbGenImage = Markup.inlineKeyboard([
  [Markup.button.callback('🎨 Generate image', 'gen_img_yes')],
  [Markup.button.callback('Skip without image', 'gen_img_no')]
]);

function kbAfterTask() {
  return Markup.inlineKeyboard([
    [Markup.button.callback('➕ Add another', 'again_smartadd')],
    [Markup.button.callback('📋 List', 'again_list'), Markup.button.callback('✏️ Update', 'again_update')],
    [Markup.button.callback('🎨 Style', 'again_style'), Markup.button.callback('🆕 New', 'again_new')],
    [Markup.button.callback('🏁 Done', 'again_done')],
  ]);
}

bot.action('again_smartadd', (ctx)=>{ ctx.answerCbQuery(); ctx.session.mode=null; ctx.session.smart=null; ctx.reply('Use /smartadd to add another.'); });
bot.action('again_list',     (ctx)=>{ ctx.answerCbQuery(); ctx.reply('Use /list to view latest.'); });
bot.action('again_update',   (ctx)=>{ ctx.answerCbQuery(); ctx.reply('Use /update <id>'); });
bot.action('again_style',    (ctx)=>{ ctx.answerCbQuery(); ctx.reply('Use /style to pick the image theme.'); });
bot.action('again_new',      (ctx)=>{ ctx.answerCbQuery(); ctx.reply('Use /table then type *products* or *exclusive*', { parse_mode: 'Markdown' }); });
bot.action('again_done',     (ctx)=>{ ctx.answerCbQuery(); ctx.reply('All set. ✅'); });

const kbEditWhich = (table) => {
  const common = ['name','plan','validity','price','description','tags','image'];
  const pro = ['originalPrice','stock','category','subcategory','gradient','features'];
  const fields = table === TABLES.products ? [...common, ...pro] : common;
  const rows = [];
  for (let i=0;i<fields.length;i+=3){
    rows.push(fields.slice(i,i+3).map(f => Markup.button.callback(f, `edit_${f}`)));
  }
  rows.push([Markup.button.callback('⬅️ Back', 'back_review')]);
  return Markup.inlineKeyboard(rows);
};

function reviewMessage(prod, ai, table) {
  const tags = uniqMerge(prod.tags || [], ai.tags || []);
  const parts = [];
  parts.push(`*Review before save*`);
  parts.push(`*Name:* ${escapeMd(prod.name || '-')}`);
  if (ok(prod.plan)) parts.push(`*Plan:* ${escapeMd(prod.plan)}`);
  if (ok(prod.validity)) parts.push(`*Validity:* ${escapeMd(prod.validity)}`);
  parts.push(`*Price:* ${escapeMd(prod.price ? `₹${prod.price}` : '-')}`);
  parts.push(`*Description:* ${escapeMd(prod.description || ai.description || '-')}`);
  parts.push(`*Tags:* ${escapeMd(tags.join(', ') || '-')}`);

  if (table === TABLES.products) {
    parts.push(`*MRP:* ${escapeMd(prod.originalPrice ?? '-')}`);
    parts.push(`*Stock:* ${escapeMd(prod.stock ?? '-')}`);
    parts.push(`*Category:* ${escapeMd(ai.category || '-')}`);
    parts.push(`*Subcategory:* ${escapeMd(ai.subcategory || '-')}`);
  }

  parts.push(`*Image:* ${escapeMd(prod.image ? 'Attached' : '-')}`);
  return parts.join('\n');
}

/* --------------------- middleware --------------------- */
bot.use(session());
bot.use(async (ctx, next) => {
  if (!isAdmin(ctx)) return;
  if (!ctx.session) ctx.session = {};
  return next();
});

/* --------------------- commands --------------------- */
bot.start(async (ctx) => {
  if (!isAdmin(ctx)) return;
  ctx.session = {};
  await ctx.reply(`Choose a table:\n• *products*\n• *exclusive*`, { parse_mode: 'Markdown' });
});

bot.command('table', (ctx) => {
  if (!isAdmin(ctx)) return;
  ctx.session = {};
  ctx.reply(`Type *products* or *exclusive*`, { parse_mode: 'Markdown' });
});

bot.command('list', async (ctx) => {
  if (!isAdmin(ctx)) return;
  if (!ctx.session.table) return ctx.reply('Choose table first with /table');

  const table = ctx.session.table;

  const selectCols =
    table === TABLES.exclusive
      ? 'id,name,price,is_active,created_at'
      : 'id,name,price,is_active';

  const orderCol = table === TABLES.exclusive ? 'created_at' : 'id';

  const { data, error } = await supabase
    .from(table)
    .select(selectCols)
    .order(orderCol, { ascending: false })
    .limit(12);

  if (error) {
    console.error('List error:', error);
    return ctx.reply(`DB error: ${error.message}`);
  }

  const items = data || [];
  if (!items.length) return ctx.reply('No items yet.');

  const msg = items
    .map((r, i) => {
      const p = Number(r.price || 0).toLocaleString('en-IN');
      const status = r.is_active ? '✅' : '⛔️';
      return `${i + 1}. ${r.name} — ₹${p} — ${status} (id: ${r.id})`;
    })
    .join('\n');

  ctx.reply('Latest:\n' + msg);
});

/* --------- ADD FLOWS --------- */
bot.command('addproduct', (ctx) => {
  if (!isAdmin(ctx)) return;
  if (!ctx.session.table) return ctx.reply('First choose a table: *products* or *exclusive*', { parse_mode: 'Markdown' });
  ctx.session.mode = 'manual';
  ctx.session.form = { step: 0, prod: {} };
  ctx.reply('Enter *Name*:', { parse_mode: 'Markdown' });
});

bot.command('addproductgemini', (ctx) => {
  if (!isAdmin(ctx)) return;
  if (!ctx.session.table) return ctx.reply('First choose a table: *products* or *exclusive*', { parse_mode: 'Markdown' });
  ctx.session.mode = 'gemini';
  ctx.session.stage = 'paste';
  ctx.session.products = [];
  ctx.session.index = 0;
  ctx.session.await = null;
  ctx.reply('Paste your list. I’ll detect products automatically.');
});

bot.command('smartadd', (ctx) => {
  if (!isAdmin(ctx)) return;
  if (!ctx.session.table) return ctx.reply('First choose a table: *products* or *exclusive*', { parse_mode: 'Markdown' });
  ctx.session.mode = 'smart';
  ctx.session.await = 'blob';
  ctx.reply('Send the product text (can be messy). You may also attach a photo or include a URL.');
});

bot.command('toggle', async (ctx) => {
  if (!isAdmin(ctx)) return;
  const id = (ctx.message.text.split(' ')[1] || '').trim();
  if (!id) return ctx.reply('Usage: /toggle <id>');
  if (!ctx.session.table) return ctx.reply('Choose table first with /table');

  const table = ctx.session.table;
  const { data, error } = await supabase.from(table).select('is_active').eq('id', id).maybeSingle();
  if (error || !data) return ctx.reply('Not found.');

  const { error: upErr } = await supabase.from(table)
    .update({ is_active: !data.is_active })
    .eq('id', id);

  if (upErr) return ctx.reply(`❌ Toggle failed: ${upErr.message}`);
  ctx.reply(`Toggled id ${id} to ${!data.is_active ? '✅ active' : '⛔️ inactive'}.`);
});

/* --------------------- text router --------------------- */
bot.on('text', async (ctx, next) => {
  if (!isAdmin(ctx)) return;

  if (!ctx.session.table) {
    const t = ctx.message.text.trim().toLowerCase();
    if (t === 'products' || t === 'exclusive') {
      ctx.session.table = t === 'exclusive' ? TABLES.exclusive : TABLES.products;
      return ctx.reply(
        escapeMd(
          `Table set to ${ctx.session.table}.
Commands:
• /addproduct (manual)
• /addproductgemini (AI bulk)
• /smartadd (one-shot messy add)
• /list  • /toggle <id>  • /update <id>`
        ),
        { parse_mode: 'MarkdownV2' }
      );
    }
    return ctx.reply('Type *products* or *exclusive*', { parse_mode: 'Markdown' });
  }

  // manual wizard
  if (ctx.session.mode === 'manual') {
    const table = ctx.session.table;
    const common = ['name','plan','validity','price','description','tags'];
    const pro = ['originalPrice','stock','category','subcategory','gradient','features'];
    const fields = table === TABLES.products ? [...common, ...pro] : common;

    const step = ctx.session.form.step;
    const prevKey = fields[step - 1];
    if (step > 0 && prevKey) {
      let value = ctx.message.text.trim();
      if (['price','originalPrice','stock'].includes(prevKey)) value = parsePrice(value);
      if (prevKey === 'tags' || prevKey === 'gradient') value = value.split(';').map(v => v.trim()).filter(Boolean);
      if (prevKey === 'features') {
        if (value.toLowerCase() === 'skip') value = [];
        else { try { value = JSON.parse(value); } catch { return ctx.reply('Features must be valid JSON or "skip"'); } }
      }
      ctx.session.form.prod[prevKey] = value;
    }

    if (step >= fields.length) {
      ctx.session.mode = 'manual-photo';
      return ctx.reply('Now send the product *image* (photo upload, not file).', { parse_mode: 'Markdown' });
    }

    const key = fields[step];
    ctx.session.form.step++;
    return ctx.reply(`Enter *${key}*${key==='features' ? ' as JSON (or "skip")' : key==='tags' ? ' (separate by ; )' : ''}:`, { parse_mode: 'Markdown' });
  }

  // smart add
  if (ctx.session.mode === 'smart' && ctx.session.await === 'blob') {
    ctx.session.smart = { text: ctx.message.text, photo: null, og: {}, prod: {} };
    const urls = findUrls(ctx.message.text);
    if (urls.length) ctx.session.smart.og = await getOG(urls[0]);

    const rough = await extractFromFreeform(ctx.message.text);
    const enriched = await enrichWithAI(rough, ctx.message.text, ctx.session.smart.og);

    ctx.session.smart.prod = {
      name: enriched.name || rough.name,
      plan: enriched.plan !== 'unknown' ? enriched.plan : rough.plan,
      validity: enriched.validity !== 'unknown' ? enriched.validity : rough.validity,
      price: enriched.price || rough.price || null,
      description: enriched.description || rough.description || '',
      tags: uniqMerge(enriched.tags),
      image: null,
      og_image: ctx.session.smart.og?.image || null
    };
    ctx.session.mode = 'smart-photo';
    return ctx.reply('If you have a product *image*, send it now. Or type "skip".', { parse_mode: 'Markdown' });
  }

  // gemini bulk continues...
  if (ctx.session.mode === 'gemini') {
    if (ctx.session.stage === 'paste') {
      await ctx.reply('⏳ Detecting products…');
      const items = await extractManyFromFreeform(ctx.message.text);
      if (!items.length) return ctx.reply('Could not detect products. Try again with one-per-line.', { parse_mode:'Markdown' });

      for (const it of items) {
        const ai = await enrichWithAI(it, it._txt || ctx.message.text, {});
        it._ai = ai;
        it.name = ai.name || it.name;
        it.description = ai.description || it.description;
        it.price = it.price ?? (ai.price || null);
        it.tags = uniqMerge(it.tags, ai.tags);
      }

      ctx.session.stage = 'step';
      ctx.session.products = items;
      ctx.session.index = 0;
      ctx.session.await = null;
      await ctx.reply(`Detected *${items.length}* item(s). I’ll ask for any missing fields.`, { parse_mode: 'Markdown' });
      return handleBulkStep(ctx);
    }

    if (ctx.session.stage === 'step' && ctx.session.await) {
      const idx = ctx.session.index || 0;
      const prod = ctx.session.products?.[idx];
      if (!prod) return ctx.reply('No active item.');

      const valRaw = ctx.message.text.trim();
      const want = ctx.session.await;

      if (want === 'price') {
        prod.price = parsePrice(valRaw);
        if (!prod.price) return ctx.reply('Enter a valid price (number).');
        ctx.session.await = null;
        return handleBulkStep(ctx);
      } else if (want === 'originalPrice') {
        prod.originalPrice = parsePrice(valRaw);
        if (!prod.originalPrice) return ctx.reply('Enter a valid MRP (number).');
        ctx.session.await = null;
        return handleBulkStep(ctx);
      } else if (want === 'stock') {
        prod.stock = parsePrice(valRaw);
        if (typeof prod.stock !== 'number') return ctx.reply('Enter a valid stock count (number).');
        ctx.session.await = null;
        return handleBulkStep(ctx);
      } else if (want === 'image') {
        if (/^generate$/i.test(valRaw)) {
          try {
            await ctx.reply('🎨 Generating image…');
            const ai = prod._ai || await enrichWithAI(prod, prod._txt || '', {});
            const url = await ensureImageForProduct(
              { ...prod, ...ai }, ctx.session.table, ctx.session.style || 'neo'
            );
            prod.image = url;
            ctx.session.await = null;
            return handleBulkStep(ctx);
          } catch (e) {
            console.error('bulk image generate failed:', e);
            return ctx.reply('⚠️ Could not generate. Upload a photo or paste an image URL, or type "skip".');
          }
        } else if (/^skip$/i.test(valRaw)) {
          prod.image = null;
          ctx.session.await = null;
          return handleBulkStep(ctx);
        } else if (/^https?:\/\//i.test(valRaw)) {
          try {
            prod.image = await rehostToSupabase(valRaw, 'prod.jpg', ctx.session.table);
            ctx.session.await = null;
            return handleBulkStep(ctx);
          } catch {
            return ctx.reply('Could not fetch that URL. Upload a photo or type "generate" or "skip".');
          }
        } else {
          return ctx.reply('Upload a photo, paste an image URL, type "generate", or "skip".');
        }
      }
    }

    if (!ctx.session.review) {
      return handleBulkStep(ctx);
    }
  }

  if (ctx.session.awaitEdit) {
    await applyInlineEdit(ctx);
    return;
  }

  return next();
});

/* --------------------- photo handlers --------------------- */
async function processIncomingImage(ctx, fileId, filenameHint = 'prod.jpg') {
  const href = await tgFileUrl(fileId);

  // 1) EDIT MODE: set new image
  if (ctx.session.awaitEdit === 'image' && ctx.session.review) {
    try {
      const { table } = ctx.session.review;
      const imgUrl = await rehostToSupabase(href, filenameHint, table);
      ctx.session.review.prod.image = imgUrl;
      ctx.session.awaitEdit = null;
      const { prod, ai } = ctx.session.review;
      await ctx.replyWithMarkdownV2(reviewMessage(prod, ai, table), kbConfirm);
    } catch (e) {
      console.error('edit image upload error:', e);
      await ctx.reply(`❌ Could not update image. ${e?.message || ''}\nPaste a direct image URL if it keeps failing.`);
    }
    return;
  }

  // 2) MANUAL WIZARD PHOTO
  if (ctx.session.mode === 'manual-photo') {
    try {
      const imgUrl = await rehostToSupabase(href, filenameHint, ctx.session.table);
      const prod = ctx.session.form.prod;
      prod.image = imgUrl;
      const ai = await enrichWithAI(prod, '', {});
      ctx.session.review = { prod, ai, table: ctx.session.table };
      await ctx.replyWithMarkdownV2(reviewMessage(prod, ai, ctx.session.table), kbConfirm);
    } catch (e) {
      console.error('manual photo upload error:', e);
      await ctx.reply(`❌ Could not upload that image. ${e?.message || ''}`);
    }
    return;
  }

  // 3) SMART ADD PHOTO
  if (ctx.session.mode === 'smart-photo') {
    try {
      ctx.session.smart.photo = await rehostToSupabase(href, filenameHint, ctx.session.table);
      const prod = { ...ctx.session.smart.prod, image: ctx.session.smart.photo };
      const ai = await enrichWithAI(prod, ctx.session.smart.text, ctx.session.smart.og);
      ctx.session.review = { prod: { ...prod, tags: uniqMerge(prod.tags, ai.tags) }, ai, table: ctx.session.table };
      ctx.session.mode = null;
      await ctx.replyWithMarkdownV2(reviewMessage(ctx.session.review.prod, ai, ctx.session.table), kbConfirm);
    } catch (e) {
      console.error('smart photo upload error:', e);
      await ctx.reply(`❌ Could not upload that image. ${e?.message || ''}`);
    }
    return;
  }

  // 4) GEMINI BULK: waiting for an image
  if (ctx.session.mode === 'gemini' && ctx.session.stage === 'step' && ctx.session.await === 'image') {
    try {
      const imgUrl = await rehostToSupabase(href, filenameHint, ctx.session.table);
      const idx = ctx.session.index || 0;
      if (ctx.session.products?.[idx]) ctx.session.products[idx].image = imgUrl;
      ctx.session.await = null;
      return handleBulkStep(ctx);
    } catch (e) {
      console.error('bulk photo upload error:', e);
      await ctx.reply(`❌ Could not upload that image. ${e?.message || ''}`);
    }
    return;
  }
}

bot.on('photo', async (ctx) => {
  if (!isAdmin(ctx)) return;
  const photos = ctx.message?.photo || [];
  const fileId = photos.at(-1)?.file_id;
  if (!fileId) return;
  await processIncomingImage(ctx, fileId, 'prod.jpg');
});

bot.on('document', async (ctx) => {
  if (!isAdmin(ctx)) return;
  const doc = ctx.message?.document;
  if (!doc || !/^image\//i.test(doc.mime_type || '')) return;
  await processIncomingImage(ctx, doc.file_id, doc.file_name || 'prod.jpg');
});

/* If smart add user types "skip" instead of photo -> ask to generate */
bot.hears(/^skip$/i, async (ctx) => {
  if (!isAdmin(ctx)) return;
  if (ctx.session.mode === 'smart-photo') {
    const prod = { ...ctx.session.smart.prod, image: null };
    ctx.session.smart.prod = prod;
    await ctx.reply('No image provided. Do you want me to generate a product image for you?', kbGenImage);
  }
});

/* --------------------- callbacks --------------------- */
bot.action('cancel', (ctx) => {
  ctx.answerCbQuery();
  ctx.session.review = null;
  ctx.session.mode = null;
  ctx.reply('Cancelled.', kbAfterTask());
});

bot.action('save', async (ctx) => {
  if (!ctx.session.review) return ctx.answerCbQuery('Nothing to save');
  ctx.answerCbQuery('Saving…');

  const { prod, ai, table, updateId } = ctx.session.review;
  const baseTags = uniqMerge(prod.tags, ai.tags);

  if (updateId) {
    if (prod.image && /^https?:\/\//i.test(prod.image) && !isSupabasePublicUrl(prod.image)) {
      try { prod.image = await ensureHostedInSupabase(prod.image, table); } catch {}
    }

    const idKey  = updateId;
    const imgCol = table === TABLES.products ? 'image' : 'image_url';

    const { error: imgWriteErr } = await supabase
      .from(table)
      .update({ [imgCol]: prod.image || null })
      .eq('id', idKey);
    if (imgWriteErr) {
      await ctx.reply(`❌ Image update failed: ${imgWriteErr.message}`);
      return;
    }

    const { data: fresh, error: readErr } = await supabase
      .from(table)
      .select(`id, ${imgCol}`).eq('id', idKey)
      .maybeSingle();
    if (readErr) {
      console.log('[image:readback] error:', readErr);
    } else {
      console.log('[image:readback]', table, idKey, '->', imgCol, fresh?.[imgCol]);
    }

    const mergedTags = uniqMerge(prod.tags, ai.tags);
    const rest = table === TABLES.products
      ? {
          name: prod.name,
          plan: prod.plan || null,
          validity: prod.validity || null,
          price: prod.price || null,
          originalPrice: prod.originalPrice || null,
          description: prod.description || ai.description || null,
          category: normalizeCategory({ ...prod, ...ai }, ai.category),
          subcategory: ai.subcategory || null,
          stock: prod.stock || null,
          tags: mergedTags,
          features: ai.features || [],
          gradient: ai.gradient || [],
        }
      : {
          name: prod.name,
          description: prod.description || ai.description || null,
          price: prod.price || null,
          tags: mergedTags,
        };

    const { error: restErr } = await supabase.from(table).update(rest).eq('id', idKey);
    if (restErr) {
      await ctx.reply(`❌ Update failed: ${restErr.message}`);
      return;
    }

    await ctx.reply(escapeMd(`✅ Updated ${table}`), { parse_mode: 'MarkdownV2' });
    await ctx.reply('What next?', kbAfterTask());

    ctx.session.review = null;
    if (ctx.session.mode === 'gemini' && ctx.session.stage === 'step' && Array.isArray(ctx.session.products)) {
      ctx.session.index = (ctx.session.index || 0) + 1;
      return handleBulkStep(ctx);
    }
    ctx.session.mode = null;
    return;
  }

  // INSERT new (dupe guard)
  const { data: dup } = await supabase
    .from(table)
    .select('id')
    .eq('name', prod.name)
    .eq('price', prod.price)
    .maybeSingle();

  if (dup) {
    await ctx.reply(`⚠️ A product with same name & price exists. Use /toggle ${dup.id} to activate/deactivate.`);
    ctx.session.review = null;
    await ctx.reply('What next?', kbAfterTask());
    return;
  }

  if (table === TABLES.products) {
    const insert = {
      name: prod.name,
      plan: prod.plan || null,
      validity: prod.validity || null,
      price: prod.price || null,
      originalPrice: prod.originalPrice || null,
      description: prod.description || ai.description || null,
      category: normalizeCategory({ ...prod, ...ai }, ai.category),
      subcategory: ai.subcategory || null,
      stock: prod.stock || null,
      tags: baseTags,
      features: ai.features || [],
      gradient: ai.gradient || [],
      image: prod.image || null,
      is_active: true,
    };
    const { error } = await supabase.from(TABLES.products).insert([insert]);
    if (error) {
      await ctx.reply(`❌ Failed: ${error.message}`);
    } else {
      await ctx.reply(escapeMd('✅ Added to products'), { parse_mode: 'MarkdownV2' });
      await ctx.reply('What next?', kbAfterTask());
    }
  } else {
    const insert = {
      name: prod.name,
      description: prod.description || ai.description || null,
      price: prod.price || null,
      image_url: prod.image || null,
      is_active: true,
      tags: baseTags,
    };
    const { error } = await supabase.from(TABLES.exclusive).insert([insert]);
    if (error) {
      await ctx.reply(`❌ Failed: ${error.message}`);
    } else {
      await ctx.reply(escapeMd('✅ Added to exclusive_products'), { parse_mode: 'MarkdownV2' });
      await ctx.reply('What next?', kbAfterTask());
    }
  }

  ctx.session.review = null;
  if (ctx.session.mode === 'gemini' && ctx.session.stage === 'step' && Array.isArray(ctx.session.products)) {
    return handleBulkStep(ctx);
  }
  ctx.session.mode = null;
  return;
});

bot.action('edit', (ctx) => {
  if (!ctx.session.review) return ctx.answerCbQuery();
  ctx.answerCbQuery();
  const table = ctx.session.review.table;
  ctx.reply('Which field do you want to edit?', kbEditWhich(table));
});

bot.action(/^edit_(.+)$/, (ctx) => {
  const field = ctx.match[1];
  ctx.session.awaitEdit = field;
  ctx.answerCbQuery();
  ctx.reply(`Send new value for *${field}*`, { parse_mode: 'Markdown' });
});

bot.action('back_review', (ctx) => {
  if (!ctx.session.review) return ctx.answerCbQuery();
  ctx.answerCbQuery();
  const { prod, ai, table } = ctx.session.review;
  ctx.replyWithMarkdownV2(reviewMessage(prod, ai, table), kbConfirm);
});

bot.action('gen_img_no', async (ctx) => {
  ctx.answerCbQuery();
  if (!ctx.session.smart?.prod) return ctx.reply('No active product.');
  const prod = { ...ctx.session.smart.prod, image: null };
  const ai = await enrichWithAI(prod, ctx.session.smart.text, ctx.session.smart.og);
  ctx.session.review = { prod: { ...prod, tags: uniqMerge(prod.tags, ai.tags) }, ai, table: ctx.session.table };
  ctx.session.mode = null;
  return ctx.replyWithMarkdownV2(reviewMessage(ctx.session.review.prod, ai, ctx.session.table), kbConfirm);
});

bot.action('gen_img_yes', async (ctx) => {
  if (!ctx.session.smart?.prod) { ctx.answerCbQuery(); return ctx.reply('No active product.'); }
  ctx.answerCbQuery('Generating…');
  try {
    await ctx.reply('🎨 Generating image… this can take ~10–20s.');
    const table = ctx.session.table;
    const prodBase = { ...ctx.session.smart.prod };

    const ai = await enrichWithAI(prodBase, ctx.session.smart.text, ctx.session.smart.og);
    const prodForPrompt = { ...prodBase, ...ai, category: normalizeCategory({ ...prodBase, ...ai }, ai.category) };

    const url = await ensureImageForProduct(prodForPrompt, table, ctx.session.style || 'neo');
    if (!url) throw new Error('No image URL');

    const prod = { ...prodBase, image: url };
    ctx.session.review = { prod: { ...prod, tags: uniqMerge(prod.tags, ai.tags) }, ai, table };
    ctx.session.mode = null;

    await ctx.reply('✅ Image generated.');
    return ctx.replyWithMarkdownV2(reviewMessage(ctx.session.review.prod, ai, table), kbConfirm);
  } catch (e) {
    console.error('AI image gen failed:', e);
    await ctx.reply('⚠️ Could not generate an image (model not enabled or quota). Proceeding without image.');
    const table = ctx.session.table;
    const prod = { ...ctx.session.smart.prod, image: null };
    const ai = await enrichWithAI(prod, ctx.session.smart.text, ctx.session.smart.og);
    ctx.session.review = { prod: { ...prod, tags: uniqMerge(prod.tags, ai.tags) }, ai, table };
    ctx.session.mode = null;
    return ctx.replyWithMarkdownV2(reviewMessage(ctx.session.review.prod, ai, table), kbConfirm);
  }
});

bot.action('gen_img', async (ctx) => {
  if (!ctx.session.smart && !ctx.session.review) return ctx.answerCbQuery('Nothing to do');
  ctx.answerCbQuery('Generating image…');

  const table = ctx.session.table;
  const base  = (ctx.session.review?.prod) || (ctx.session.smart?.prod) || {};

  try {
    const url = await ensureImageForProduct(base, table, ctx.session.style || 'neo');
    if (ctx.session.review?.prod) ctx.session.review.prod.image = url;
    if (ctx.session.smart?.prod) ctx.session.smart.prod.image = url;

    const prod = (ctx.session.review?.prod) || (ctx.session.smart?.prod);
    const ai   = ctx.session.review?.ai || {};
    const tbl  = ctx.session.review?.table || table;

    await ctx.reply('🖼️ AI image generated and attached.');
    await ctx.replyWithMarkdownV2(reviewMessage(prod, ai, tbl), kbConfirm);
  } catch (e) {
    console.error('AI image gen failed:', e);
    await ctx.reply('❌ Could not generate image right now.');
  }
});

bot.action('skip_img', async (ctx) => {
  ctx.answerCbQuery();
  const prod = (ctx.session.review?.prod) || (ctx.session.smart?.prod);
  const ai   = ctx.session.review?.ai || {};
  const tbl  = ctx.session.review?.table || ctx.session.table;
  await ctx.replyWithMarkdownV2(reviewMessage(prod, ai, tbl), kbConfirm);
});

/* --------------------- inline edit apply --------------------- */
async function applyInlineEdit(ctx) {
  const field = ctx.session.awaitEdit;
  const rvw = ctx.session.review;
  if (!rvw) return;

  const { prod, ai } = rvw;
  let val = ctx.message.text.trim();

  if (['price','originalPrice','stock'].includes(field)) val = parsePrice(val);
  if (['tags','gradient'].includes(field)) val = val.split(';').map(x => x.trim()).filter(Boolean);
  if (field === 'features') {
    try { val = val === 'skip' ? [] : JSON.parse(val); }
    catch { return ctx.reply('Features must be JSON or "skip"'); }
  }

  if (field === 'image' && val.toLowerCase() === 'generate') {
    try {
      await ctx.reply('🎨 Generating image…');
      const url = await ensureImageForProduct({ ...prod, ...ai }, rvw.table, ctx.session.style || 'neo');
      prod.image = url;
    } catch (e) {
      console.error('edit image generate failed:', e);
      return ctx.reply('⚠️ Could not generate an image. Paste an image URL or upload a photo.');
    }
    ctx.session.awaitEdit = null;
    return ctx.replyWithMarkdownV2(reviewMessage(prod, ai, rvw.table), kbConfirm);
  }

  if (field === 'image' && /^https?:\/\//i.test(val)) {
    prod.image = await ensureHostedInSupabase(val, rvw.table);
  } else if (field === 'category') {
    ai.category = normalizeCategory({ ...prod, ...ai }, val);
  } else if (['name','plan','validity','price','description','originalPrice','stock'].includes(field)) {
    prod[field] = val;
  } else {
    ai[field] = val;
  }

  ctx.session.awaitEdit = null;
  await ctx.replyWithMarkdownV2(reviewMessage(prod, ai, rvw.table), kbConfirm);
}

/* --------------------- /update <id> --------------------- */
bot.command('update', async (ctx) => {
  if (!isAdmin(ctx)) return;
  const id = (ctx.message.text.split(' ')[1] || '').trim();
  if (!id) return ctx.reply('Usage: /update <id>');
  const table = ctx.session.table || TABLES.exclusive;
  const sel = table === TABLES.products
    ? 'id,name,plan,validity,price,originalPrice,description,category,subcategory,stock,tags,gradient,features,image,is_active'
    : 'id,name,description,price,image_url,tags,is_active';
  const { data, error } = await supabase.from(table).select(sel).eq('id', id).maybeSingle();
  if (error || !data) return ctx.reply('Not found.');
  const prod = table === TABLES.products
    ? { ...data, image: data.image }
    : { name: data.name, description: data.description, price: data.price, image: data.image_url, tags: data.tags };
  const ai = await enrichWithAI(prod, '', {});
  ctx.session.review = { prod, ai, table, updateId: id };
  return ctx.replyWithMarkdownV2('*Editing existing item*\n' + reviewMessage(prod, ai, table), kbConfirm);
});

/* --------------------- bulk driver --------------------- */
async function handleBulkStep(ctx) {
  const table = ctx.session.table;
  const items = ctx.session.products || [];
  const idx = ctx.session.index || 0;

  if (idx >= items.length) {
    ctx.session.mode = null;
    ctx.session.stage = null;
    ctx.session.await = null;
    return ctx.reply('All products processed. ✅', kbAfterTask());
  }

  const prod = items[idx];

  if (!prod._ai) {
    prod._ai = await enrichWithAI(prod, prod._txt || '', {});
  }

  if (!ok(prod.price)) {
    ctx.session.await = 'price';
    return ctx.reply(`Enter *price* for "${prod.name}"`, { parse_mode: 'Markdown' });
  }
  if (table === TABLES.products && !ok(prod.originalPrice)) {
    ctx.session.await = 'originalPrice';
    return ctx.reply(`Enter *MRP* for "${prod.name}"`, { parse_mode: 'Markdown' });
  }
  if (table === TABLES.products && !ok(prod.stock)) {
    ctx.session.await = 'stock';
    return ctx.reply(`Enter *stock* for "${prod.name}"`, { parse_mode: 'Markdown' });
  }
  if (!ok(prod.image)) {
    ctx.session.await = 'image';
    return ctx.reply(`Upload *image* for "${prod.name}" (or type "generate" / "skip" / paste URL)`, { parse_mode: 'Markdown' });
  }

  ctx.session.review = { prod, ai: prod._ai, table };
  await ctx.replyWithMarkdownV2(reviewMessage(prod, prod._ai, table), kbConfirm);

  ctx.session.index = idx + 1;
  ctx.session.await = null;
}

/* --------------------- errors & launch --------------------- */
bot.catch((err, ctx) => {
  console.error('Bot error', err);
  if (isAdmin(ctx)) ctx.reply('⚠️ Unexpected error.');
});

bot.telegram.deleteWebhook({ drop_pending_updates: true }).catch(() => {});
bot.launch();
console.log('🚀 Product Bot running with /smartadd and /update');
