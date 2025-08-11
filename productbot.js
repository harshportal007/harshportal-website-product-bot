require('dotenv').config();
const { Telegraf, session, Markup } = require('telegraf');
const { createClient } = require('@supabase/supabase-js');
const fetch = globalThis.fetch ?? ((...a) => import('node-fetch').then(({ default: f }) => f(...a)));
const { GoogleGenerativeAI } = require('@google/generative-ai'); // ✅ keep
const OpenAI = require('openai');
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const ADMIN_IDS = (process.env.ADMIN_IDS || '7057639075')
  .split(',').map(s => Number(s.trim())).filter(Boolean);

const TABLES = { products: 'products', exclusive: 'exclusive_products' };

const bot = new Telegraf(process.env.TELEGRAM_BOT_TOKEN);
const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_KEY);
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY); // ✅ keep

const CATEGORIES_ALLOWED = ['OTT Accounts', 'IPTV', 'Product Key', 'Download'];
const categories = CATEGORIES_ALLOWED;

/* ----------------------- utils ----------------------- */
const isAdmin = (ctx) => ADMIN_IDS.includes(ctx.from.id);
const ok = (x) => typeof x !== 'undefined' && x !== null && x !== '';

// Convert any value to string safely (prevents s.replace crash)
const toStr = (v) => {
  if (v == null) return '';
  if (typeof v === 'string') return v;
  if (Array.isArray(v)) return v.join(', ');
  if (typeof v === 'object') return JSON.stringify(v);
  return String(v);
};

const TEXT_MODEL = process.env.GEMINI_TEXT_MODEL || 'gemini-2.5-flash';
const DEFAULT_IMAGE_MODEL =
  (process.env.GEMINI_IMAGE_MODEL && process.env.GEMINI_IMAGE_MODEL.trim()) ||
  'gemini-2.0-flash-preview-image-generation';


// Escape MarkdownV2 for Telegram safely
const escapeMd = (v = '') => toStr(v).replace(/([_\*\[\]\(\)~`>#+\-=|{}\.!])/g, '\\$1');

const parsePrice = (raw) => {
  if (!raw) return null;
  const m = String(raw).match(/(\d[\d,\.]*)/);
  if (!m) return null;
  return Math.round(parseFloat(m[1].replace(/,/g, '')));
};

/** Merge arrays without dupes, normalized */
const uniqMerge = (...arrs) => {
  const set = new Set();
  arrs.flat().filter(Boolean).forEach(x => set.add(String(x).trim()));
  return Array.from(set).filter(Boolean);
};

// maps fuzzy terms → one of the allowed categories
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
  // 1) already allowed?
  if (CATEGORIES_ALLOWED.includes(aiCategory)) return aiCategory;


  const hay = [
    prodLike.name, prodLike.description, prodLike.category, prodLike.subcategory,
    Array.isArray(prodLike.tags) ? prodLike.tags.join(' ') : prodLike.tags
  ].filter(Boolean).join(' ');

  // 2) infer from content
  const inferred = normalizeCategoryFromText(hay) || normalizeCategoryFromText(aiCategory || '');
  if (inferred) return inferred;

  // 3) safe default
  return 'Download';
}

// Reliable Telegram file URL for both photos and documents
async function tgFileUrl(fileId) {
  try {
    const f = await bot.telegram.getFile(fileId); // { file_path: '...' }
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
                 head[8]===0x57 && head[9]===0x45 && head[10]===0x42 && head[11]===0x50; // WEBP

  if (isPNG) return { mime: 'image/png', ext: 'png' };
  if (isJPG) return { mime: 'image/jpeg', ext: 'jpg' };
  if (isRIFF) return { mime: 'image/webp', ext: 'webp' };

  if (ext === 'png') return { mime: 'image/png', ext: 'png' };
  if (ext === 'jpg' || ext === 'jpeg') return { mime: 'image/jpeg', ext: 'jpg' };
  if (ext === 'webp') return { mime: 'image/webp', ext: 'webp' };
  if (ext === 'svg') return { mime: 'image/svg+xml', ext: 'svg' };

  return { mime: 'image/jpeg', ext: 'jpg' };
}



/* --------------- OpenGraph hinting from URLs --------------- */
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
    // true for {your}.supabase.co/storage/v1/object/public/<bucket>/...
    return url.hostname === base.hostname && /\/storage\/v1\/object\/public\//.test(url.pathname);
  } catch { return false; }
}

async function ensureHostedInSupabase(u, table, filenameHint='prod.jpg') {
  if (!u || !/^https?:\/\//i.test(u)) return u;       // not a URL → nothing to do
  if (isSupabasePublicUrl(u)) return u;               // already in your storage
  return rehostToSupabase(u, filenameHint, table);    // rehost external URL
}


/* --------------- Supabase Storage (unchanged) --------------- */
async function rehostToSupabase(fileUrl, filenameHint = 'image.jpg', table) {
  try {
    const res = await fetch(fileUrl);
    if (!res.ok) throw new Error(`Fetch failed: ${res.status}`);
    const buf = Buffer.from(await res.arrayBuffer());

    // choose bucket/folder
    const bucket = table === TABLES.products
      ? (process.env.SUPABASE_BUCKET_PRODUCTS || 'images')
      : (process.env.SUPABASE_BUCKET_EXCLUSIVE || 'exclusiveproduct-images');
    const folder = table === TABLES.products ? 'products' : 'exclusive-products';

    // decide mime/ext
    const headerCT = (res.headers.get('content-type') || '').toLowerCase();
    let { mime, ext } = guessMime(buf, filenameHint);
    if (headerCT.startsWith('image/') && headerCT !== 'application/octet-stream') {
      // trust real image content-types from upstream, otherwise keep our guess
      mime = headerCT;
      ext  = headerCT.includes('png') ? 'png'
          : headerCT.includes('webp') ? 'webp'
          : headerCT.includes('svg') ? 'svg' : 'jpg';
    }

    // make filename match the mime we’ll store
    const base = filenameHint.replace(/\.[A-Za-z0-9]+$/, '');
    const safeName = `${base}.${ext}`;
    const key = `${folder}/${Date.now()}-${Math.random().toString(36).slice(2)}-${safeName}`;

    console.log('[upload] bucket=', bucket, 'key=', key, 'mime=', mime);

    const { error: upErr } = await supabase.storage.from(bucket).upload(key, buf, {
      contentType: mime,
      upsert: true,
    });
    if (upErr) {
      console.error('[upload] error', upErr);
      throw upErr;
    }

    const { data: pub } = supabase.storage.from(bucket).getPublicUrl(key);
    console.log('[upload] publicUrl=', pub?.publicUrl);
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
  {
    match: /\bspotify\b/i,
    name: 'Spotify',
    palette: ['#1DB954', '#121212', '#1e1e1e'],
    keywords: 'music streaming, circular equalizer, waveform rings, rounded icon silhouette'
  },
  {
    match: /\bnetflix\b/i,
    name: 'Netflix',
    palette: ['#E50914', '#0B0B0B'],
    keywords: 'cinematic streaming, widescreen frame, ribbon arc, spotlight vignette'
  },

{
  match: /\bv0(\.dev)?|\bvercel\b/i,
  name: 'Vercel v0',
  palette: ['#0b0b0b', '#111111', '#ffffff'],
  keywords: 'minimal monochrome, rounded square tile, official v0 glyph, black background, white logomark'
},

  {
    match: /\byou ?tube|yt ?premium\b/i,
    name: 'YouTube',
    palette: ['#FF0000', '#FFFFFF', '#0f0f0f'],
    keywords: 'play triangle motif, video tile grid, glossy rounded icon silhouette'
  },
  {
    match: /\bprime|amazon prime\b/i,
    name: 'Prime Video',
    palette: ['#00A8E1', '#0B0B0B', '#0a2533'],
    keywords: 'streaming, arc swoosh, cinematic glow'
  },
  {
    match: /\bdisney\b/i,
    name: 'Disney+',
    palette: ['#0d3df2', '#0b0b0b', '#1a73e8'],
    keywords: 'night sky arc, starlight sparkles, smooth blue glow'
  },
  {
    match: /\biptv\b/i,
    name: 'IPTV',
    palette: ['#00E0FF', '#141414', '#00FFA3'],
    keywords: 'channel mosaic, antenna waves, signal bars, modern tv glyph'
  },
  {
    match: /\bspotify|netflix|youtube|yt|prime|amazon|disney|iptv|hbo|hotstar|zee|sonyliv|hulu|paramount\b/i,
    // generic streaming palette if we hit other OTT names
    name: 'OTT',
    palette: ['#00FFC6', '#0B0B0B', '#7C4DFF'],
    keywords: 'abstract tv glyph, play icons, cinematic glow'
  },
];



function getBrandStyle(prod) {
  const tags = Array.isArray(prod?.tags) ? prod.tags.join(' ') : (prod?.tags || '');
  const hay = [prod?.name, prod?.description, prod?.category, prod?.subcategory, tags]
    .filter(Boolean).join(' ').toLowerCase();
  return BRAND_STYLES.find(b => b.match.test(hay)) || null;
}



// Brand icon "recipes" to force real look (rounded-square app icon style)
const BRAND_ICON_RECIPES = {
  spotify: {
    bg: '#121212', fg: '#1DB954',
    describe: 'rounded-square black app icon, centered green circle with three horizontal curved bars (audio waves)'
  },
  netflix: {
    bg: '#0B0B0B', fg: '#E50914',
    describe: 'rounded-square black app icon, centered red ribbon N monogram with diagonal fold'
  },
  youtube: {
    bg: '#0f0f0f', fg: '#FF0000',
    describe: 'rounded-square dark app icon, centered red rounded rectangle with white play triangle'
  },
  'v0': {
    bg: '#0b0b0b', fg: '#ffffff',
    describe: 'rounded-square pure black tile, minimalist white v0 monogram glyph'
  },
  raycast: {
    bg: '#0b0b0b', fg: '#FF6363',
    describe: 'rounded-square black tile, centered coral asterisk-like raycast glyph made of 4 bars'
  },
  warp: {
    bg: '#0b0b0b', fg: '#7C4DFF',
    describe: 'rounded-square black tile, centered angular W ribbon glyph in purple'
  }
};

function brandRecipe(prod) {
  const n = String(prod?.name || '').toLowerCase();
  for (const key of Object.keys(BRAND_ICON_RECIPES)) {
    if (n.includes(key)) return BRAND_ICON_RECIPES[key];
  }
  return null;
}

/* ---------------- NEW: small helpers for refs + watermark ---------------- */
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

// optional watermark with sharp; fall back to prompt watermark
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



/* ---------------- AI enrichment (unchanged logic) ---------------- */
async function enrichWithAI(prod, textHints = '', ogHints = {}) {
  const model = genAI.getGenerativeModel({ model: "gemini-2.5-pro" });
  const brand = getBrandStyle?.(prod); // optional: from your brand helpers

  const prompt = `
You are a product data normalizer. Fix misspellings, normalize brands, infer likely plan/validity,
and return compact JSON. Do NOT invent prices. If unsure, use "unknown" (not empty).
Keep "description" <= 220 chars. Also extract 3–6 short "features" (bullet-style phrases).

Given:
- Raw product JSON: ${JSON.stringify(prod)}
- Extra text (can be messy): """${textHints}"""
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
`;

  try {
    const out = await model.generateContent(prompt);
    const text = await out.response.text();
    const json = JSON.parse(text.match(/\{[\s\S]*\}$/m)[0]);

    // Clean + coerce
    json.name = (json.name || prod.name || '').toString().trim();
    json.description = (json.description || prod.description || '').toString().trim();
    if (json.price !== 'unknown') json.price = parsePrice(json.price);

    // clamp to allowed categories
    json.category = normalizeCategory({ ...prod, ...json }, json.category);

    json.tags = Array.isArray(json.tags) ? json.tags.slice(0, 8) : [];

    // Features (normalize to 3–6 short items)
    let feats = Array.isArray(json.features) ? json.features : [];
    if (!feats.length && textHints) {
      feats = String(textHints)
        .split(/\n|[;•·\-–—]\s+/g)
        .map(s => s.trim())
        .filter(s => s.length > 3 && s.length <= 80)
        .slice(0, 6);
    }
    json.features = feats
      .map(s => s.replace(/^[\-\*\•\·–—]\s*/, '').trim())
      .filter(Boolean)
      .slice(0, 6);

    // Gradient (2 colors) fallback to brand palette if missing
    const palette = (brand?.palette && brand.palette.length >= 2)
      ? brand.palette.slice(0, 2)
      : ['#0ea5e9', '#7c3aed']; // cyan → violet fallback
    if (!Array.isArray(json.gradient) || json.gradient.length < 2) {
      json.gradient = palette;
    } else {
      json.gradient = json.gradient.slice(0, 2);
    }

    return json;
  } catch (e) {
    console.error('AI enrich error:', e.message);
    // Safe fallback
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

/** For messy freeform message → product JSON */
async function extractFromFreeform(text) {
  const model = genAI.getGenerativeModel({ model: "gemini-2.5-pro" });
  const prompt = `
Extract a single product from this freeform text (it might be messy). If quantity/pack/plan/validity are present, keep them.
Return ONLY JSON: {"name":"","plan":"","validity":"","price":"","description":""}

Text:
${text}
`;
  try {
    const out = await model.generateContent(prompt);
    const raw = out.response.text();
    const item = JSON.parse(raw.match(/\{[\s\S]*\}$/m)[0]);
    item.price = parsePrice(item.price);
    return item;
  } catch {
    return { name: '', plan: '', validity: '', price: null, description: '' };
  }
}

/* ---- Robust array parse to avoid "Could not detect products" ---- */
function tryParseJsonArrayAnywhere(raw) {
  if (!raw) return null;
  let s = String(raw).trim();
  s = s.replace(/```(?:json)?\s*([\s\S]*?)\s*```/gi, '$1'); // strip code fences
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
  const model = genAI.getGenerativeModel({ model: "gemini-2.5-pro" });
  const prompt = `
You will extract a LIST of products from messy text.
Return ONLY JSON array in this exact shape (no prose):

[
  {"name":"","plan":"","validity":"","price":"","description":"","tags":[]}
]

Text:
"""${text}"""
`;
  try {
    const out = await model.generateContent(prompt);
    const raw = out.response.text();

    // robust parse
    let arr = tryParseJsonArrayAnywhere(raw);
    if (!Array.isArray(arr)) {
      // fallback: try to split by lines if not array
      const lines = String(text).split(/\n+/).map(s => s.trim()).filter(Boolean);
      arr = [];
      for (const line of lines) {
        const one = await extractFromFreeform(line);
        if (one?.name) arr.push(one);
      }
    }

    // normalize numbers + keep original text
    arr = (arr || []).map(x => ({
      name: x.name || '',
      plan: x.plan || '',
      validity: x.validity || '',
      price: parsePrice(x.price),
      description: x.description || '',
      tags: Array.isArray(x.tags) ? x.tags.slice(0,8) : [],
      _txt: text
    }));

    // filter empties
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


/* ---------------- IMAGE GENERATION: match 80–99% + watermark ---------------- */

// Build prompt for packshot (no text) or title mode, allow prompt-watermark flag
// 80–99% match prompt builder (supports optional prompt-watermark)
async function buildImagePrompt(prod, styleName = 'neo', wantPromptWatermark = false) {
  const model = genAI.getGenerativeModel({ model: TEXT_MODEL });
  const themeText = STYLE_THEMES[styleName] || STYLE_THEMES.neo;
  const brand = getBrandStyle(prod);
  const brandName = shortBrandName(prod);
  const palette = brand?.palette?.join(', ') || 'brand-accurate colors';
  const brandHints = brand?.keywords || 'official logomark proportions';

  const sys = `
Create a 1024x1024 ecommerce product tile. You may receive up to 3 reference images.
MUST reproduce the official brand/logo silhouette and colors **80–99%** as in the references.
Layout:
- large logo centered
- short brand/product name beneath the logo in bold, clean sans-serif
- keep at least 10% padding from every edge
No borders. No extra iconography. Do NOT invent a new logo.`.trim();

  const recipe = brandRecipe(prod);
  const motif = recipe
    ? `Visual recipe: ${recipe.describe}. Background ${recipe.bg}, logo color ${recipe.fg}.`
    : brandHints;

  const user = `
Product:
${JSON.stringify({ name: prod?.name, category: prod?.category, tags: prod?.tags })}

Brand/context: ${brandName}
Palette: ${palette}
Motifs: ${motif}

Rules:
- Treat reference images as ground truth for shape and color.
- Keep name text short and **fully readable** within the tile bounds.
- Avoid generic gradient circles; vary the look per brand.
${wantPromptWatermark ? 'Add a tiny bottom-right watermark text: "Harshportal".' : ''}

Negative: placeholder blobs, tiny/cropped text, extra logos, heavy glare, borders, watermarks (unless requested).`.trim();

  try {
    const out = await model.generateContent({
  contents: [{ role: 'user', parts: [{ text: sys + '\n\n' + user }]}]
});
    return out.response.text().trim().replace(/\s+/g, ' ');
  } catch {
    return `Brand-accurate 1:1 tile for ${brandName}; large centered official logo; readable brand name under it; ${themeText}; match references 80–99%; no borders.`;
  }
}




async function generateImageOpenAI({ prompt, size = '1024x1024' }) {
  const out = await openai.images.generate({
    model: process.env.OPENAI_IMAGE_MODEL || 'gpt-image-1',
    prompt,
    size,
  });
  const b64 = out.data?.[0]?.b64_json;
  if (!b64) throw new Error('OpenAI did not return an image');
  return Buffer.from(b64, 'base64'); // PNG buffer
}


function bgOnlyPromptFrom(prompt) {
  return `${prompt}

IMPORTANT:
- Produce ONLY a clean, aesthetic background (gradients, subtle glow, light effects).
- Do NOT include any text, symbols, icons, watermarks, or logos.`;
}

async function generateBackgroundWithOpenAI(prompt) {
  const res = await openai.images.generate({
    model: process.env.OPENAI_IMAGE_MODEL || 'gpt-image-1',
    prompt: bgOnlyPromptFrom(prompt),
    size: '1024x1024',
    n: 1
  });
  const b64 = res.data?.[0]?.b64_json;
  if (!b64) throw new Error('OpenAI returned no background');
  return Buffer.from(b64, 'base64');
}

// --- tiny timeout helper (no AbortSignal needed) ---
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

// Accept refs and build parts accordingly
async function generateProductImageBytes({ prompt, refImages = [], brandName }) {
  const backend = (process.env.IMG_BACKEND || '').toLowerCase();
  console.log('[img] backend:', backend, 'openaiModel:', process.env.OPENAI_IMAGE_MODEL || 'gpt-image-1');

  if (backend === 'openai') {
    // pure OpenAI image gen
    return generateImageOpenAI({ prompt });
  }

  if (backend === 'openai-bg') {
    // OpenAI background + exact logo overlay
    const bgBuf = await generateBackgroundWithOpenAI(prompt);
    const logoRef = refImages?.[0];
    if (!logoRef?.b64) throw new Error('No logo reference available');
    return composeTileWithLogo({ bgBuf, logoRef, brandName });
  }

  // Fallback to Gemini image models (with refs)
  const candidates = Array.from(new Set([
    DEFAULT_IMAGE_MODEL,
    'gemini-2.0-flash-preview-image-generation',
    'gemini-2.0-flash-exp',
    'gemini-2.0-flash',
  ]));

  let lastErr;
  const parts = [];
  for (const r of (refImages || []).slice(0, 3)) {
    if (r?.b64) parts.push({ inlineData: { data: r.b64, mimeType: r.mime || 'image/jpeg' } });
  }
  const safePrompt = String(prompt || '');
  parts.push({ text: safePrompt });

for (const id of candidates) {
  try {
    const model = genAI.getGenerativeModel({ model: id });
    const t = withTimeout(20000); // 20s per attempt

    const res = await t.race(
      model.generateContent({
        contents: [{ role: 'user', parts }],
        generationConfig: { temperature: 0.2, responseModalities: ['TEXT', 'IMAGE'] },
      })
    );
    t.clear();

    const prts = res.response?.candidates?.[0]?.content?.parts ?? [];
    const imagePart =
      prts.find(p => p.inlineData && p.inlineData.data) ||
      prts.find(p => p.media && p.media.data);
    if (!imagePart) throw new Error('No inline image returned');

    return Buffer.from(
      imagePart.inlineData?.data || imagePart.media.data,
      'base64'
    );
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
  const key = `${folder}/${Date.now()}-${Math.random()
    .toString(36)
    .slice(2)}-${filename}`;

  const { error: upErr } = await supabase.storage
    .from(bucket)
    .upload(key, buf, { contentType, upsert: true });
  if (upErr) throw upErr;

  const { data: pub } = supabase.storage.from(bucket).getPublicUrl(key);
  return pub.publicUrl;
}

/* Title SVG (bigger, auto-scales) */
function makeTextCardSvg(title = 'Product', subtitle = '') {
  const brand = getBrandStyle?.({ name: title }) || null;
  const [c1, c2] = (brand?.palette?.length >= 2) ? brand.palette.slice(0, 2) : ['#0ea5e9', '#7c3aed'];

  const esc = s => String(s || '').replace(/[<&>]/g, c => ({'<':'&lt;','>':'&gt;','&':'&amp;'}[c]));
  const tRaw = (title || '').trim();
  const t = esc(tRaw).slice(0, 48);
  const sub = esc(subtitle).slice(0, 64);

  // Fit text to width: 880px area centered
  // Start big then constrain with textLength to avoid clipping
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



/** MAIN: generate image for a product with ref-match + watermark
 *  Priority inside generation:
 *   1) Photoreal/hyperreal packshot (no text), match refs 80–99%, watermark applied server-side if possible
 *   2) Title image (big readable text), brand/vibe matched, watermark
 *   3) SVG fallback (always works), watermark baked
 */

// --- map common products to their real domains (extend anytime) ---
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
};


function resolveBrandDomain(name = '') {
  const n = String(name).toLowerCase().trim();
  for (const k of Object.keys(BRAND_DOMAIN_MAP)) {
    if (n.includes(k)) return BRAND_DOMAIN_MAP[k];
  }
  // heuristic: strip spaces/punct → .com
  const slug = n.replace(/[^a-z0-9]/g,'');
  if (slug.length >= 3) return `${slug}.com`;
  return null;
}

// pull high-confidence refs: Clearbit logo + homepage OG + unavatar (best-effort)
async function fetchBrandRefs(name) {
  const refs = [];
  const domain = resolveBrandDomain(name);

  if (domain) {
    refs.push(`https://logo.clearbit.com/${domain}?size=1024`);
    try {
      const og = await getOG(`https://${domain}`);
      if (og?.image) refs.push(og.image);
    } catch {}
    // small but sometimes useful:
    refs.push(`https://icons.duckduckgo.com/ip3/${domain}.ico`);
    refs.push(`https://unavatar.io/${domain}`);
  }

  // Wikipedia often returns the official wordmark/logo
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

  // fetch as base64
  const uniq = Array.from(new Set(refs.filter(Boolean))).slice(0, 3);
  const blobs = [];
  for (const u of uniq) {
    const b = await fetchAsBase64(u).catch(() => null);
    if (b) blobs.push(b);
  }
  return blobs;
}

async function composeTileWithLogo({ bgBuf, logoRef, brandName }) {
  if (!_sharp) throw new Error('sharp not available');

  // decode logo
  const logoBuf = Buffer.from(logoRef.b64, 'base64');

  // scale logo to ~55% width, auto height, centered
  const canvas = await _sharp(bgBuf)
    .resize(1024, 1024, { fit: 'cover' })
    .toBuffer();

  const logoPng = await _sharp(logoBuf)
    .resize({ width: 560, withoutEnlargement: true }) // ~55%
    .png()
    .toBuffer();

  // brand name SVG (big, readable, within 10% padding)
  const safe = (s='').replace(/[<&>]/g,c=>({ '<':'&lt;','>':'&gt;','&':'&amp;' }[c]));
  const title = safe(brandName || 'Product').slice(0,48);
  const textSvg = Buffer.from(`
    <svg xmlns="http://www.w3.org/2000/svg" width="1024" height="300">
      <style>
        .t{ font-family: "Inter","Segoe UI",Roboto,Arial; font-weight: 800; }
      </style>
      <text x="512" y="230" text-anchor="middle" class="t" font-size="96" fill="#fff">${title}</text>
    </svg>
  `);

  // compose: logo centered vertically at 420px, text near bottom
  const composed = await _sharp(canvas)
    .composite([
      { input: logoPng, top: 260, left: Math.round((1024-560)/2) },
      { input: textSvg, top: 724, left: 0 }
    ])
    .png()
    .toBuffer();

  // watermark (your existing helper)
  const withWm = await addWatermarkToBuffer(composed, 'Harshportal') || composed;
  return withWm;
}





// --- Search the web for product name and get up to 3 image refs ---
async function webSearchRefs(productName) {
  if (!productName) return [];
  try {
    const query = encodeURIComponent(productName);
    const searchUrl = `https://api.duckduckgo.com/?q=${query}&format=json&no_redirect=1&t=hpbot`;
    const res = await fetch(searchUrl);
    const data = await res.json();

    let urls = [];
    if (data?.Image) urls.push(data.Image);
    if (Array.isArray(data?.RelatedTopics)) {
      for (const t of data.RelatedTopics) {
        const u = t?.Icon?.URL;
        if (!u) continue;
        const abs = u.startsWith('http') ? u : `https://duckduckgo.com${u}`;
        urls.push(abs);
      }
    }

    // keep only likely real images
    urls = urls
      .filter(Boolean)
      .filter(u => /\.(jpg|jpeg|png|webp)(\?|$)/i.test(u))
      .slice(0, 3);

    const refs = [];
    for (const u of urls) {
      const b = await fetchAsBase64(u).catch(()=>null);
      if (b) refs.push(b);
    }
    return refs;
  } catch (err) {
    console.warn('webSearchRefs failed:', err.message);
    return [];
  }
}

// --- deterministic brand tile (uses first fetched ref) ---
// PLACE THIS JUST ABOVE ensureImageForProduct(...)
async function composeBrandTile(prod, table) {
  if (!_sharp) return null;

  const refs = await fetchBrandRefs(prod?.name || '');
  const ref = refs[0];
  if (!ref) return null;

  const size = 1024;
  const recipe = brandRecipe(prod) || { bg: '#0b0b0b', fg: '#ffffff' };
  const brand = shortBrandName(prod);

  // base background
  let out = await _sharp({
    create: { width: size, height: size, channels: 3, background: recipe.bg }
  }).png().toBuffer();

  // center the logo (~54% width)
  const logoBuf = Buffer.from(ref.b64, 'base64');
  const resized = await _sharp(logoBuf)
    .resize(Math.round(size * 0.54), null, { fit: 'inside' })
    .png()
    .toBuffer();

  const meta = await _sharp(resized).metadata();
  const left = Math.round((size - (meta.width || 0)) / 2);
  const top  = Math.round(size * 0.18);

  out = await _sharp(out).composite([{ input: resized, left, top }]).png().toBuffer();

  // brand name text (SVG overlay)
  const textSvg = Buffer.from(`
    <svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}">
      <text x="50%" y="${Math.round(size * 0.85)}"
            font-family="Inter, Segoe UI, Roboto, Arial"
            font-size="${Math.round(size * 0.09)}"
            font-weight="800" fill="#ffffff" text-anchor="middle">${brand}</text>
    </svg>`);
  out = await _sharp(out).composite([{ input: textSvg }]).png().toBuffer();

  // watermark (optional)
  out = (await addWatermarkToBuffer(out, 'Harshportal')) || out;

  return uploadImageBufferToSupabase(out, { table, filename: 'brand.png', contentType: 'image/png' });
}



async function ensureImageForProduct(prod, table, style = 'neo') {
  // Respect existing image (manual upload / URL)
  if (prod?.image && String(prod.image).trim()) return prod.image;

  // Collect up to 3 strong refs
  const brandRefs = await fetchBrandRefs(prod?.name || prod?.title || '');
  const ogRef = prod?.og_image ? await fetchAsBase64(prod.og_image).catch(() => null) : null;
  const refBlobs = [...brandRefs, ...(ogRef ? [ogRef] : [])].slice(0, 3);

    // 1) Try photoreal / brand-true AI image first
  try {
   const prompt = await buildImagePrompt(prod, style, !_sharp);
const brandName = shortBrandName(prod);
let buf = await generateProductImageBytes({
  prompt,            // <= add this
  refImages: refBlobs,
  brandName
});


    // Prefer server watermark (consistent & always visible)
      if (_sharp) {
    const wm = await addWatermarkToBuffer(buf, 'Harshportal');
    if (wm) buf = wm;
  }
  return await uploadImageBufferToSupabase(buf, { table, filename: 'ai.png', contentType: 'image/png' });
} catch (e) {
   console.warn('AI image generation failed, will try brand tile:', e?.message || e);
   }

  // 2) Deterministic brand tile fallback
  try {
    const deterministic = await composeBrandTile(prod, table);
    if (deterministic) return deterministic;
  } catch (e) {
    console.warn('composeBrandTile failed:', e?.message || e);
  }

 // 3) Last resort: SVG title card
   const svg = makeTextCardSvg(prod?.name || 'Digital Product', prod?.plan || prod?.subcategory || '');
   const svgBuf = Buffer.from(svg, 'utf8');
   return uploadImageBufferToSupabase(svgBuf, { table, filename: 'ai.svg', contentType: 'image/svg+xml' });
 }


/* --------------------- /style command --------------------- */
const STYLE_KEYS = Object.keys(STYLE_THEMES); // ['neo','minimal','gradient','cyber','clay']

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
bot.action('again_new',      (ctx)=>{ ctx.answerCbQuery(); ctx.reply('Use /table then type *products* or *exclusive*', { parse_mode:'Markdown' }); });
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
// 1) manual wizard
bot.command('addproduct', (ctx) => {
  if (!isAdmin(ctx)) return;
  if (!ctx.session.table) return ctx.reply('First choose a table: *products* or *exclusive*', { parse_mode: 'Markdown' });
  ctx.session.mode = 'manual';
  ctx.session.form = { step: 0, prod: {} };
  ctx.reply('Enter *Name*:', { parse_mode: 'Markdown' });
});

// 2) AI bulk
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

// 3) SMART ADD — one messy message (optionally with URL/photo)
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

  // choose table
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
      og_image: ctx.session.smart.og?.image || null // ✅ pass OG image as potential reference (generation only)
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

  // inline edit handler text
  if (ctx.session.awaitEdit) {
    await applyInlineEdit(ctx);
    return;
  }

  return next();
});

/* --------------------- photo handlers (unchanged upload flow) --------------------- */
  // -- put this helper near your other helpers --
async function processIncomingImage(ctx, fileId, filenameHint = 'prod.jpg') {
  const href = await tgFileUrl(fileId); // robust, works for photo & document

  // 1) EDIT MODE: user tapped "image" on the review keyboard
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


  // 4) GEMINI BULK: waiting for an image for current item
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
  const fileId = photos.at(-1)?.file_id; // largest size
  if (!fileId) return;
  await processIncomingImage(ctx, fileId, 'prod.jpg');
});

bot.on('document', async (ctx) => {
  if (!isAdmin(ctx)) return;
  const doc = ctx.message?.document;
  if (!doc || !/^image\//i.test(doc.mime_type || '')) return; // ignore non-image docs
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
  // Rehost if needed
  if (prod.image && /^https?:\/\//i.test(prod.image) && !isSupabasePublicUrl(prod.image)) {
    try { prod.image = await ensureHostedInSupabase(prod.image, table); } catch {}
  }

 const idKey  = updateId; // keep as string to support UUIDs
 const imgCol = table === TABLES.products ? 'image' : 'image_url';

 // 1) IMAGE ONLY — no .select() here
 console.log('[save] about to write', table, idKey, 'imgCol=', imgCol, 'value=', prod.image);

  const { error: imgWriteErr } = await supabase
    .from(table)
    .update({ [imgCol]: prod.image || null })
    .eq('id', idKey);
  if (imgWriteErr) {
    await ctx.reply(`❌ Image update failed: ${imgWriteErr.message}`);
    return;
  }

  // 2) Read-after-write (separate select)
  const { data: fresh, error: readErr } = await supabase
    .from(table)
    .select(`id, ${imgCol}`).eq('id', idKey)
    .maybeSingle();
  if (readErr) {
    console.log('[image:readback] error:', readErr);
  } else {
    console.log('[image:readback]', table, idKey, '->', imgCol, fresh?.[imgCol]);
  }

  // 3) Update the rest (avoid touching image column again)
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
  ctx.session.index = (ctx.session.index || 0) + 1; // move pointer now
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

    // enrich first for a richer prompt
    const ai = await enrichWithAI(prodBase, ctx.session.smart.text, ctx.session.smart.og);

    // include OG image as ref for generation only
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

  // finished?
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

// ➜ advance to the next item now, so after "Save" we move on
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

