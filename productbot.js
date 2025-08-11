require('dotenv').config();
const { Telegraf, session, Markup } = require('telegraf');
const { createClient } = require('@supabase/supabase-js');
const fetch = globalThis.fetch ?? ((...a) => import('node-fetch').then(({ default: f }) => f(...a)));
const { GoogleGenerativeAI } = require('@google/generative-ai'); // Gemini for TEXT
let _sharp = null;
try { _sharp = require('sharp'); } catch { console.warn('[img] `sharp` not installed. Some image features will be disabled.'); }


/* -------------------- config -------------------- */
const ADMIN_IDS = (process.env.ADMIN_IDS || '7057639075')
  .split(',').map(s => Number(s.trim())).filter(Boolean);

const TABLES = { products: 'products', exclusive: 'exclusive_products' };

const bot = new Telegraf(process.env.TELEGRAM_BOT_TOKEN);
const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_KEY);

const GEMINI_KEY = process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY;
const genAI = new GoogleGenerativeAI(GEMINI_KEY, { apiVersion: 'v1beta' });

const CATEGORIES_ALLOWED = ['OTT Accounts', 'IPTV', 'Product Key', 'Download'];
const categories = CATEGORIES_ALLOWED;

const TEXT_MODEL = process.env.GEMINI_TEXT_MODEL || 'gemini-2.5-pro';

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

const findUrls = (txt = '') => (txt.match(/https?:\/\/\S+/gi) || []).map(u => u.replace(/[),.\]]+$/, '')).slice(0, 5);


/* ---------------- category helpers ---------------- */
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

/* --------------- Telegram file URL --------------- */
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

/* --------------- Supabase host/rehost --------------- */
function isSupabasePublicUrl(u = '') {
  try {
    const url = new URL(u);
    const base = new URL(process.env.SUPABASE_URL);
    return url.hostname === base.hostname && /\/storage\/v1\/object\/public\//.test(url.pathname);
  } catch { return false; }
}

async function rehostToSupabase(fileUrl, filenameHint = 'image.jpg', table) {
  try {
    const res = await fetch(fileUrl);
    if (!res.ok) throw new Error(`Fetch failed: ${res.status}`);
    const buf = Buffer.from(await res.arrayBuffer());
    const bucket = table === TABLES.products ?
      (process.env.SUPABASE_BUCKET_PRODUCTS || 'images') :
      (process.env.SUPABASE_BUCKET_EXCLUSIVE || 'exclusiveproduct-images');
    const folder = table === TABLES.products ? 'products' : 'exclusive-products';
    const key = `${folder}/${Date.now()}-${Math.random().toString(36).slice(2)}-${filenameHint}`;
    const { error: upErr } = await supabase.storage.from(bucket).upload(key, buf, { upsert: true });
    if (upErr) throw upErr;
    const { data: pub } = supabase.storage.from(bucket).getPublicUrl(key);
    return pub.publicUrl;
  } catch (e) {
    console.error('Rehost failed:', e.message);
    throw e;
  }
}

async function ensureHostedInSupabase(u, table, filenameHint = 'prod.jpg') {
    if (!u || !/^https?:\/\//i.test(u)) return u;
    if (isSupabasePublicUrl(u)) return u;
    return rehostToSupabase(u, filenameHint, table);
}

/* ---------------- Brand/style helpers ---------------- */
const BRAND_STYLES = [
    { match: /\bspotify\b/i, name: 'Spotify', palette: ['#1DB954', '#121212'] },
    { match: /\bnetflix\b/i, name: 'Netflix', palette: ['#E50914', '#0B0B0B'] },
    { match: /\byou ?tube|yt ?premium\b/i, name: 'YouTube', palette: ['#FF0000', '#FFFFFF'] },
    { match: /\bcrunchyroll\b/i, name: 'Crunchyroll', palette: ['#F47521', '#FFFFFF'] },
];

function getBrandStyle(prod) {
  const hay = [prod?.name, prod?.description].filter(Boolean).join(' ').toLowerCase();
  return BRAND_STYLES.find(b => b.match.test(hay)) || null;
}

/* -------------------- AI enrichment -------------------- */
async function enrichWithAI(prod, textHints = '', ogHints = {}) {
    const model = genAI.getGenerativeModel({ model: TEXT_MODEL });
    const prompt = `You are a product data normalizer. Fix misspellings, normalize brands, infer likely plan/validity, and return compact JSON. Do NOT invent prices. If unsure, use "unknown". Keep "description" <= 220 chars. Also extract 3–6 short "features" (bullet-style phrases). Given: - Raw product JSON: ${JSON.stringify(prod)} - Extra text: "${textHints}" - OpenGraph hints: ${JSON.stringify(ogHints)} - Allowed categories: ${categories.join(' | ')} Return ONLY JSON: {"name": "string", "plan": "string|unknown", "validity": "string|unknown", "price": "number|unknown", "description": "string", "tags": ["3-8 tags"], "category": "one of: ${categories.join(' | ')}", "subcategory": "string|unknown", "features": ["3-6 concise bullet phrases"]}`;
    try {
        const out = await model.generateContent(prompt);
        const text = out.response.text().trim();
        const json = safeParseFirstJsonObject(text) || {};
        json.name = (json.name || prod.name || '').toString().trim();
        json.description = (json.description || prod.description || '').toString().trim();
        if (json.price !== 'unknown') json.price = parsePrice(json.price);
        json.category = normalizeCategory({ ...prod, ...json }, json.category);
        json.tags = Array.isArray(json.tags) ? json.tags.slice(0, 8) : [];
        let feats = Array.isArray(json.features) ? json.features : [];
        if (!feats.length && textHints) {
            feats = String(textHints).split(/\n|[;•·\-–—]\s+/g).map(s => s.trim()).filter(s => s.length > 3 && s.length <= 80).slice(0, 6);
        }
        json.features = feats.map(s => s.replace(/^[\-\*\•\•–—]\s*/, '').trim()).filter(Boolean).slice(0, 6);
        return json;
    } catch (e) {
        console.error('AI enrich error:', e.message);
        return {
            name: prod.name || '', plan: 'unknown', validity: 'unknown', price: parsePrice(prod.price) || null,
            description: prod.description || '', tags: [], category: normalizeCategory(prod),
            subcategory: 'unknown', features: []
        };
    }
}

async function extractFromFreeform(text) {
    const model = genAI.getGenerativeModel({ model: TEXT_MODEL });
    const prompt = `Extract a single product from this text. Return ONLY JSON: {"name":"","plan":"","validity":"","price":"","description":""}\nText: ${text}`;
    try {
        const out = await model.generateContent(prompt);
        const item = safeParseFirstJsonObject(out.response.text()) || {};
        item.price = parsePrice(item.price);
        return item;
    } catch { return {}; }
}

function shortBrandName(prod) {
  const n = String(prod?.name || '')
    .replace(/\b(premium|pro|subscription|subs|account|license|key|activation|fan|mega fan)\b/ig, '')
    .trim();
  return n || (prod?.name || 'Product');
}

/* ===================== IMAGE GENERATION (Final Logic: Logo-Centric) ===================== */

// Helper to find a brand's website domain from its name
function resolveBrandDomain(name = '') {
    const n = String(name).toLowerCase().trim();
    const map = {
        spotify: 'spotify.com', netflix: 'netflix.com', youtube: 'youtube.com',
        'yt premium': 'youtube.com', disney: 'disneyplus.com', 'prime video': 'primevideo.com',
        prime: 'primevideo.com', 'amazon prime': 'primevideo.com', crunchyroll: 'crunchyroll.com',
        'sony liv': 'sonyliv.com', sonyliv: 'sonyliv.com', zee5: 'zee5.com',
        tradingview: 'tradingview.com', canva: 'canva.com',
    };
    for (const k of Object.keys(map)) if (n.includes(k)) return map[k];
    const slug = n.replace(/[^a-z0-9]/g, '');
    return slug.length >= 3 ? `${slug}.com` : null;
}

// Helper to fetch any URL and return its content as a Base64 string
async function fetchAsBase64(url) {
    try {
        const res = await fetch(url, { signal: AbortSignal.timeout(8000) });
        if (!res.ok) return null;
        return { b64: Buffer.from(await res.arrayBuffer()).toString('base64') };
    } catch { return null; }
}

// Helper to scrape a website for its OpenGraph `og:image` tag
async function getOG(url) {
    try {
        const res = await fetch(url, { signal: AbortSignal.timeout(8000) });
        if (!res.ok) return {};
        const html = await res.text();
        const pick = (prop) => {
            const r = new RegExp(`<meta[^>]+property=["']${prop}["'][^>]+content=["']([^"']+)["']`, 'i').exec(html);
            return r ? r[1] : null;
        };
        return { image: pick('og:image:secure_url') || pick('og:image') };
    } catch { return {}; }
}

// Color and contrast helpers
const esc = (s = '') => String(s).replace(/[<&>]/g, c => ({ '<': '<', '>': '>', '&': '&' }[c]));
const hex2rgb = (h) => {
    const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(h);
    return m ? [parseInt(m[1], 16), parseInt(m[2], 16), parseInt(m[3], 16)] : [0, 0, 0];
};
const luma = (hex) => {
    const [r, g, b] = hex2rgb(hex);
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
};

// Extracts a dominant color palette from an image buffer using sharp
async function paletteFromB64(b64) {
    if (!_sharp) return [];
    const stats = await _sharp(Buffer.from(b64, 'base64')).stats();
    if (stats.dominant) {
        const { r, g, b } = stats.dominant;
        const toHex = (c) => c.toString(16).padStart(2, '0');
        return [`#${toHex(r)}${toHex(g)}${toHex(b)}`];
    }
    return [];
}

// Gathers all visual information about a brand
async function brandVisualHints(prod) {
    const name = shortBrandName(prod);
    const domain = resolveBrandDomain(name);
    let imageUrl = null, logoUrl = null, palette = [], fg = '#FFFFFF', bg = '#111111';

    if (domain) {
        try {
            const og = await getOG(`https://${domain}`);
            if (og.image) imageUrl = og.image;
        } catch {}
        logoUrl = `https://logo.clearbit.com/${domain}?size=512&format=png`;
    }

    const predefinedStyle = getBrandStyle(prod);
    if (predefinedStyle?.palette) {
        palette = predefinedStyle.palette;
    } else if (imageUrl || logoUrl) {
        try {
            const { b64 } = await fetchAsBase64(imageUrl || logoUrl);
            if (b64) palette = await paletteFromB64(b64);
        } catch {}
    }
    
    palette = Array.from(new Set(palette.filter(Boolean))).slice(0, 5);
    if (palette.length > 0) {
        palette.sort((a, b) => luma(a) - luma(b));
        bg = palette[0];
        fg = palette[palette.length - 1];
        if (luma(fg) - luma(bg) < 80) {
            bg = '#111111';
            fg = '#FFFFFF';
        }
    }
    return { name, imageUrl, logoUrl, bg, fg };
}

// Programmatically creates a "Logo on a Gradient" image
async function createLogoImage({ logoUrl, productName, plan, brandColors }) {
    if (!_sharp) throw new Error("sharp library is required for this feature.");
    if (!logoUrl) throw new Error("Logo URL was not provided.");

    let logoBuffer;
    try {
        const logoRes = await fetch(logoUrl, { signal: AbortSignal.timeout(8000) });
        if (!logoRes.ok) throw new Error('Failed to fetch logo.');
        logoBuffer = Buffer.from(await logoRes.arrayBuffer());
    } catch(e) { throw new Error(`Could not fetch logo: ${e.message}`); }
    
    const resizedLogo = await _sharp(logoBuffer)
        .trim()
        .resize({ width: 450, height: 300, fit: 'inside', withoutEnlargement: true })
        .toBuffer();

    const title = esc(productName).slice(0, 40);
    const subtitle = esc(plan || '').slice(0, 50);
    const textColor = luma(brandColors.bg) > 128 ? '#111111' : '#FFFFFF';
    
    const textSvg = `<svg width="900" height="250"><style>.title{font-size:80px;font-weight:800;font-family:Inter,Segoe UI,sans-serif}.subtitle{font-size:45px;font-weight:500;font-family:Inter,Segoe UI,sans-serif}</style><text x="450" y="100" text-anchor="middle" class="title" fill="${textColor}">${title}</text>${subtitle ? `<text x="450" y="170" text-anchor="middle" class="subtitle" fill="${textColor}" opacity="0.8">${subtitle}</text>` : ''}</svg>`;
    const textBuffer = await _sharp(Buffer.from(textSvg)).png().toBuffer();

    return _sharp({ create: { width: 1024, height: 1024, channels: 4, background: { r: 0, g: 0, b: 0, alpha: 0 } } })
        .composite([
            { input: Buffer.from(`<svg><defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="${brandColors.fg}"/><stop offset="100%" stop-color="${brandColors.bg}"/></linearGradient></defs><rect width="1024" height="1024" fill="url(#g)"/></svg>`), top: 0, left: 0 },
            { input: resizedLogo, gravity: 'center', top: -100 },
            { input: textBuffer, gravity: 'center', top: 200 }
        ])
        .png().toBuffer();
}

// Last resort local fallback card
function makeCleanCardSVG(name = 'Product', plan = '', brandColor = '#4DA3FF', bgColor = '#FFFFFF') {
    const title = [name, plan].filter(Boolean).join(' ').trim();
    const textColor = luma(bgColor) > 128 ? '#111111' : '#FFFFFF';
    return `<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="1024"><rect width="1024" height="1024" fill="${esc(bgColor)}"/><g transform="translate(512,350) scale(1.5)"><path d="M64 192v-64a64 64 0 0 1 64-64h192a64 64 0 0 1 64 64v64h64a64 64 0 0 1 64 64v64h-64a64 64 0 0 1 -64-64v-64h-192v64a64 64 0 0 1 -64 64h-64v-64a64 64 0 0 1 64-64z" fill="${esc(brandColor)}" transform="translate(-256, -160)"/></g><text x="512" y="620" text-anchor="middle" font-family="Inter, Segoe UI, sans-serif" font-size="78" font-weight="800" fill="${esc(textColor)}">${esc(title).slice(0,44)}</text><text x="980" y="980" text-anchor="end" font-family="Inter, Segoe UI, sans-serif" font-size="34" fill="${esc(textColor)}" opacity="0.4">Harshportal</text></svg>`.trim();
}
async function localCardToPng({ name, plan }) {
    if (!_sharp) throw new Error('sharp is not available.');
    const { fg, bg } = await brandVisualHints({ name });
    const svg = makeCleanCardSVG(name, plan, fg, bg);
    return await _sharp(Buffer.from(svg)).png().toBuffer();
}

// Main image generation orchestrator
async function ensureImageForProduct(prod, table) {
    if (prod?.image && String(prod.image).trim()) {
        return ensureHostedInSupabase(prod.image, table);
    }

    const hints = await brandVisualHints(prod);

    if (hints.imageUrl) {
        console.log(`[img] Found official image: ${hints.imageUrl}. Rehosting.`);
        try {
            return await rehostToSupabase(hints.imageUrl, `${prod.name}.jpg`, table);
        } catch (e) { console.warn(`[img] Failed to rehost official image. Falling back. ${e.message}`); }
    }

    if (hints.logoUrl) {
        console.log('[img] No official image found. Creating custom logo-based image.');
        try {
            const imageBuffer = await createLogoImage({
                logoUrl: hints.logoUrl, productName: prod.name,
                plan: prod.plan, brandColors: { fg: hints.fg, bg: hints.bg }
            });
            return await rehostToSupabase(imageBuffer, 'logo_card.png', table);
        } catch (e) { console.error(`[img] Failed to create logo-based image. ${e.message}`); }
    }
    
    console.warn('[img] All methods failed. Generating simple local card.');
    try {
        const imageBuffer = await localCardToPng({ name: prod.name, plan: prod.plan });
        return await rehostToSupabase(imageBuffer, 'local_fallback.png', table);
    } catch(e) {
        console.error("[img] CRITICAL: Local card generation failed.", e);
        return null;
    }
}


/* --------------------- keyboards / messages --------------------- */
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
    [Markup.button.callback('🏁 Done', 'again_done')],
  ]);
}

/* --------------------- review message --------------------- */
const kbEditWhich = (table) => {
  const common = ['name', 'plan', 'validity', 'price', 'description', 'tags', 'image'];
  const pro = ['originalPrice', 'stock', 'category', 'subcategory'];
  const fields = table === TABLES.products ? [...common, ...pro] : common;
  const rows = [];
  for (let i = 0; i < fields.length; i += 3) {
    rows.push(fields.slice(i, i + 3).map(f => Markup.button.callback(f, `edit_${f}`)));
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

  parts.push(`*Image:* ${prod.image ? `[View Image](${prod.image})` : 'No Image'}`);
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
  const { data, error } = await supabase.from(ctx.session.table).select('id,name,price,is_active').order('id', { ascending: false }).limit(12);
  if (error) return ctx.reply(`DB error: ${error.message}`);
  const items = data || [];
  if (!items.length) return ctx.reply('No items yet.');
  const msg = items.map((r, i) => `${i + 1}. ${r.name} — ₹${Number(r.price || 0).toLocaleString('en-IN')} — ${r.is_active ? '✅' : '⛔️'} (id: ${r.id})`).join('\n');
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
  const { data, error } = await supabase.from(ctx.session.table).select('is_active').eq('id', id).maybeSingle();
  if (error || !data) return ctx.reply('Not found.');
  const { error: upErr } = await supabase.from(ctx.session.table).update({ is_active: !data.is_active }).eq('id', id);
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
            return ctx.reply(escapeMd(`Table set to ${ctx.session.table}.\nCommands:\n• /addproduct\n• /smartadd\n• /list\n• /toggle <id>\n• /update <id>`), { parse_mode: 'MarkdownV2' });
        }
        return ctx.reply('Type *products* or *exclusive*', { parse_mode: 'Markdown' });
    }

    if (ctx.session.mode === 'smart' && ctx.session.await === 'blob') {
        ctx.session.smart = { text: ctx.message.text, photo: null };
        const rough = await extractFromFreeform(ctx.message.text);
        const enriched = await enrichWithAI(rough, ctx.message.text, {});
        
        const prod = {
            name: enriched.name || rough.name || '',
            plan: enriched.plan !== 'unknown' ? enriched.plan : (rough.plan || ''),
            validity: enriched.validity !== 'unknown' ? enriched.validity : (rough.validity || ''),
            price: enriched.price || rough.price || null,
            description: enriched.description || rough.description || '',
            tags: uniqMerge(enriched.tags),
            image: null,
        };

        await ctx.reply('⏳ Finding product image...', { parse_mode: 'Markdown' });
        prod.image = await ensureImageForProduct(prod, ctx.session.table);

        const ai = await enrichWithAI(prod, ctx.message.text, {});

        ctx.session.review = { prod, ai, table: ctx.session.table };
        ctx.session.mode = null;
        ctx.session.await = null;
        
        const caption = reviewMessage(prod, ai, ctx.session.table);
        if (prod.image) {
            await ctx.replyWithPhoto({ url: prod.image }, { caption, parse_mode: 'Markdown', ...kbConfirm });
        } else {
            await ctx.reply(caption, { parse_mode: 'Markdown', ...kbConfirm });
        }
        return;
    }

    if (ctx.session.awaitEdit) {
        await applyInlineEdit(ctx);
        return;
    }
    
    return next();
});

/* --------------------- photo handlers --------------------- */
async function processIncomingImage(ctx, fileId) {
    const href = await tgFileUrl(fileId);
    if (ctx.session.awaitEdit === 'image' && ctx.session.review) {
        try {
            const { table } = ctx.session.review;
            const imgUrl = await rehostToSupabase(href, 'edited.jpg', table);
            ctx.session.review.prod.image = imgUrl;
            ctx.session.awaitEdit = null;
            const { prod, ai } = ctx.session.review;
            await ctx.deleteMessage().catch(()=>{});
            await ctx.replyWithPhoto({ url: prod.image }, { caption: reviewMessage(prod, ai, table), parse_mode: 'Markdown', ...kbConfirm });
        } catch (e) {
            await ctx.reply(`❌ Could not update image. ${e?.message || ''}`);
        }
        return;
    }
    // Handle manual photo upload if needed
}

bot.on('photo', async (ctx) => {
    if (!isAdmin(ctx)) return;
    const photos = ctx.message?.photo || [];
    const fileId = photos.at(-1)?.file_id;
    if (!fileId) return;
    await processIncomingImage(ctx, fileId);
});

/* --------------------- callbacks --------------------- */
bot.action('cancel', (ctx) => {
    ctx.answerCbQuery();
    ctx.session.review = null;
    ctx.session.mode = null;
    ctx.deleteMessage().catch(()=>{});
    ctx.reply('Cancelled.', kbAfterTask());
});

bot.action('save', async (ctx) => {
    if (!ctx.session.review) return ctx.answerCbQuery('Nothing to save');
    await ctx.answerCbQuery('Saving…');
    const { prod, ai, table, updateId } = ctx.session.review;
    const baseTags = uniqMerge(prod.tags, ai.tags);

    const dataToSave = table === TABLES.products ? {
        name: prod.name, plan: prod.plan || null, validity: prod.validity || null,
        price: prod.price || null, originalPrice: prod.originalPrice || null,
        description: prod.description || ai.description || null, category: ai.category,
        subcategory: ai.subcategory || null, stock: prod.stock || null, tags: baseTags,
        features: ai.features || [], image: prod.image || null, is_active: true,
    } : {
        name: prod.name, description: prod.description || ai.description || null,
        price: prod.price || null, image_url: prod.image || null,
        is_active: true, tags: baseTags,
    };

    let error;
    if (updateId) {
        ({ error } = await supabase.from(table).update(dataToSave).eq('id', updateId));
    } else {
        ({ error } = await supabase.from(table).insert([dataToSave]));
    }

    await ctx.deleteMessage().catch(()=>{});
    if (error) {
        await ctx.reply(`❌ Save failed: ${error.message}`);
    } else {
        await ctx.reply(escapeMd(`✅ Saved to ${table}`), { parse_mode: 'MarkdownV2' });
        await ctx.reply('What next?', kbAfterTask());
    }

    ctx.session.review = null;
    ctx.session.mode = null;
});

bot.action('edit', async (ctx) => {
    if (!ctx.session.review) return ctx.answerCbQuery();
    ctx.answerCbQuery();
    const table = ctx.session.review.table;
    await ctx.deleteMessage().catch(()=>{});
    await ctx.reply('Which field to edit?', kbEditWhich(table));
});

bot.action(/^edit_(.+)$/, (ctx) => {
    const field = ctx.match[1];
    ctx.session.awaitEdit = field;
    ctx.answerCbQuery();
    ctx.reply(`Send new value for *${field}*`, { parse_mode: 'Markdown' });
});

bot.action('back_review', async (ctx) => {
    if (!ctx.session.review) return ctx.answerCbQuery();
    ctx.answerCbQuery();
    const { prod, ai, table } = ctx.session.review;
    await ctx.deleteMessage().catch(()=>{});
    const caption = reviewMessage(prod, ai, table);
    if (prod.image) {
        await ctx.replyWithPhoto({ url: prod.image }, { caption, parse_mode: 'Markdown', ...kbConfirm });
    } else {
        await ctx.reply(caption, { parse_mode: 'Markdown', ...kbConfirm });
    }
});

/* --------------------- inline edit apply --------------------- */
async function applyInlineEdit(ctx) {
    const field = ctx.session.awaitEdit;
    const rvw = ctx.session.review;
    if (!rvw) return;
    const { prod, table } = rvw;
    let val = ctx.message.text.trim();

    await ctx.deleteMessage(ctx.message.message_id - 1).catch(() => {}); // Delete "Which field to edit?"
    await ctx.deleteMessage().catch(() => {}); // Delete user's reply

    if (field === 'image' && val.toLowerCase() === 'generate') {
        await ctx.reply('🎨 Generating image…');
        prod.image = await ensureImageForProduct(prod, table);
    } else if (field === 'image' && /^https?:\/\//i.test(val)) {
        prod.image = await ensureHostedInSupabase(val, table);
    } else {
        if (['price', 'originalPrice', 'stock'].includes(field)) val = parsePrice(val);
        if (['tags'].includes(field)) val = val.split(';').map(x => x.trim()).filter(Boolean);
        prod[field] = val;
    }
    
    ctx.session.awaitEdit = null;
    const newAi = await enrichWithAI(prod, '', {});
    ctx.session.review.ai = newAi;

    const caption = reviewMessage(prod, newAi, table);
    if (prod.image) {
        await ctx.replyWithPhoto({ url: prod.image }, { caption, parse_mode: 'Markdown', ...kbConfirm });
    } else {
        await ctx.reply(caption, { parse_mode: 'Markdown', ...kbConfirm });
    }
}

/* --------------------- /update <id> --------------------- */
bot.command('update', async (ctx) => {
    if (!isAdmin(ctx)) return;
    const id = (ctx.message.text.split(' ')[1] || '').trim();
    if (!id) return ctx.reply('Usage: /update <id>');
    const table = ctx.session.table || TABLES.products;
    const { data, error } = await supabase.from(table).select('*').eq('id', id).maybeSingle();
    if (error || !data) return ctx.reply('Not found.');
    
    const prod = {
        name: data.name, plan: data.plan, validity: data.validity,
        price: data.price, originalPrice: data.originalPrice,
        description: data.description, tags: data.tags,
        stock: data.stock, image: data.image || (data.image_url || null),
    };
    const ai = await enrichWithAI(prod, '', {});
    
    ctx.session.review = { prod, ai, table, updateId: id };
    const caption = `*Editing existing item*\n` + reviewMessage(prod, ai, table);
    if(prod.image) {
        await ctx.replyWithPhoto({ url: prod.image }, { caption, parse_mode: 'Markdown', ...kbConfirm });
    } else {
        await ctx.reply(caption, { parse_mode: 'Markdown', ...kbConfirm });
    }
});

/* --------------------- errors & launch --------------------- */
bot.catch((err, ctx) => {
  console.error(`Bot error for user ${ctx.from?.id}:`, err);
  try {
    if (isAdmin(ctx)) ctx.reply('⚠️ Unexpected error. Check console logs.');
  } catch {}
});

bot.telegram.deleteWebhook({ drop_pending_updates: true }).catch(() => {});
bot.launch();
console.log('🚀 Product Bot running with /smartadd and /update');