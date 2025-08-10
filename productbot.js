require('dotenv').config();
const { Telegraf, session, Markup } = require('telegraf');
const { createClient } = require('@supabase/supabase-js');
const fetch = (...a) => import('node-fetch').then(({ default: f }) => f(...a));
const { GoogleGenerativeAI } = require('@google/generative-ai'); // ✅ keep

const ADMIN_IDS = (process.env.ADMIN_IDS || '7057639075')
  .split(',').map(s => Number(s.trim())).filter(Boolean);

const TABLES = { products: 'products', exclusive: 'exclusive_products' };

const bot = new Telegraf(process.env.TELEGRAM_BOT_TOKEN);
const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_KEY);
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY); // ✅ keep

// optional default bucket (only used by rehostToSupabase)
const STORAGE_BUCKET = process.env.SUPABASE_BUCKET || 'exclusiveproduct-images';


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
const textModel = genAI.getGenerativeModel({ model: TEXT_MODEL });


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


/* --------------- OpenGraph hinting from URLs --------------- */
const findUrls = (txt = '') => (txt.match(/https?:\/\/\S+/gi) || []).slice(0, 3);

async function getOG(url) {
  try {
    const res = await fetch(url, { timeout: 10_000 });
    if (!res.ok) throw new Error(res.statusText);
    const html = await res.text();
    const pick = (prop) => {
      const r = new RegExp(`<meta[^>]+property=["']${prop}["'][^>]+content=["']([^"']+)["']`, 'i').exec(html)
             || new RegExp(`<meta[^>]+name=["']${prop}["'][^>]+content=["']([^"']+)["']`, 'i').exec(html);
      return r ? r[1] : null;
    };
    return {
      title: pick('og:title') || pick('twitter:title'),
      description: pick('og:description') || pick('twitter:description'),
      image: pick('og:image') || pick('twitter:image'),
    };
  } catch {
    return {};
  }
}

/* --------------- Supabase Storage (optional) --------------- */
async function rehostToSupabase(fileUrl, filenameHint = 'image.jpg', table) {
  try {
    const res = await fetch(fileUrl);
    if (!res.ok) throw new Error(`Fetch failed: ${res.status}`);
    const buf = Buffer.from(await res.arrayBuffer());

    const bucket = table === TABLES.products
      ? (process.env.SUPABASE_BUCKET_PRODUCTS || 'images')
      : (process.env.SUPABASE_BUCKET_EXCLUSIVE || 'exclusiveproduct-images');

    const folder = table === TABLES.products ? 'products' : 'exclusive-products';
    const key = `${folder}/${Date.now()}-${Math.random().toString(36).slice(2)}-${filenameHint}`;

    console.log('[upload] bucket=', bucket, 'key=', key);

    const { error: upErr } = await supabase.storage.from(bucket).upload(key, buf, {
      contentType: res.headers.get('content-type') || 'image/jpeg',
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
    throw e; // rethrow so your catch shows the ❌ message
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
  const hay = [
    prod?.name, prod?.description, prod?.category, prod?.subcategory,
    Array.isArray(prod?.tags) ? prod.tags.join(' ') : prod?.tags
  ].filter(Boolean).join(' ').toLowerCase();

  for (const b of BRAND_STYLES) {
    if (b.match.test(hay)) return b;
  }
  return null;
}

/* --------------- AI helpers --------------- */
const categoriesStr = categories.join(' | ');

/** Stronger enrichment prompt – fix mess, but say "unknown" if unsure */
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
  "features": ["3-6 concise bullet phrases"],        // e.g. ["Ad-free listening","Offline downloads"]
  "gradient": ["#hex1","#hex2"]                      // 2 colors best matching brand/theme
}
`;

  try {
    const out = await model.generateContent(prompt);
    const text = await out.response.text();
    const json = JSON.parse(text.match(/\{[\s\S]*\}$/m)[0]);

    // Clean + coerce
   // Clean
json.name = (json.name || prod.name || '').toString().trim();
json.description = (json.description || prod.description || '').toString().trim();
if (json.price !== 'unknown') json.price = parsePrice(json.price);

// ⬇️ clamp to allowed categories
json.category = normalizeCategory({ ...prod, ...json }, json.category);

json.tags = Array.isArray(json.tags) ? json.tags.slice(0, 8) : [];


    // Features (normalize to 3–6 short items)
    let feats = Array.isArray(json.features) ? json.features : [];
    if (!feats.length && textHints) {
      // fallback: mine bullet-y phrases from the user text
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
    // Safe fallback with minimal fields; still try to give a gradient
    const palette = (getBrandStyle?.(prod)?.palette?.slice(0,2)) || ['#0ea5e9','#7c3aed'];
    return {
      name: prod.name || '',
      plan: prod.plan || 'unknown',
      validity: prod.validity || 'unknown',
      price: parsePrice(prod.price) || null,
      description: prod.description || '',
      tags: [],
      category: 'Other',
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

/* ---------------- Gemini 2.0 Flash Preview image gen + Supabase upload ---------------- */
// Replace your whole generateProductImageBytes() with this version
async function generateProductImageBytes(prompt) {
  const modelName = process.env.GEMINI_IMAGE_MODEL || 'gemini-2.0-flash-preview-image-generation';
  const model = genAI.getGenerativeModel({ model: modelName });

  // Ask the model to return an IMAGE (it will also return short TEXT metadata)
  const res = await model.generateContent({
    contents: [{ role: 'user', parts: [{ text: prompt }]}],
    generationConfig: {
      // 👇 this is the important bit
      responseModalities: ['TEXT', 'IMAGE'],
    },
  });

  // Grab the first inline image
  const parts = res.response?.candidates?.[0]?.content?.parts ?? [];
  const imagePart = parts.find(p => p.inlineData && p.inlineData.data);
  if (!imagePart) {
    throw new Error('No image returned by the image model');
  }

  // inlineData is base64; convert to a Buffer
  return Buffer.from(imagePart.inlineData.data, 'base64');
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

/** Convenience: generate then upload → return public URL */
async function generateProductImageUrl(prompt, table) {
  const bytes = await generateProductImageBytes(prompt);
  return uploadImageBufferToSupabase(bytes, { table, filename: 'ai.png', contentType: 'image/png' });
}

/* ---------- AI image generation (brand-adjacent, no faces/logos) ---------- */

const BRAND_HINTS = [
  {
    match: [/spotify/i],
    palette: ['#1DB954', '#191414'],
    motifs: [
      'concentric circular sound waves',
      'rounded-square app tile glow',
      'equalizer bars',
      'subtle vinyl-disc silhouette'
    ],
  },
  {
    match: [/netflix/i],
    palette: ['#E50914', '#000000'],
    motifs: ['red light beam forming abstract N', 'cinema screen glow', 'soft film grain vignette'],
  },
  {
    match: [/prime\s?video|amazon\s?prime/i],
    palette: ['#00A8E1', '#0E1111'],
    motifs: ['clean play arrow silhouette', 'smile-curve arrow hint', 'edge glow'],
  },
  {
    match: [/disney/i, /hotstar/i],
    palette: ['#04144E', '#0B5AE0'],
    motifs: ['starry arc glow', 'night-sky gradient', 'subtle bokeh'],
  },
  {
    match: [/youtube/i],
    palette: ['#FF0000', '#0F0F0F'],
    motifs: ['rounded play triangle', 'screen reflection', 'subtle red glow'],
  },
  {
    match: [/canva/i],
    palette: ['#00C4CC', '#7D2AE8'],
    motifs: ['fluid gradient splash', 'curvy vector shapes', 'pen-nib silhouette'],
  },
  {
    match: [/vpn/i],
    palette: ['#00D4FF', '#0B132B'],
    motifs: ['shield silhouette', 'network nodes and lines', 'tunnel glow'],
  },
  {
    match: [/iptv/i, /live tv/i],
    palette: ['#22D3EE', '#0F172A'],
    motifs: ['channel tiles mosaic', 'play button aura', 'signal waves'],
  },
];

function pickBrandHints(prod = {}) {
  const hay = [
    prod.name, prod.description, prod.category, prod.subcategory, ...(prod.tags || []),
  ].filter(Boolean).join(' ');
  return BRAND_HINTS.find(h => h.match.some(rx => rx.test(hay))) || null;
}

/* ---------- AI image generation (brand-aware) ---------- */
async function buildImagePrompt(prod, styleName = 'neo') {
  const model = genAI.getGenerativeModel({ model: TEXT_MODEL }); // reuse your text model
  const themeText = STYLE_THEMES[styleName] || STYLE_THEMES.neo;

  const brand = getBrandStyle(prod);
  const brandName = brand?.name || (prod?.category || 'Digital product');
  const palette = brand?.palette?.join(', ') || 'teal, cyan, indigo on dark';
  const brandHints = brand?.keywords || 'clean geometric icon, product glyph';

  // Ask Gemini to compress to one powerful line
  const sys = `
You create single-line image prompts for a model that **must not render real logos or text**.
Use **brand-colored abstract/icon-like motifs** only (logo-like, not the actual trademark).
Keep it 1:1 composition, centered product tile/card look, shop-ready.
`;
  const user = `
Product JSON:
${JSON.stringify({
  name: prod?.name,
  plan: prod?.plan,
  validity: prod?.validity,
  category: prod?.category,
  subcategory: prod?.subcategory,
  tags: prod?.tags
}, null, 0)}

Brand/context: ${brandName}
Palette: ${palette}
Motifs: ${brandHints}
Theme: ${themeText}

Return ONE compact line describing the scene for an image generator, INCLUDING color words.
Hard rules: no real brand logos, no trademark shapes, no text, no watermarks. Ratio 1:1.
`;

  try {
    const out = await model.generateContent([{role:'user',parts:[{text: sys + '\n' + user}]}]);
    return out.response.text().trim().replace(/\s+/g, ' ');
  } catch {
    // solid fallback
    return `Abstract ${brandName} product tile, ${brandHints}, colors ${palette}, ${themeText}, centered, 1:1, no text or logos`;
  }
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
  ctx.reply(`🎨 Style set to *${pick}*`, { parse_mode: 'Markdown' });
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

  // ✅ escape this too
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

  // pick columns + sort column per table
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
    console.error('List error:', error);               // <-- keep for debugging
    return ctx.reply(`DB error: ${error.message}`);    // or keep it generic if you prefer
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

/* --------------------- text router --------------------- */
bot.on('text', async (ctx, next) => {
  if (!isAdmin(ctx)) return;

  // choose table
  if (!ctx.session.table) {
    const t = ctx.message.text.trim().toLowerCase();
    if (t === 'products' || t === 'exclusive') {
      ctx.session.table = t === 'exclusive' ? TABLES.exclusive : TABLES.products;
      return ctx.reply(
        `Table set to *${ctx.session.table}*.\nCommands:\n• /addproduct (manual)\n• /addproductgemini (AI bulk)\n• /smartadd (one-shot messy add)\n• /list  • /toggle <id>  • /update <id>`,
        { parse_mode: 'Markdown' }
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
    if (urls.length) {
      // fetch first OG as hint
      ctx.session.smart.og = await getOG(urls[0]);
    }
    // try AI extraction/enrichment
    const rough = await extractFromFreeform(ctx.message.text);
    const enriched = await enrichWithAI(rough, ctx.message.text, ctx.session.smart.og);
    ctx.session.smart.prod = {
      name: enriched.name || rough.name,
      plan: enriched.plan !== 'unknown' ? enriched.plan : rough.plan,
      validity: enriched.validity !== 'unknown' ? enriched.validity : rough.validity,
      price: enriched.price || rough.price || null,
      description: enriched.description || rough.description || '',
      tags: uniqMerge(enriched.tags),
      image: null
    };
    // ask for photo if not provided
    ctx.session.mode = 'smart-photo';
    return ctx.reply('If you have a product *image*, send it now. Or type "skip".', { parse_mode: 'Markdown' });
  }

  // gemini bulk continues...
  if (ctx.session.mode === 'gemini') {
    if (ctx.session.stage === 'paste') {
      await ctx.reply('⏳ Extracting with Gemini…');
      const items = await extractFromFreeform(ctx.message.text).then(p => p.name ? [p] : []);
      if (!items.length) return ctx.reply('Could not detect a product. Use /smartadd and include more details.');
      ctx.session.stage = 'step';
      ctx.session.products = items;
      ctx.session.index = 0;
      await ctx.reply(`Detected *${items.length}* item. I’ll ask missing fields.`, { parse_mode: 'Markdown' });
    }
    return handleBulkStep(ctx);
  }

  // inline edit handler text (applies for /update too)
  if (ctx.session.awaitEdit) {
    await applyInlineEdit(ctx);
    return;
  }

  return next();
});

/* --------------------- photo handlers --------------------- */
bot.on('photo', async (ctx) => {
  if (!isAdmin(ctx)) return;

  /* --- EDIT MODE: user tapped "image" and uploaded a new photo --- */
  // EDIT MODE: upload photo -> rehost to the right bucket by table
if (ctx.session.awaitEdit === 'image' && ctx.session.review) {
  try {
    const fileId = ctx.message.photo.pop().file_id;
    const fileLink = await ctx.telegram.getFileLink(fileId);
    const imgUrl = await rehostToSupabase(fileLink.href, 'prod.jpg', ctx.session.review.table);
    ctx.session.review.prod.image = imgUrl;
    ctx.session.awaitEdit = null;
    const { prod, ai, table } = ctx.session.review;
    await ctx.replyWithMarkdownV2(reviewMessage(prod, ai, table), kbConfirm);
  } catch (e) {
    console.error('edit image upload error:', e);
    await ctx.reply('❌ Could not update image. Try again or paste a direct image URL.');
  }
}


  // --- manual wizard photo ---
  if (ctx.session.mode === 'manual-photo') {
    const fileId = ctx.message.photo.pop().file_id;
    const fileLink = await ctx.telegram.getFileLink(fileId);
    const imgUrl = await rehostToSupabase(fileLink.href, 'prod.jpg', ctx.session.table);

    const prod = ctx.session.form.prod;
    prod.image = imgUrl;

    const ai = await enrichWithAI(prod, '', {});
    ctx.session.review = { prod, ai, table: ctx.session.table };
    return ctx.replyWithMarkdownV2(reviewMessage(prod, ai, ctx.session.table), kbConfirm);
  }

  // --- smart add photo ---
  if (ctx.session.mode === 'smart-photo') {
    const fileId = ctx.message.photo.pop().file_id;
    const fileLink = await ctx.telegram.getFileLink(fileId);
   ctx.session.smart.photo = await rehostToSupabase(fileLink.href, 'prod.jpg', ctx.session.table);

    const prod = { ...ctx.session.smart.prod, image: ctx.session.smart.photo };
    const ai = await enrichWithAI(prod, ctx.session.smart.text, ctx.session.smart.og);
    ctx.session.review = { prod: { ...prod, tags: uniqMerge(prod.tags, ai.tags) }, ai, table: ctx.session.table };

    ctx.session.mode = null;
    return ctx.replyWithMarkdownV2(reviewMessage(ctx.session.review.prod, ai, ctx.session.table), kbConfirm);
  }
});


/* If smart add user types "skip" instead of photo -> ask to generate */
bot.hears(/^skip$/i, async (ctx) => {
  if (!isAdmin(ctx)) return;
  if (ctx.session.mode === 'smart-photo') {
    // store what we have; ask if they want AI image
    const prod = { ...ctx.session.smart.prod, image: null };
    ctx.session.smart.prod = prod;  // keep it
    await ctx.reply('No image provided. Do you want me to generate a product image for you?', kbGenImage);
  }
});

/* --------------------- callbacks --------------------- */
bot.action('cancel', (ctx) => {
  ctx.answerCbQuery();
  ctx.session.review = null;
  ctx.session.mode = null;
  ctx.reply('Cancelled. Use /addproduct • /smartadd • /addproductgemini to start again.');
});

bot.action('save', async (ctx) => {
  if (!ctx.session.review) return ctx.answerCbQuery('Nothing to save');
  ctx.answerCbQuery('Saving…');

  const { prod, ai, table, updateId } = ctx.session.review;

  // common payload builders
  const baseTags = uniqMerge(prod.tags, ai.tags);

  if (updateId) {
    // UPDATE existing
    if (table === TABLES.products) {
      const payload = {
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
      };
      const { error } = await supabase.from(TABLES.products).update(payload).eq('id', updateId);
      if (error) await ctx.reply(`❌ Update failed: ${error.message}`);
       else await ctx.reply(escapeMd('✅ Updated products'), { parse_mode: 'MarkdownV2' });
    } else {
      const payload = {
        name: prod.name,
        description: prod.description || ai.description || null,
        price: prod.price || null,
        image_url: prod.image || null,
        tags: baseTags,
      };
      const { error } = await supabase.from(TABLES.exclusive).update(payload).eq('id', updateId);
      if (error) await ctx.reply(`❌ Update failed: ${error.message}`);
      else await ctx.reply(escapeMd('✅ Updated exclusive_products'), { parse_mode: 'MarkdownV2' });
    }
    ctx.session.review = null;
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
    if (error) await ctx.reply(`❌ Failed: ${error.message}`);
    else await ctx.reply(escapeMd('✅ Added to products'), { parse_mode: 'MarkdownV2' });
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
    if (error) await ctx.reply(`❌ Failed: ${error.message}`);
    else await ctx.reply(escapeMd('✅ Added to exclusive_products'), { parse_mode: 'MarkdownV2' });
  }

   ctx.session.review = null;
  ctx.session.mode = null;
  await ctx.reply('Done ✅  — Use /smartadd to add another, or /list to view latest.');
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
  const prod = { ...ctx.session.smart.prod, image: null };
  const ai = await enrichWithAI(prod, ctx.session.smart.text, ctx.session.smart.og);
  ctx.session.review = { prod: { ...prod, tags: uniqMerge(prod.tags, ai.tags) }, ai, table: ctx.session.table };
  ctx.session.mode = null;
  return ctx.replyWithMarkdownV2(reviewMessage(ctx.session.review.prod, ai, ctx.session.table), kbConfirm);
});

bot.action('gen_img_yes', async (ctx) => {
  ctx.answerCbQuery('Generating…');
  try {
    await ctx.reply('🎨 Generating image… this can take ~10–20s.');
    const table = ctx.session.table;
    const prodBase = { ...ctx.session.smart.prod };

    // enrich first for a richer prompt
    const ai = await enrichWithAI(prodBase, ctx.session.smart.text, ctx.session.smart.og);
    const prodForPrompt = { ...prodBase, ...ai };
    const prompt = await buildImagePrompt(prodForPrompt, ctx.session.style || 'neo');
    const url = await generateProductImageUrl(prompt, table);
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
   const base = (ctx.session.review?.prod) || (ctx.session.smart?.prod) || {};
  const prompt = await buildImagePrompt(base, ctx.session.style || 'neo');



  try {
    const url = await generateProductImageUrl(prompt, table);
    // attach to the current product in session
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
    const prompt = await buildImagePrompt({ ...prod, ...ai }, ctx.session.style || 'neo');
    const url = await generateProductImageUrl(prompt, rvw.table);
    prod.image = url;
  } catch (e) {
    console.error('edit image generate failed:', e);
    return ctx.reply('⚠️ Could not generate an image. Paste an image URL or upload a photo.');
  }
  ctx.session.awaitEdit = null;
  return ctx.replyWithMarkdownV2(reviewMessage(prod, ai, rvw.table), kbConfirm);
}


if (field === 'image' && /^https?:\/\//i.test(val)) {
   // allow pasting a direct URL (route to the correct bucket by table)
  prod.image = await rehostToSupabase(val, 'prod.jpg', rvw.table);
} else if (field === 'category') {
  // ⬇️ clamp typed value to one of the 4 allowed categories
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
  const idx = ctx.session.index;
  if (idx >= ctx.session.products.length) {
    ctx.session.mode = null;
    return ctx.reply('All products processed.');
  }
  const prod = ctx.session.products[idx];
  if (!prod._ai) prod._ai = await enrichWithAI(prod, '', {});
  if (!ok(prod.price)) { ctx.session.await = 'price'; return ctx.reply(`Enter *price* for "${prod.name}"`, { parse_mode: 'Markdown' }); }
  if (table === TABLES.products && !ok(prod.originalPrice)) { ctx.session.await = 'originalPrice'; return ctx.reply(`Enter *MRP* for "${prod.name}"`, { parse_mode: 'Markdown' }); }
  if (table === TABLES.products && !ok(prod.stock)) { ctx.session.await = 'stock'; return ctx.reply(`Enter *stock* for "${prod.name}"`, { parse_mode: 'Markdown' }); }
  if (!ok(prod.image)) { ctx.session.await = 'image'; return ctx.reply(`Upload *image* for "${prod.name}"`, { parse_mode: 'Markdown' }); }

  ctx.session.review = { prod, ai: prod._ai, table };
  await ctx.replyWithMarkdownV2(reviewMessage(prod, prod._ai, table), kbConfirm);
  ctx.session.index += 1;
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
