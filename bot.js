'use strict';
require('dotenv').config();
const fs = require('fs');
const path = require('path');
const FONT_PATH = path.join(__dirname, 'assets', 'Inter.ttf');

// ✅ Safe font embedding (no crash if file missing)
let EMBED_FONT_B64 = '';
try {
  EMBED_FONT_B64 = fs.readFileSync(FONT_PATH).toString('base64');
} catch (e) {
  console.warn('[img] Inter.ttf not found, falling back to system fonts');
}
const SVG_FONT_STYLE = EMBED_FONT_B64
  ? `
  <style>
    @font-face {
      font-family: "AppInter";
      src: url(data:font/ttf;base64,${EMBED_FONT_B64}) format("truetype");
      font-weight: 100 900;
      font-style: normal;
    }
    .title { font-family: "AppInter", sans-serif; font-weight: 800; }
    .sub   { font-family: "AppInter", sans-serif; font-weight: 600; }
  </style>`
  : `<style>
      .title { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; font-weight: 800; }
      .sub   { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; font-weight: 600; }
    </style>`;


// Also allow a file:// URL fallback for SVG renderers that ignore data: fonts
const ABS_FONT = path.resolve(FONT_PATH).replace(/\\/g, '/');
const SVG_FONT_STYLE_FILE = `
  <style>
    @font-face {
      font-family: "AppInter";
      src: url("file://${ABS_FONT}") format("truetype");
      font-weight: 100 900;
      font-style: normal;
    }
    .title { font-family: "AppInter", sans-serif; font-weight: 800; }
    .sub   { font-family: "AppInter", sans-serif; font-weight: 600; }
  </style>`;

// Optional but recommended: draw text with node-canvas (works everywhere)
let _canvas = null;
try { _canvas = require('canvas'); }
catch (e) { console.warn('[img] canvas not available:', e.message); }

// Helper to rasterize the text overlay using the bundled font
async function rasterizeTextOverlayPNG(W, H, titleLines, titleFontPx, planText, planFontPx) {
  if (!_canvas) return null;
  const { createCanvas, registerFont } = _canvas;

  try { registerFont(FONT_PATH, { family: 'AppInter' }); }
  catch (e) { /* ignore re-register errors */ }

  const c = createCanvas(W, H);
  const ctx = c.getContext('2d');

  ctx.fillStyle = 'rgba(0,0,0,0.35)';
  ctx.fillRect(0, 0, W, H);

  ctx.fillStyle = '#FFFFFF';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'alphabetic';
  ctx.font = `800 ${titleFontPx}px AppInter`;

  const lhTitle = 1.15;
  const titleBlockHeight = Math.round(titleFontPx * (titleLines.length + (titleLines.length - 1) * (lhTitle - 1)));
  const titleYStart = Math.round(H * 0.50) - Math.round(titleBlockHeight * 0.25);

  let y = titleYStart;
  for (let i = 0; i < titleLines.length; i++) {
    if (i > 0) y += Math.round(titleFontPx * lhTitle);
    ctx.fillText(titleLines[i], W / 2, y);
  }

  if (planText) {
    const minGapPx = Math.max(Math.round(titleFontPx * 0.22), 28);
    const lastBaseline = y;
    const planY = lastBaseline + Math.round(titleFontPx * 0.9) + minGapPx;

    ctx.fillStyle = '#E5E7EB';
    ctx.font = `600 ${planFontPx}px AppInter`;
    ctx.fillText(planText, W / 2, planY);
  }

  return c.toBuffer('image/png');
}

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
const REQUIRED_ENV = ['TELEGRAM_BOT_TOKEN','SUPABASE_URL'];
if (!process.env.SUPABASE_SERVICE_ROLE_KEY && !process.env.SUPABASE_KEY) {
  console.error('❌ Missing env: SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_KEY)');
}
for (const k of REQUIRED_ENV) if (!process.env[k]) console.error(`❌ Missing env: ${k}`);

const HF_KEY = process.env.HUGGING_FACE_API_KEY || '';
const DEEPAI_KEY = process.env.DEEPAI_API_KEY || '';

const CF_ACCOUNT_ID = process.env.CLOUDFLARE_ACCOUNT_ID || '';
const CF_API_TOKEN  = process.env.CLOUDFLARE_API_TOKEN  || '';

const GEMINI_KEYS = (process.env.GEMINI_API_KEYS || process.env.GEMINI_API_KEY || '')
  .split(',')
  .map(s => s.trim())
  .filter(Boolean);

const GEMINI_TEXT_MODEL = process.env.GEMINI_TEXT_MODEL || 'gemini-1.5-pro-latest';
const GEMINI_BASE = process.env.GEMINI_BASE || 'https://generativelanguage.googleapis.com/v1beta';

/* -------------------- config -------------------- */
const ADMIN_IDS = (process.env.ADMIN_IDS || '7057639075')
  .split(',').map(s => Number(s.trim())).filter(n => Number.isFinite(n) && n>0);

const TABLES = { products: 'products', exclusive: 'exclusive_products' };
const CATEGORIES_ALLOWED = ['OTT Accounts', 'IPTV', 'Product Key', 'Download'];

const bot = new Telegraf(process.env.TELEGRAM_BOT_TOKEN, { handlerTimeout: 90000 });
bot.webhookReply = false;

const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_KEY
);

const groq = new OpenAI({
  apiKey: process.env.GROQ_API_KEY,
  baseURL: 'https://api.groq.com/openai/v1',
});

const hf = new HfInference(HF_KEY);
// NEW: Allow advanced Hugging Face model via .env file
const HF_IMAGE_MODEL = process.env.HF_PRO_MODEL || 'stabilityai/stable-diffusion-xl-base-1.0';


/* ---- idle / pause control ---- */
const INACTIVITY_MS = Math.max(0, parseInt(process.env.INACTIVITY_MS || '300000', 10));
const idleTimers = new Map();

function clearIdleTimer(userId) {
  const h = idleTimers.get(userId);
  if (h) { clearTimeout(h); idleTimers.delete(userId); }
}

async function triggerPause(ctx) {
  try {
    ctx.session.paused = true;
    await ctx.reply(
      '⏸️ Bot paused due to inactivity.',
      Markup.inlineKeyboard([[Markup.button.callback('▶️ Start', 'resume_bot')]])
    );
  } catch {}
}

function startIdleTimer(ctx) {
  if (!isAdmin(ctx) || INACTIVITY_MS <= 0) return;
  const uid = ctx.from.id;
  clearIdleTimer(uid);
  const h = setTimeout(() => triggerPause(ctx), INACTIVITY_MS);
  idleTimers.set(uid, h);
}

/* -------------------- utils -------------------- */
const isAdmin = (ctx) => !!ctx?.from && ADMIN_IDS.includes(ctx.from.id);
const ok = (x) => typeof x !== 'undefined' && x !== null && x !== '';
const toStr = (v) => String(v ?? '');
const escapeMd = (v = '') => toStr(v).replace(/([_*[\]()~`>#+\-=|{}.!])/g, '\\$1');
const replyMD = (ctx, text, extra={}) => ctx.reply(text, { parse_mode: 'Markdown', ...extra });
function sanitizeForFilename(name = 'product') { return name.replace(/[^a-z0-9_.-]/gi, '_').substring(0, 100); }
const parsePrice = (raw) => { if (!raw) return null; const m = String(raw).match(/(\d[\d,.]*)/); if (!m) return null; const n = parseFloat(m[1].replace(/,/g, '')); return Number.isFinite(n) ? Math.round(n) : null; };
const uniqMerge = (...arrs) => { const set = new Set(); arrs.flat().filter(Boolean).forEach(x => set.add(String(x).trim())); return Array.from(set).filter(Boolean); };
function sanitizeTextForAI(text = '') { return String(text || '').replace(/[*_`~]/g, '').replace(/\s{2,}/g, ' ').trim(); }
function safeParseFirstJsonObject(s) {
  if (!s) return null;
  s = String(s).replace(/```(?:json)?\s*([\s\S]*?)\s*```/gi, '$1').trim();
  const m = s.match(/{[\s\S]*}/m);
  if (!m) return null;
  try { return JSON.parse(m[0]); } catch { return null; }
}

function sniffImageType(buf = Buffer.alloc(0)) {
    if (buf.slice(0, 8).equals(Buffer.from([0x89,0x50,0x4E,0x47,0x0D,0x0A,0x1A,0x0A]))) return { mime: 'image/png', ext: '.png' };
    if (buf.slice(0, 3).equals(Buffer.from([0xFF,0xD8,0xFF]))) return { mime: 'image/jpeg', ext: '.jpg' };
    if (buf.slice(0, 12).toString('utf8').startsWith('RIFF') && buf.slice(8, 12).toString('utf8') === 'WEBP') return { mime: 'image/webp', ext: '.webp' };
    if (buf.slice(0, 5).toString('utf8') === '<?xml' || buf.slice(0, 4).toString('utf8') === '<svg') return { mime: 'image/svg+xml', ext: '.svg' };
    return { mime: 'application/octet-stream', ext: '' };
}
function extFromName(name=''){ const m = String(name).toLowerCase().match(/\.(png|jpe?g|webp|svg)$/i); return m ? `.${m[1].toLowerCase().replace('jpeg','jpg')}` : ''; }
function mimeFromExt(ext=''){ const e = ext.toLowerCase(); if (e === '.png') return 'image/png'; if (e === '.jpg' || e === '.jpeg') return 'image/jpeg'; if (e === '.webp') return 'image/webp'; if (e === '.svg') return 'image/svg+xml'; return null; }
function hostOf(u){ try{ return new URL(u).hostname.replace(/^www\./,''); }catch{ return null; } }
function stripTLD(host){ return (host||'').split('.').slice(0,-1).join('.') || host; }
function levenshtein(a, b){ a=String(a||''); b=String(b||''); const m=a.length,n=b.length; if(!m) return n; if(!n) return m; const dp=Array.from({length:m+1},(_,i)=>[i,...Array(n).fill(0)]); for(let j=1;j<=n;j++) dp[0][j]=j; for(let i=1;i<=m;i++){ for(let j=1;j<=n;j++){ const cost = a[i-1]===b[j-1]?0:1; dp[i][j]=Math.min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost); } } return dp[m][n]; }
const sleep = (ms)=>new Promise(r=>setTimeout(r, ms));
async function tryWithRetries(label, fn, attempts = 2, baseDelay = 800) { let lastErr; for (let i=0;i<attempts;i++) { try { return await fn(i); } catch (e) { lastErr = e; console.warn(`[retry] ${label} attempt ${i+1} failed: ${e.message}`); await sleep(baseDelay * Math.pow(2, i)); } } if (lastErr) console.warn(`[retry] ${label} giving up after ${attempts} attempts: ${lastErr.message}`); return null; }
const TEXT_RETRIES = Math.max(1, parseInt(process.env.TEXT_RETRIES||'2',10));
const IMAGE_RETRIES = Math.max(1, parseInt(process.env.IMAGE_RETRIES||'2',10));
function normalizeCategory(prodLike = {}, aiCategory) { if (CATEGORIES_ALLOWED.includes(aiCategory)) return aiCategory; const hay = [prodLike.name, prodLike.description, prodLike.category, prodLike.subcategory, Array.isArray(prodLike.tags) ? prodLike.tags.join(' ') : prodLike.tags].filter(Boolean).join(' '); const cat = CATEGORIES_ALLOWED.find(c => new RegExp(c.split(' ')[0], 'i').test(hay)); return cat || 'Download'; }
const URL_RX = /(https?:\/\/[^\s)]+)|(www\.[^\s)]+)/ig;
async function fetchWebsiteRaw(url) { if (!url) return { html: '', text: '' }; const normalized = url.startsWith('http') ? url : `https://${url}`; const toText = (html) => html.replace(/<script[^>]*>[\s\S]*?<\/script>/gi, '').replace(/<style[^>]*>[\s\S]*?<\/style>/gi, '').replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim(); try { const res = await fetch(normalized, { headers: { 'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)', 'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8', }, redirect: 'follow', signal: typeof AbortSignal?.timeout === 'function' ? AbortSignal.timeout(12000) : undefined, }); if (res.ok) { const html = await res.text(); const text = toText(html); if (text.length > 200) return { html, text }; } } catch {} try { const proxied = normalized.replace(/^https?:\/\//, ''); const res2 = await fetch(`https://r.jina.ai/${proxied}`, { headers: { 'user-agent': 'Mozilla/5.0' }, signal: typeof AbortSignal?.timeout === 'function' ? AbortSignal.timeout(12000) : undefined, }); if (res2.ok) { const txt = await res2.text(); if (txt && txt.length > 200) return { html: '', text: txt.slice(0, 20000) }; } } catch {} return { html: '', text: '' }; }

// RESTORED: Definition for pollinationsTextJSON was missing.
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
    return safeParseFirstJsonObject(content) ?? JSON.parse(content);
  } catch (e) {
    console.warn('[text] Pollinations failed:', e.message);
    return null;
  }
}

// RESTORED: Definition for geminiTextJSON was missing.
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

  for (let i = 0; i < GEMINI_KEYS.length; i++) {
    const key = GEMINI_KEYS[_geminiIndex++ % GEMINI_KEYS.length];
    try {
      const res = await fetch(
        `${GEMINI_BASE}/models/${encodeURIComponent(GEMINI_TEXT_MODEL)}:generateContent?key=${encodeURIComponent(key)}`,
        {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify(body),
          signal: typeof AbortSignal?.timeout === 'function' ? AbortSignal.timeout(25000) : undefined,
        }
      );

      if (!res.ok) {
        const msg = await res.text().catch(()=>'' );
        console.warn(`[text] Gemini HTTP ${res.status} — ${msg.slice(0, 300)}`);
        continue;
      }

      const data = await res.json();
      const parts = data?.candidates?.[0]?.content?.parts || [];
      const text = parts.map(p => p.text).filter(Boolean).join('');
      if (!text) continue;

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


// NEW: Helper function to get text from a Hugging Face model
async function generateTextFromHuggingFace(systemPrompt, userPrompt) {
  if (!HF_KEY) {
    console.warn('[text] Hugging Face skipped: HF_KEY not set.');
    return null;
  }
  const model = process.env.HF_TEXT_MODEL || "mistralai/Mixtral-8x7B-Instruct-v0.1";
  console.log(`[text] Using Hugging Face model: ${model}`);

  try {
    const fullPrompt = `${systemPrompt}\n\n${userPrompt}`;
    const result = await hf.conversational({
      model: model,
      inputs: fullPrompt,
      parameters: {
          max_new_tokens: 1024,
          temperature: 0.1,
      }
    });

    if (result && result.generated_text) {
        return safeParseFirstJsonObject(result.generated_text);
    }
    return null;
  } catch (e) {
    console.error('[text] Hugging Face conversational model failed:', e.message);
    return null;
  }
}

// CHANGED: This function now uses the GROQ_TEXT_MODEL from your .env file.
async function getTextFromProvider(provider, systemPrompt, userPrompt) {
  if (provider === 'pollinations') {
    return await pollinationsTextJSON(systemPrompt, userPrompt, 'searchgpt');
  }
  if (provider === 'groq') {
    try {
      // Use the model from .env, with a fallback just in case.
      const model = process.env.GROQ_TEXT_MODEL || 'llama-3.1-70b-versatile';
      console.log(`[text] Using Groq model: ${model}`);
      
      const r = await groq.chat.completions.create({
        messages: [{ role: 'system', content: systemPrompt }, { role: 'user', content: userPrompt }],
        model: model,
        temperature: 0.1,
        response_format: { type: 'json_object' },
      });
      const s = r.choices[0]?.message?.content || '';
      return safeParseFirstJsonObject(s) ?? (s ? JSON.parse(s) : null);
    } catch (e) {
      console.warn(`[text] Groq failed: ${e.message}`);
      return null;
    }
  }
  if (provider === 'gemini') {
    try { return await geminiTextJSON(systemPrompt, userPrompt); }
    catch (e) { console.warn('[text] Gemini failed:', e.message); return null; }
  }
  return null;
}

function extractMetaTags(html = '') { const pick = (prop, attr='property') => { const re = new RegExp(`<meta[^>]+${attr}=["']${prop}["'][^>]+content=["']([^"']+)["']`, 'i'); return re.exec(html)?.[1] || null; }; return { ogTitle: pick('og:title') || pick('twitter:title','name'), ogDesc: pick('og:description') || pick('twitter:description','name'), ogImage: pick('og:image:secure_url') || pick('og:image') || pick('twitter:image','name'), }; }
function extractJsonLdProduct(html = '') { const blocks = []; const rx = /<script[^>]+type=["']application\/ld\+json["'][^>]*>([\s\S]*?)<\/script>/gi; let m; while ((m = rx.exec(html)) !== null) { const raw = m[1].trim(); try { blocks.push(JSON.parse(raw)); } catch { try { const cleaned = raw.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/.*$/gm, '').replace(/,(\s*[}\]])/g, '$1'); blocks.push(JSON.parse(cleaned)); } catch {} } } const flat = blocks.flatMap(b => Array.isArray(b) ? b : [b]); const productNode = flat.find(n => { const t = (n['@type'] || n.type || ''); return (Array.isArray(t) ? t : [t]).some(x => String(x).toLowerCase() === 'product'); }); if (!productNode) return null; const offers = Array.isArray(productNode.offers) ? productNode.offers[0] : productNode.offers || {}; const priceNum = parsePrice(offers.price || offers.priceSpecification?.price); const validity = offers.availabilityEnds || offers.validThrough || productNode.validThrough || 'unknown'; const features = []; if (Array.isArray(productNode.additionalProperty)) { for (const p of productNode.additionalProperty) if (p?.name && p?.value) features.push(`${p.name}: ${p.value}`); } if (Array.isArray(productNode.featureList)) features.push(...productNode.featureList.filter(Boolean)); return { name: productNode.name || null, description: productNode.description || null, price: priceNum || null, validity, features }; }
async function rehostToSupabase(fileUrlOrBuffer, filenameHint = 'image.jpg', table) { let buf, serverType = null, finalName = sanitizeForFilename(filenameHint || 'image'); if (Buffer.isBuffer(fileUrlOrBuffer)) { buf = fileUrlOrBuffer; } else { const res = await fetch(fileUrlOrBuffer, { signal: typeof AbortSignal?.timeout==='function' ? AbortSignal.timeout(15000) : undefined, }); if (!res.ok) throw new Error(`Fetch failed: ${res.status}`); serverType = res.headers.get('content-type'); const ab = await res.arrayBuffer(); buf = Buffer.from(ab); try { const u = new URL(fileUrlOrBuffer); const urlExt = extFromName(u.pathname); if (urlExt && !extFromName(finalName)) finalName += urlExt; } catch {} } const hintExt = extFromName(finalName); let mime = (serverType && serverType.startsWith('image/')) ? serverType.split(';')[0] : mimeFromExt(hintExt); if (!mime || mime === 'application/octet-stream') { const sniff = sniffImageType(buf); if (!hintExt && sniff.ext) finalName += sniff.ext; if (!mime || mime === 'application/octet-stream') mime = sniff.mime; } if (!extFromName(finalName)) { finalName += '.jpg'; if (mime === 'application/octet-stream') mime = 'image/jpeg'; } const bucket = table === TABLES.products ? (process.env.SUPABASE_BUCKET_PRODUCTS || 'images') : (process.env.SUPABASE_BUCKET_EXCLUSIVE || 'exclusiveproduct-images'); const folder = table === TABLES.products ? 'products' : 'exclusive-products'; const key = `${folder}/${Date.now()}-${Math.random().toString(36).slice(2)}-${sanitizeForFilename(finalName)}`; console.log(`[upload] ${finalName} -> bucket=${bucket}, key=${key}, type=${mime}`); const { error: upErr } = await supabase.storage.from(bucket).upload(key, buf, { upsert: true, contentType: mime || 'image/jpeg', cacheControl: 'public, max-age=31536000, immutable' }); if (upErr) throw upErr; const { data: pub } = supabase.storage.from(bucket).getPublicUrl(key); return pub.publicUrl; }
function shortBrandName(prod) { const commonWords = ['premium','pro','plus','subscription','subs','account','license','key','activation','fan','mega','plan','tier','access','year','years','month','months','day','days','lifetime','annual','basic','standard','advanced','creator','business','enterprise','personal','family','student','individual']; const regex = new RegExp(`\\b(${commonWords.join('|')})\\b`, 'ig'); let name = String(prod?.name || 'Product').trim().split(/[-–—(]/)[0]; name = name.replace(regex, ''); name = name.replace(/\b\d+\b/g, ''); name = name.replace(/\s+/g, ' ').trim(); return name || prod?.name || 'Product'; }

// CHANGED: This function now includes the Hugging Face fallback logic.
// CHANGED: The "second pass" now also uses the configurable Groq model.
async function enrichWithAI(textHints = '', websiteContent = '', providerOrderParam = null, editStatus = async () => {}) {
  const cleanTextHints = sanitizeTextForAI(textHints);
  const guessedName = cleanTextHints.split('\n')[0].slice(0, 120);
  const planGuess = (cleanTextHints.match(/plan[:\-]?\s*([^\n]+)/i)?.[1] || '').slice(0, 80);

  const webBundle = await searchWebForProduct(guessedName, planGuess);
  const combinedSite = sanitizeTextForAI(`${websiteContent}\n\n${webBundle}`).slice(0, 16000);

  const systemPrompt = 'You MUST output ONLY one JSON object with EXACT keys: {"name":"string","plan":"string|unknown","validity":"string|unknown","price":"number|unknown","description":"string","tags":["string"],"category":"string","subcategory":"string|unknown","features":["string"]}';
  const userPrompt = `User text:\n"""${cleanTextHints}"""\n\nTrusted sources (use for description & features; do NOT invent):\n"""${combinedSite}"""\n\nRules:\n1) Prefer user's explicit name/plan/validity/price if present.\n2) Description: 1–3 factual sentences taken from the sources.\n3) Features: 4–6 short factual bullets taken from the sources.\n4) Category must be one of: ${CATEGORIES_ALLOWED.join(' | ')}.\n5) If some field is unknown, use "unknown". Return JSON only.`;

  const providerOrder = (Array.isArray(providerOrderParam) && providerOrderParam.length && providerOrderParam) ||
    (process.env.TEXT_PROVIDER_ORDER ? process.env.TEXT_PROVIDER_ORDER.split(',').map(s=>s.trim()).filter(Boolean) : ['groq','gemini','pollinations']);

  console.log('[text] providerOrder resolved to:', providerOrder);
  let { json, provider } = await runTextProvidersWithOrder(providerOrder, systemPrompt, userPrompt, editStatus);

  if (!json) {
    console.warn('[text] Primary providers failed. Trying Hugging Face fallback.');
    await editStatus('⚠️ Primary models failed. Trying HF fallback...');
    json = await generateTextFromHuggingFace(systemPrompt, userPrompt);
    provider = 'Hugging Face';
  }

  if (!json) {
    console.error('[text] All providers failed, including Hugging Face. Using minimal extraction.');
    await editStatus('❌ All AI models failed. Extracting basic info...');
    const extractedName = cleanTextHints.split('\n')[0].trim() || 'Product';
    return {
      name: extractedName, plan: 'unknown', validity: 'unknown',
      price: parsePrice(cleanTextHints), description: extractedName, tags: [], features: [],
      category: normalizeCategory({ name: extractedName, description: cleanTextHints }),
    };
  }

  json.name = json.name || guessedName || 'Product';
  json.plan = json.plan || planGuess || 'unknown';
  json.validity = json.validity || 'unknown';
  json.price = parsePrice(json.price || textHints);
  if (!Array.isArray(json.tags)) json.tags = (json.tags ? String(json.tags) : '').split(/[;,]/).map(s=>s.trim()).filter(Boolean);
  if (!Array.isArray(json.features)) json.features = [];

  const needDetail = (!json.description || json.description.length < 150 || json.features.length < 3);
  if (needDetail && combinedSite.length > 400 && provider !== 'Hugging Face') {
    try {
      await editStatus('✍️ Refining details with a second AI pass...');
      const detailPrompt = `From ONLY the following sources, write:\nA) A concise, factual 2–3 sentence description of "${json.name}" (${json.plan}).\nB) 5 short factual bullet features.\n\nSources:\n"""${combinedSite.slice(0, 8000)}"""`;
      const more = await groq.chat.completions.create({
        messages: [{ role: 'user', content: detailPrompt }],
        model: process.env.GROQ_TEXT_MODEL || 'llama-3.1-70b-versatile', // Use configurable model here too
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

async function runTextProvidersWithOrder(order, systemPrompt, userPrompt, editStatus = async () => {}) {
  for (const provider of order) {
    await editStatus(`➡️ Trying text provider: ${provider}...`);
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
    if (json) {
        await editStatus(`✅ Text provider ${provider} succeeded!`);
        return { json, provider };
    }
    await editStatus(`❌ Text provider ${provider} failed.`);
  }
  return { json: null, provider: null };
}

/* ===================== IMAGE GENERATION ===================== */
// RESTORED: This function was accidentally deleted.
function buildImagePrompt(prod = {}) {
  const name = (prod.name || 'Unnamed Product').trim();
  const plan = (prod.plan && !/^(unknown|null|n\/a|na|none|-|\s*)$/i.test(String(prod.plan)))
    ? String(prod.plan).trim()
    : '';
  const desc = (prod.description || '').replace(/\s+/g, ' ').trim();
  const shortDesc = desc.length > 220 ? desc.slice(0, 220) + '…' : desc;

  return [
    `High-quality, detailed hero image for: ${name}${plan ? ' — ' + plan : ''}.`,
    shortDesc ? `Visual theme inspired by: ${shortDesc}.` : '',
    'No text, no watermarks, no logos, ultra realistic, 4K, photorealistic lighting, cinematic style'
  ]
  .filter(Boolean)
  .join(' ');
}

function gradientBackgroundSVG(width = 1024, height = 1024) {
  const svg = `<svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg"><defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="#111827"/><stop offset="50%" stop-color="#1f2937"/><stop offset="100%" stop-color="#374151"/></linearGradient></defs><rect width="100%" height="100%" fill="url(#g)"/></svg>`;
  return Buffer.from(svg);
}

const escXML = (s='') => String(s).replace(/[&<>]/g, c => ({'&':'&','<':'<','>':'>'}[c]));

async function composeTextOverBackground(backgroundBuffer, prod) {
  if (!_sharp) return backgroundBuffer;

  let base = _sharp(backgroundBuffer);
  let meta = await base.metadata().catch(() => ({}));
  let W = Math.max(1, meta.width || 1024);
  let H = Math.max(1, meta.height || 1024);

  if (!meta.width || !meta.height) {
    const tmp = await base.png().toBuffer();
    base = _sharp(tmp);
    meta = await base.metadata();
    W = Math.max(1, meta.width || 1024);
    H = Math.max(1, meta.height || 1024);
  }

  const titleRaw = String(prod?.name || 'Product').trim();
  const planText = (!prod?.plan || /^(unknown|null|n\/a|na|none|-|\s*)$/i.test(String(prod.plan))) ? '' : String(prod.plan);

  const scale = Math.min(W, H) / 1024;
  const clamp = (n, min, max) => Math.max(min, Math.min(max, n));
  let titleFont = clamp(Math.round(120 * scale), 28, 180);
  const planFont = clamp(Math.round(58 * scale), 18, 120);

  function wrapByChars(text, maxCharsPerLine) {
    const words = String(text||'').trim().split(/\s+/);
    const lines = [];
    let cur = '';
    for (const w of words) {
      const cand = cur ? `${cur} ${w}` : w;
      if (cand.length <= maxCharsPerLine) cur = cand;
      else { if (cur) lines.push(cur); cur = w; }
      if (lines.length >= 2) break;
    }
    if (cur && lines.length < 2) lines.push(cur);
    return lines;
  }

  const maxChars = Math.max(12, Math.round(18 * (W / 1024)));
  let titleLines = wrapByChars(titleRaw, maxChars);
  while (titleLines.join(' ').length > maxChars * 2 && titleFont > 28) {
    titleFont -= 2;
    titleLines = wrapByChars(titleRaw, maxChars);
  }
  titleLines = titleLines.slice(0, 2);

  const raster = await rasterizeTextOverlayPNG(W, H, titleLines, titleFont, planText, planFont);
  if (raster) {
    return base
      .composite([{ input: raster, left: 0, top: 0 }])
      .png()
      .toBuffer();
  }

  const lhTitle = 1.15;
  const titleBlockHeight = Math.round(titleFont * (titleLines.length + (titleLines.length - 1) * (lhTitle - 1)));
  const titleY = Math.round(H * 0.50) - Math.round(titleBlockHeight * 0.25);
  const minGapPx = Math.max(Math.round(titleFont * 0.22), Math.round(28 * scale));
  const lastTitleBaseline = titleY + Math.round((titleLines.length - 1) * (titleFont * lhTitle));
  const planY = planText ? (lastTitleBaseline + Math.round(titleFont * 0.9) + minGapPx) : null;

  const titleTspans = titleLines.map((line, i) =>
    i === 0
      ? `<tspan x="${W/2}" y="${titleY}">${escXML(line)}</tspan>`
      : `<tspan x="${W/2}" dy="${Math.round(titleFont * lhTitle)}">${escXML(line)}</tspan>`
  ).join('');

  const overlaySvg = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg">${SVG_FONT_STYLE_FILE}<rect width="100%" height="100%" fill="black" opacity="0.35"/><text class="title" text-anchor="middle" font-size="${titleFont}" fill="#FFFFFF">${titleTspans}</text>${planText ? `<text class="sub" x="${W/2}" y="${planY}" text-anchor="middle" font-size="${planFont}" fill="#E5E7EB">${escXML(planText)}</text>` : ''}</svg>`.trim();

  const overlayBuf = await _sharp(Buffer.from(overlaySvg)).png().toBuffer();

  return base
    .composite([{ input: overlayBuf, left: 0, top: 0 }])
    .png()
    .toBuffer();
}

/* ---- providers ---- */
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

async function generateImageFromHuggingFace(prompt) {
  if (!HF_KEY) { console.warn('[img] HF key missing'); return null; }
  try {
    const result = await hf.textToImage({
      model: HF_IMAGE_MODEL,
      inputs: prompt,
      parameters: {
        negative_prompt: 'blurry, ugly, deformed, noisy, plain, boring, text, watermark, signature',
        guidance_scale: 7.5, num_inference_steps: 28, width: 1024, height: 1024,
      },
    });
    return Buffer.from(await result.arrayBuffer());
  } catch (e) {
    console.warn(`[img] HF failed (${e?.httpResponse?.status||'??'}): ${e?.httpResponse?.body?.error || e.message}`);
    return null;
  }
}

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
    return Buffer.from(await imgRes.arrayBuffer());
  } catch (e) {
    console.warn('[img] DeepAI failed:', e.message);
    return null;
  }
}

async function generateImageFromCloudflare(prompt) {
  if (!CF_ACCOUNT_ID || !CF_API_TOKEN) {
    console.warn('[img] Cloudflare creds missing');
    return null;
  }
  const model = '@cf/black-forest-labs/flux-1-schnell';
  const safePrompt = [String(prompt||'').replace(/\banime|animation|character\b/gi, 'abstract motion graphics').replace(/\b(sexy|nsfw|nude|nudity)\b/gi, 'sfw').trim(), 'abstract geometric product background, shapes only, no people, no faces, no bodies, no text, SFW, corporate, clean'].join('. ');
  const url = `https://api.cloudflare.com/client/v4/accounts/${CF_ACCOUNT_ID}/ai/run/${model}`;
  const body = { prompt: safePrompt, negative_prompt: 'nsfw, nude, nudity, cleavage, erotic, sexual, suggestive, bikini, lingerie, skin, body, people, face, human, watermark, text, logo, hands, portrait, character, anime, cartoon, doll, ugly, deformed', width: 768, height: 768, num_steps: 4, guidance: 3.5 };

  const res = await fetch(url, {
    method: 'POST',
    headers: { Authorization: `Bearer ${CF_API_TOKEN}`, 'Content-Type': 'application/json', Accept: 'application/json' },
    body: JSON.stringify(body),
    signal: typeof AbortSignal?.timeout === 'function' ? AbortSignal.timeout(60000) : undefined
  });

  if (!res.ok) {
    const txt = await res.text().catch(() => '');
    if (/NSFW|safety|adult/i.test(txt)) return null;
    if (res.status === 429 || /rate|quota|limit/i.test(txt)) return null;
    throw new Error(`Cloudflare AI HTTP ${res.status}: ${txt.slice(0, 500)}`);
  }

  if ((res.headers.get('content-type') || '').startsWith('image/')) {
    return Buffer.from(await res.arrayBuffer());
  }

  let json;
  try { json = await res.json(); }
  catch {
    const txt = await res.text().catch(() => '');
    try { json = JSON.parse(txt); } catch { throw new Error(`Cloudflare returned non-image, non-JSON payload: ${txt.slice(0, 500)}`); }
  }

  if (json?.success === false) throw new Error(`Cloudflare AI error: ${(json.errors?.[0]?.message) || 'Unknown'}`);
  const b64 = json?.result?.image || json?.result?.images?.[0] || json?.result?.output?.[0] || json?.image;
  if (!b64) { console.warn('[img] Cloudflare JSON had no image field:', JSON.stringify(json).slice(0, 500)); return null; }
  return Buffer.from(b64, 'base64');
}

async function generateBackgroundWithOrder(prod, table, order = [], editStatus = async () => {}) {
  const prompt = buildImagePrompt(prod);
  console.log('[img] Using image prompt:', prompt);

  const defaultOrder = ['cloudflare', 'hf', 'deepai', 'pollinations'];
  order = order.length ? order : defaultOrder;

  for (const provider of order) {
    await editStatus(`➡️ Trying image provider: ${provider}...`);
    let buf = null;
    if (provider === 'hf') {
        buf = await tryWithRetries('image:hf', () => generateImageFromHuggingFace(prompt), IMAGE_RETRIES);
    } else if (provider === 'cloudflare') {
        buf = await tryWithRetries('image:cloudflare', () => generateImageFromCloudflare(prompt), IMAGE_RETRIES);
    } else if (provider === 'deepai') {
        buf = await tryWithRetries('image:deepai', () => generateImageFromDeepAI(prompt), IMAGE_RETRIES);
    } else if (provider === 'pollinations') {
        buf = await tryWithRetries('image:pollinations', () => generateImageFromPollinations(prompt), IMAGE_RETRIES);
    }

    if (buf?.length) {
        await editStatus(`✅ ${provider} succeeded! Rehosting image...`);
        return await rehostToSupabase(buf, `${prod.name}_${provider}.png`, table);
    }
    await editStatus(`❌ ${provider} failed. Trying next...`);
  }

  try {
    await editStatus('⚠️ All providers failed. Creating a fallback image...');
    const base = gradientBackgroundSVG();
    const finalBuf = await composeTextOverBackground(base, prod);
    return await rehostToSupabase(finalBuf, `${prod.name}_fallback.png`, table);
  } catch (e) {
    console.error('[img] Fallback generation failed', e);
    return null;
  }
}

/* ---- brand/og/search helpers ---- */
// RESTORED: This function was accidentally deleted.
function resolveBrandDomain(name = '') {
  if (!name) return null;
  const cleanName = shortBrandName({ name });
  const n = cleanName.toLowerCase().trim();
  const map = {
    v0: 'v0.dev', gamma: 'gamma.app', spotify: 'spotify.com', netflix: 'netflix.com', youtube: 'youtube.com',
    crunchyroll: 'crunchyroll.com', elevenlabs: 'elevenlabs.io', coursera: 'coursera.org', scribd: 'scribd.com',
    skillshare: 'skillshare.com', kittl: 'kittl.com', perplexity: 'perplexity.ai'
  };
  for (const k in map) {
      if (n.includes(k)) return map[k];
  }
  const slug = n.replace(/[^a-z0-9]/g, '');
  return slug.length >= 3 ? `${slug}.com` : null;
}

async function getOG(url) {
  try {
    const res = await fetch(url, { signal: typeof AbortSignal?.timeout==='function' ? AbortSignal.timeout(8000) : undefined });
    if (!res.ok) return {};
    const html = await res.text();
    const pick = (prop) => new RegExp(`<meta[^>]+property=["']${prop}["'][^>]+content=["']([^"']+)["']`, 'i').exec(html)?.[1];
    return { image: pick('og:image:secure_url') || pick('og:image') };
  } catch { return {}; }
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

async function tryBrandImages(prod, table) {
  const domain = resolveBrandDomain(prod.name);
  if (domain) {
    const fullUrl = (domain.startsWith('http') ? '' : 'https://') + domain;
    try {
      const og = await getOG(fullUrl);
      if (og.image) {
        return await rehostToSupabase(og.image, `${prod.name}.jpg`, table);
      }
    } catch (e) { console.warn(`[img] OG fetch for ${fullUrl} failed: ${e.message}`); }

    try {
      const res = await fetch(`https://api.brandfetch.io/v2/logo/${domain}`, {
        signal: typeof AbortSignal?.timeout==='function' ? AbortSignal.timeout(5000) : undefined,
      });
      if (res.ok) {
        const data = await res.json();
        const logo = data?.formats?.find(f => f.format === 'png') || data?.formats?.find(f => f.format === 'svg');
        if (logo?.src) {
          return await rehostToSupabase(logo.src, `${prod.name}_logo.png`, table);
        }
      }
    } catch (e) { console.warn(`[img] Brandfetch for ${domain} failed: ${e.message}`); }
  }

  const searchImageUrl = await findBestImageWithSearch(`${shortBrandName(prod)} logo png`);
  if (searchImageUrl) {
    try { return await rehostToSupabase(searchImageUrl, `${prod.name}_search.jpg`, table); }
    catch (e) { console.warn('[img] rehost search failed:', e.message); }
  }
  return null;
}

/* ---- DuckDuckGo + Wikipedia helpers ---- */
async function ddgSearchHTML(query, max = 8) {
  const url = `https://duckduckgo.com/html/?q=${encodeURIComponent(query)}&ia=web`;
  try {
    const res = await fetch(url, { headers: { 'user-agent': 'Mozilla/5.0' }, signal: typeof AbortSignal?.timeout === 'function' ? AbortSignal.timeout(12000) : undefined });
    if (!res.ok) return [];
    const html = await res.text();
    const links = Array.from(html.matchAll(/<a[^>]+class="result__a"[^>]+href="([^"]+)"/gi)).map(m => m[1]).filter(u => /^https?:\/\//i.test(u));
    return Array.from(new Set(links)).slice(0, max);
  } catch { return []; }
}
async function ddgSearchLite(query, max = 8) {
  const url = `https://duckduckgo.com/lite/?q=${encodeURIComponent(query)}`;
  try {
    const res = await fetch(url, { headers: { 'user-agent': 'Mozilla/5.0' }, signal: typeof AbortSignal?.timeout === 'function' ? AbortSignal.timeout(12000) : undefined });
    if (!res.ok) return [];
    const html = await res.text();
    const links = Array.from(html.matchAll(/<a href="(https?:\/\/[^"]+)"/gi)).map(m => m[1]).filter(Boolean);
    return Array.from(new Set(links)).slice(0, max);
  } catch { return []; }
}
async function wikiBestPage(query) {
  try {
    const url = `https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=${encodeURIComponent(query)}&srlimit=5&format=json`;
    const res = await fetch(url, { headers: { 'user-agent': 'Mozilla/5.0' }, signal: typeof AbortSignal?.timeout==='function' ? AbortSignal.timeout(12000) : undefined });
    if (!res.ok) return null;
    const data = await res.json();
    return data?.query?.search?.[0]?.title || null;
  } catch { return null; }
}
async function wikiExtractByTitle(title) {
  try {
    const url = `https://en.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&redirects=1&format=json&titles=${encodeURIComponent(title)}`;
    const res = await fetch(url, { headers: { 'user-agent': 'Mozilla/5.0' }, signal: typeof AbortSignal?.timeout==='function' ? AbortSignal.timeout(12000) : undefined });
    if (!res.ok) return '';
    const data = await res.json();
    return Object.values(data?.query?.pages || {})[0]?.extract?.slice(0, 12000) || '';
  } catch { return ''; }
}
async function searchWebForProduct(productName, plan) {
  const q = [productName, plan, 'price features premium plan'].filter(Boolean).join(' ');
  const urls = new Set();
  (await ddgSearchHTML(q, 8)).forEach(u => urls.add(u));
  (await ddgSearchLite(q, 8)).forEach(u => urls.add(u));

  const hosts = Array.from(urls).map(hostOf).filter(Boolean);
  const brand = String(productName||'').toLowerCase().replace(/[^a-z0-9]+/g,'');
  let bestHost = null, bestScore = Infinity;
  for (const h of hosts) {
    const base = stripTLD(h).toLowerCase().replace(/[^a-z0-9]+/g,'');
    const d = levenshtein(brand, base);
    if (d < bestScore) { bestScore = d; bestHost = h; }
  }
  if (bestHost && bestScore <= 3) {
    const base = `https://${bestHost}`;
    ['/','/pricing','/plans','/premium','/subscribe','/membership','/features','/help','/faq'].map(p => base.replace(/\/$/,'') + p).forEach(u => urls.add(u));
  }

  const chunks = [];
  let count = 0;
  for (const u of urls) {
    if (count >= 12) break;
    try {
      const { text } = await fetchWebsiteRaw(u);
      if (text?.length > 200) {
        chunks.push(`SOURCE: ${u}\n${text.slice(0, 4000)}`);
        count++;
      }
    } catch {}
  }

  let bundle = chunks.join('\n\n').slice(0, 20000);

  if (bundle.length < 800) {
    const title = await wikiBestPage(productName);
    if (title) {
      const wikiText = await wikiExtractByTitle(title);
      if (wikiText?.length > 400) {
        bundle += `\n\nSOURCE: https://en.wikipedia.org/wiki/${encodeURIComponent(title)}\n${wikiText}`;
      }
    }
  }

  console.log(`[text] evidence: ${bundle.length} chars from ${(bundle.match(/SOURCE:/g) || []).length} pages; official=${bestHost || '-'}`);
  return bundle;
}

/* ---- keyboards ---- */
const kbTextAPIs = Markup.inlineKeyboard([
  [Markup.button.callback('🦙 Pollinations (SearchGPT)', 'txtapi_pollinations')],
  [Markup.button.callback('🟪 Groq (Llama3)', 'txtapi_groq')],
  [Markup.button.callback('🔷 Gemini (rotation)', 'txtapi_gemini')],
  [Markup.button.callback('🤖 Auto (best → fallback)', 'txtapi_auto')],
  [Markup.button.callback('❌ Cancel', 'txtapi_cancel')],
]);

const kbBeforeImage = Markup.inlineKeyboard([
  [Markup.button.callback('✅ Generate Image', 'confirm_generate_image')],
  [Markup.button.callback('✏️ Edit Text', 'edit_text')],
  [Markup.button.callback('❌ Cancel', 'cancel')],
]);

const kbImageAPIs = Markup.inlineKeyboard([
  [Markup.button.callback('🖼️ Pollinations (Free)', 'imgapi_pollinations')],
  [Markup.button.callback('🤗 Hugging Face', 'imgapi_hf')],
  [Markup.button.callback('🟦 DeepAI', 'imgapi_deepai')],
  [Markup.button.callback('🟧 Cloudflare Workers AI', 'imgapi_cloudflare')],
  [Markup.button.callback('🤖 Auto (best effort)', 'imgapi_auto')],
  [Markup.button.callback('❌ Cancel', 'imgapi_cancel')],
]);

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

// FIXED: This function now truncates the description to avoid exceeding Telegram's caption limit.
function reviewMessage(prod, ai, table) {
  const tags = uniqMerge(prod.tags || [], ai.tags || []);
  const MAX_CAPTION_LEN = 1024;

  const fixedParts = [];
  fixedParts.push(`*Review before save*`);
  fixedParts.push(`*Name:* ${escapeMd(prod.name)}`);
  if (ok(prod.plan)) fixedParts.push(`*Plan:* ${escapeMd(prod.plan)}`);
  if (ok(prod.validity)) fixedParts.push(`*Validity:* ${escapeMd(prod.validity)}`);
  fixedParts.push(`*Price:* ${escapeMd(prod.price ? `₹${prod.price}` : '-')}`);
  
  const featuresPart = (ai.features?.length > 0)
    ? `\n*Key Features:*\n${ai.features.map(f => `- ${escapeMd(f)}`).join('\n')}`
    : '';

  const tagsPart = `\n*Tags:* ${escapeMd(tags.join(', ') || '-')}`;
  
  let productsPart = '';
  if (table === TABLES.products) {
    productsPart = `\n*MRP:* ${escapeMd(prod.originalPrice ?? '-')}\n*Stock:* ${escapeMd(prod.stock ?? '-')}\n*Category:* ${escapeMd(ai.category || '-')}\n*Subcategory:* ${escapeMd(ai.subcategory || '-')}`;
  }

  const imagePart = `\n*Image:* ${prod.image ? `[View Image](${prod.image})` : 'No Image'}`;

  // Calculate remaining length for the description
  const fixedLength = fixedParts.join('\n').length + featuresPart.length + tagsPart.length + productsPart.length + imagePart.length;
  const remainingLength = MAX_CAPTION_LEN - fixedLength - 30; // -30 for safety margin and "Description" title

  let description = ai.description || prod.description || '';
  if (description.length > remainingLength) {
    description = description.substring(0, remainingLength) + '... *(truncated)*';
  }
  
  fixedParts.push(`*Description:* ${description}`);

  // Re-assemble the final message
  const finalMessage = [ ...fixedParts, featuresPart, tagsPart, productsPart, imagePart ]
    .filter(Boolean) // Remove any empty parts
    .join('\n');

  return finalMessage;
}


/* --------------------- bot wiring --------------------- */
bot.use(session());
bot.use(async (ctx, next) => {
  if (!isAdmin(ctx)) return;
  if (!ctx.session) ctx.session = {};

  if (ctx.session.paused) {
    const isResumeCb = ctx.update?.callback_query?.data === 'resume_bot';
    const isStartCmd = !!ctx.message?.text?.startsWith('/start');
    const isAnyMessage = !!ctx.message;

    if (!(isResumeCb || isStartCmd || isAnyMessage)) return;
    ctx.session.paused = false;
    try { if (ctx.answerCbQuery) await ctx.answerCbQuery('Resumed'); } catch {}
    try { await ctx.reply('✅ Resumed.'); } catch {}
  }

  startIdleTimer(ctx);
  return next();
});

bot.start(async (ctx) => {
  if (!isAdmin(ctx)) return;
   ctx.session.paused = false;
  startIdleTimer(ctx);
  ctx.session = {};
  await ctx.reply('Welcome! Please choose which table you want to work with:', kbChooseTable);
});

bot.command('list', async (ctx) => {
  if (!isAdmin(ctx) || !ctx.session.table) return ctx.reply('Choose table first with /table');
  const { data, error } = await supabase.from(ctx.session.table).select('id,name,price,is_active').order('id', { ascending: false }).limit(12);
  if (error) return ctx.reply(`DB error: ${error.message}`);
  const items = data || [];
  if (!items.length) return ctx.reply('No items yet.');
  const msg = items.map((r, i) => `${i + 1}. ${r.name} — ₹${Number(r.price || 0).toLocaleString('en-IN')} — ${r.is_active ? '✅' : '⛔️'} (id: ${r.id})`).join('\n');
  ctx.reply('Latest:\n' + msg);
});

bot.command('smartadd', (ctx) => {
  if (!isAdmin(ctx) || !ctx.session.table) return replyMD(ctx, 'First choose a table with `/start`');
  ctx.session.mode = 'smart';
  ctx.session.await = 'blob';
  ctx.reply('Send the product text (can be messy). You may also attach a photo or include a URL.');
});

bot.command('toggle', async (ctx) => {
  if (!isAdmin(ctx) || !ctx.session.table) return ctx.reply('Choose table first with /table');
  const id = (ctx.message.text.split(' ')[1] || '').trim();
  if (!id) return ctx.reply('Usage: /toggle <id>');
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
  ctx.session.await = null;
  await ctx.answerCbQuery(`Table set to ${ctx.session.table}`);
  await ctx.deleteMessage().catch(()=>{});
   await replyMD(ctx, `✅ Table set to *${escapeMd(ctx.session.table)}*.\n\nUse /smartadd to begin.`);
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
  await ctx.reply('🏁 All done! Session cleared. Ready for a new task.', Markup.removeKeyboard());
  await ctx.reply('Please choose a table to begin:', kbChooseTable);
});

bot.action('resume_bot', async (ctx) => {
  if (!isAdmin(ctx)) return;
  await ctx.answerCbQuery('Resuming…').catch(()=>{});
  ctx.session.paused = false;
  try { await ctx.editMessageText('✅ Resumed.'); } catch {}
  startIdleTimer(ctx);
  if (!ctx.session.table) await ctx.reply('Pick a table to continue:', kbChooseTable);
});

function generateEditForm(reviewData) {
    const { prod, ai, table } = reviewData;
    const isProductsTable = table === TABLES.products;
    const lines = [
        `Name: ${prod.name || ''}`,
        `Plan: ${prod.plan || ''}`,
        `Validity: ${prod.validity || ''}`,
        `Price: ${prod.price || ''}`,
        `Description: ${prod.description || ai.description || ''}`,
        `Tags: ${(uniqMerge(prod.tags, ai.tags)).join(', ')}`,
        `Features: ${(ai.features || []).join('; ')}`
    ];
    if (isProductsTable) {
        lines.push(`OriginalPrice: ${prod.originalPrice || ''}`);
        lines.push(`Stock: ${prod.stock || ''}`);
        lines.push(`Category: ${prod.category || ai.category || ''}`);
        lines.push(`Subcategory: ${prod.subcategory || ai.subcategory || ''}`);
    }
    return lines.join('\n');
}

bot.action('edit_text', async (ctx) => {
  if (!ctx.session.review) return ctx.answerCbQuery();
  await ctx.answerCbQuery();
  await ctx.deleteMessage().catch(()=>{});
  const formText = generateEditForm(ctx.session.review);
  await ctx.replyWithHTML(
    'Copy the text below, edit the values, and then send the entire message back.\n\n' +
    '<b>Tips:</b>\n' +
    '- <b>Tags</b> are comma-separated (<code>,</code>).\n' +
    '- <b>Features</b> are semicolon-separated (<code>;</code>).\n\n' +
    `<code>${formText.replace(/</g, '<').replace(/>/g, '>')}</code>`
  );
  ctx.session.await = 'edit_all';
});

bot.action('change_image', async (ctx) => {
  if (!ctx.session.review) return ctx.answerCbQuery();
  await ctx.answerCbQuery();
  ctx.session.await = 'image_choice';
  await ctx.deleteMessage().catch(()=>{});
  await ctx.reply('Pick a generator, or send a new image URL / upload a photo:', kbImageAPIs);
});

function setTextOrder(ctx, order) {
  ctx.session.textOrder = order;
  ctx.session.await = null;
  ctx.answerCbQuery().catch(()=>{});
  ctx.deleteMessage().catch(()=>{});

  if (ctx.session.pendingSmart) {
    return resumeSmartAddAfterTextChoice(ctx);
  }
  return ctx.reply('✅ Text provider set.');
}

bot.action('txtapi_pollinations', (ctx)=>setTextOrder(ctx, ['pollinations','groq','gemini']));
bot.action('txtapi_groq',         (ctx)=>setTextOrder(ctx, ['groq','gemini','pollinations']));
bot.action('txtapi_gemini',       (ctx)=>setTextOrder(ctx, ['gemini','groq','pollinations']));
bot.action('txtapi_auto',         (ctx)=>setTextOrder(ctx, ['groq','gemini','pollinations']));
bot.action('txtapi_cancel', async (ctx)=>{
  await ctx.answerCbQuery();
  await ctx.deleteMessage().catch(()=>{});
  ctx.session.await = null;
  ctx.session.pendingSmart = null;
  await ctx.reply('Product addition cancelled.');
  await ctx.reply('What would you like to do next?', kbAfterTask);
});

const smartAddHandler = async (ctx) => {
  if (!ctx.session.table) { await ctx.reply('Please choose a table first.', kbChooseTable); return; }

  const text = ctx.message?.text || '';
  const statusMsg = await ctx.reply('⏳ Analyzing your request...');

  const editStatus = async (text) => {
      try {
          await ctx.telegram.editMessageText(ctx.chat.id, statusMsg.message_id, null, text);
      } catch (e) {
          console.warn('Could not edit status message:', e.message);
      }
  };

  try {
    const urlMatch = Array.from(text.matchAll(URL_RX)).map(m => m[0])[0] || null;
    const guessedName = text.split('\n')[0].trim();
    
    await editStatus('🔎 Searching the web for product info...');
    let domain = urlMatch || resolveBrandDomain(guessedName);
    if (!urlMatch && !domain) {
      try {
        const bestHost = await (async function pickOfficialDomainFromSearch(brandName){
          const urls = new Set();
          try { (await ddgSearchHTML(brandName, 10)).forEach(u=>urls.add(u)); } catch {}
          try { (await ddgSearchLite(brandName, 10)).forEach(u=>urls.add(u)); } catch {}
          const hosts = Array.from(urls).map(hostOf).filter(Boolean);
          if (!hosts.length) return null;
          const brand = String(brandName||'').toLowerCase().replace(/[^a-z0-9]+/g,'');
          let best=null,bestScore=Infinity;
          for (const h of hosts) {
            const base = stripTLD(h).toLowerCase().replace(/[^a-z0-9]+/g,'');
            const d = levenshtein(brand, base);
            if (d < bestScore) { bestScore = d; best = h; }
          }
          return bestScore <= 3 ? best : null;
        })(guessedName);
        if (bestHost) domain = `https://${bestHost}`;
      } catch {}
    }

    let websiteContent = '', ogImageFromPage = null;
    if (domain) {
      const { html, text: pageText } = await fetchWebsiteRaw(domain.startsWith('http') ? domain : `https://${domain}`);
      websiteContent = pageText;
      if (html) ogImageFromPage = extractMetaTags(html).ogImage || null;
    }

    if (!ctx.session.textOrder) {
      ctx.session.await = 'choose_text_api';
      ctx.session.pendingSmart = { text, websiteContent, ogImageFromPage };
      await ctx.reply('Choose which **text model** to fill product details (I’ll retry and fallback automatically):', kbTextAPIs);
      await ctx.telegram.deleteMessage(ctx.chat.id, statusMsg.message_id).catch(()=>{});
      return;
    }
    
    const aiData = await enrichWithAI(text, websiteContent, ctx.session.textOrder, editStatus);

    const prod = { ...aiData, image: null };
    ctx.session.review = { prod, ai: aiData, table: ctx.session.table, ogImageFromPage };
    ctx.session.mode = null;

    const msg = `📝 *Parsed (Review & Edit)*\n\n*Name:* ${escapeMd(aiData.name)}\n*Plan:* ${escapeMd(aiData.plan)}\n*Validity:* ${escapeMd(aiData.validity)}\n*Price:* ${escapeMd(aiData.price ?? '-')}\n*Category:* ${escapeMd(aiData.category)}\n\nTap *Generate Image* when the text looks good.`;
    await replyMD(ctx, msg, kbBeforeImage);

  } catch (e) {
    console.error('Smart add flow failed:', e);
    await ctx.reply(`❌ An error occurred: ${e.message}. Please try again.`);
    ctx.session = { table: ctx.session.table };
  } finally {
    await ctx.telegram.deleteMessage(ctx.chat.id, statusMsg.message_id).catch(()=>{});
  }
};

async function resumeSmartAddAfterTextChoice(ctx) {
  const pending = ctx.session.pendingSmart;
  if (!pending) return ctx.reply('No pending product to resume. Send /smartadd again.');

  const statusMsg = await ctx.reply('🤖 Asking AI to fill in the details...');
  const editStatus = async (text) => {
      try { await ctx.telegram.editMessageText(ctx.chat.id, statusMsg.message_id, null, text); }
      catch (e) { console.warn('Could not edit status message:', e.message); }
  };

  ctx.session.pendingSmart = null;
  const { text, websiteContent, ogImageFromPage } = pending;

  try {
    const aiData = await enrichWithAI(text, websiteContent, ctx.session.textOrder, editStatus);
    const prod = { ...aiData, image: null };
    ctx.session.review = { prod, ai: aiData, table: ctx.session.table, ogImageFromPage };
    ctx.session.mode = null;

    const msg = `📝 *Parsed (Review & Edit)*\n\n*Name:* ${escapeMd(aiData.name)}\n*Plan:* ${escapeMd(aiData.plan)}\n*Validity:* ${escapeMd(aiData.validity)}\n*Price:* ${escapeMd(aiData.price ?? '-')}\n*Category:* ${escapeMd(aiData.category)}\n\nTap *Generate Image* when the text looks good.`;
    await replyMD(ctx, msg, kbBeforeImage);
  } catch (e) {
      console.error('Smart add resume flow failed:', e);
      await ctx.reply(`❌ An error occurred during AI enrichment: ${e.message}.`);
  } finally {
      await ctx.telegram.deleteMessage(ctx.chat.id, statusMsg.message_id).catch(()=>{});
  }
}

async function presentReview(ctx) {
  if (!ctx.session.review) return;
  const { prod, ai, table } = ctx.session.review;
  const caption = reviewMessage(prod, ai, table);

  let photoUrl = prod.image;
  if (photoUrl && /api\.telegram\.org\/file\/bot/i.test(photoUrl)) {
    try {
      photoUrl = await rehostToSupabase(photoUrl, `${sanitizeForFilename(prod.name || 'product')}.jpg`, table);
      ctx.session.review.prod.image = photoUrl;
    } catch (e) {
      console.warn('[presentReview] rehost failed:', e.message);
      photoUrl = null;
    }
  }

  if(ctx.callbackQuery?.message?.message_id) {
    await ctx.deleteMessage().catch(()=>{});
  }

  if (photoUrl) {
    try {
      await ctx.replyWithPhoto(
        { url: photoUrl },
        { caption, parse_mode: 'Markdown', reply_markup: kbConfirm.reply_markup }
      );
    } catch (e) {
      console.warn('[presentReview] sendPhoto failed:', e.message);
      // If sending with photo fails due to caption length, send as separate messages
      if (e.message.includes('caption is too long')) {
          await ctx.replyWithPhoto({ url: photoUrl });
          await replyMD(ctx, caption, kbConfirm);
      } else {
          await replyMD(ctx, caption + '\n_(image preview unavailable)_', kbConfirm);
      }
    }
  } else {
    await replyMD(ctx, caption, kbConfirm);
  }
}

async function applyAllEdits(ctx, text) {
    if (!ctx.session.review) return;
    const { prod, ai, table } = ctx.session.review;
    const lines = text.split('\n');

    for (const line of lines) {
        const i = line.indexOf(':');
        if (i === -1) continue;
        const key = line.slice(0, i).trim().toLowerCase().replace(/ /g, '');
        const val = line.slice(i + 1).trim();

        switch (key) {
            case 'name': prod.name = val; break;
            case 'plan': prod.plan = val; break;
            case 'validity': prod.validity = val; break;
            case 'price': prod.price = parsePrice(val); break;
            case 'description': prod.description = val; break;
            case 'tags': prod.tags = val.split(',').map(t => t.trim()).filter(Boolean); break;
            case 'features': ai.features = val.split(';').map(f => f.trim()).filter(Boolean); break;
            case 'originalprice': if(table === TABLES.products) prod.originalPrice = parsePrice(val); break;
            case 'stock': if(table === TABLES.products) prod.stock = parsePrice(val); break;
            case 'category': if(table === TABLES.products) prod.category = val; break;
            case 'subcategory': if(table === TABLES.products) prod.subcategory = val; break;
        }
    }

    ctx.session.await = null;
    await ctx.reply('✅ All fields updated.');
    await presentReview(ctx);
}

bot.on('text', async (ctx, next) => {
  if (!isAdmin(ctx)) return;
  const text = ctx.message?.text || '';
  if (ctx.session.await === 'choose_text_api') return;
  if (!ctx.session.table && !text.startsWith('/')) return ctx.reply('Welcome! To get started, please choose a table.', kbChooseTable);
  if (ctx.session.await === 'edit_all') return applyAllEdits(ctx, text);

  if (ctx.session.await === 'image_choice' && text.startsWith('http')) {
    await ctx.reply('🔗 Got it. Rehosting your image URL...');
    try {
      const imageUrl = await rehostToSupabase(text, `${ctx.session.review?.prod?.name || 'product'}.jpg`, ctx.session.review.table);
      ctx.session.review.prod.image = imageUrl;
      ctx.session.await = null;
      await presentReview(ctx);
    } catch (e) {
      await ctx.reply('❌ That URL didn’t work. Please try another one, or upload a photo.');
    }
    return;
  }

  if (ctx.session.mode === 'smart' && ctx.session.await === 'blob') return await smartAddHandler(ctx);
  if (ctx.session.table && !text.startsWith('/') && (text.includes('\n') || text.length > 40)) return await smartAddHandler(ctx);

  return next();
});

bot.action('confirm_generate_image', async (ctx) => {
  if (!ctx.session?.review) return ctx.answerCbQuery();
  await ctx.answerCbQuery();
  await ctx.deleteMessage().catch(()=>{});

  const { prod, table } = ctx.session.review;
  const statusMsg = await ctx.reply('🎯 Looking for an official brand image first…');

  try {
      if (!prod.image) {
        const hosted = await tryBrandImages(prod, table);
        if (hosted) {
          prod.image = hosted;
          await presentReview(ctx);
          return;
        }
      }
      ctx.session.await = 'image_choice';
      await ctx.reply('No official image found. Choose a generator or send a URL/upload:', kbImageAPIs);
  } catch (e) {
      console.error('confirm_generate_image failed:', e);
      await ctx.reply(`❌ An error occurred: ${e.message}`);
  } finally {
      await ctx.telegram.deleteMessage(ctx.chat.id, statusMsg.message_id).catch(() => {});
  }
});

async function handleImageChoice(ctx, order) {
  if (!ctx.session.review) return ctx.answerCbQuery();
  await ctx.answerCbQuery();
  await ctx.deleteMessage().catch(()=>{});
  const statusMsg = await ctx.reply('🎨 Starting image generation...');
  const editStatus = async (text) => {
    try { await ctx.telegram.editMessageText(ctx.chat.id, statusMsg.message_id, null, text); }
    catch (e) { console.warn('Could not edit status message:', e.message); }
  };

  try {
      const { prod, table } = ctx.session.review;
      const hosted = await generateBackgroundWithOrder(prod, table, order, editStatus);
      
      if (hosted) {
          ctx.session.review.prod.image = hosted;
          ctx.session.await = null;
      }
      await presentReview(ctx);
  } catch(e) {
      console.error('Image choice handling failed:', e);
      await ctx.reply(`❌ An error occurred during image generation: ${e.message}`);
  } finally {
      await ctx.telegram.deleteMessage(ctx.chat.id, statusMsg.message_id).catch(()=>{});
  }
}

bot.action('imgapi_pollinations', (ctx)=>handleImageChoice(ctx, ['pollinations','hf','deepai']));
bot.action('imgapi_hf', (ctx)=>handleImageChoice(ctx, ['hf','pollinations','deepai']));
bot.action('imgapi_deepai', (ctx)=>handleImageChoice(ctx, ['deepai','pollinations','hf']));
bot.action('imgapi_cloudflare', (ctx)=>handleImageChoice(ctx, ['cloudflare','pollinations','hf','deepai']));
bot.action('imgapi_auto', (ctx)=>handleImageChoice(ctx, ['cloudflare','hf','deepai','pollinations']));
bot.action('imgapi_cancel', async (ctx) => {
  await ctx.answerCbQuery();
  await ctx.deleteMessage().catch(()=>{});
  await ctx.reply('Image generation cancelled.');
  ctx.session.await = null;
  if (ctx.session.review) await presentReview(ctx);
});

async function processIncomingImage(ctx, fileId, filenameHint = 'upload.jpg') {
  try {
    const table = ctx.session?.review?.table || ctx.session?.table;
    if (!table) return await ctx.reply('Please choose a table first.', kbChooseTable);
    const fileUrl = await tgFileUrl(fileId);
    await ctx.reply('🖼️ Rehosting your image...');
    const hosted = await rehostToSupabase(fileUrl, `${sanitizeForFilename(ctx.session?.review?.prod?.name || filenameHint)}`, table);
    if (ctx.session?.review?.prod) {
      ctx.session.review.prod.image = hosted;
      if (ctx.session.await === 'image_choice') ctx.session.await = null;
      await presentReview(ctx);
    } else {
      ctx.session.lastUploadedImage = hosted;
      await ctx.reply('✅ Image uploaded. I’ll use it when we get to the review.');
    }
  } catch (e) {
    console.error('[upload] manual image failed:', e);
    await ctx.reply(`❌ Could not process the image: ${e.message}`);
  }
}

bot.on('photo', (ctx) => isAdmin(ctx) && processIncomingImage(ctx, ctx.message.photo.pop().file_id, 'upload.jpg'));
bot.on('document', (ctx) => isAdmin(ctx) && ctx.message.document.mime_type?.startsWith('image/') && processIncomingImage(ctx, ctx.message.document.file_id, ctx.message.document.file_name));

bot.action('cancel', (ctx) => {
  ctx.answerCbQuery();
  ctx.session.review = null;
  ctx.session.mode = null;
  ctx.deleteMessage().catch(()=>{});
  ctx.reply('Cancelled.', kbAfterTask);
});

bot.action('save', async (ctx) => {
  if (!isAdmin(ctx) || !ctx.session?.review) return ctx.answerCbQuery();
  await ctx.answerCbQuery();
  const { prod, ai, table, updateId } = ctx.session.review;
  const isProducts = table === TABLES.products;

  const data = isProducts
    ? { name: prod.name, plan: prod.plan || ai.plan || null, validity: prod.validity || ai.validity || null, price: prod.price || ai.price || null, originalPrice: prod.originalPrice || null, description: prod.description || ai.description || null, category: prod.category || ai.category || null, subcategory: prod.subcategory || ai.subcategory || null, stock: prod.stock || null, tags: uniqMerge(prod.tags, ai.tags), features: ai.features || [], image: prod.image }
    : { name: prod.name, plan: prod.plan || ai.plan || null, validity: prod.validity || ai.validity || null, description: prod.description || ai.description || null, price: prod.price || ai.price || null, tags: uniqMerge(prod.tags, ai.tags), features: ai.features || [], image_url: prod.image };
  
  // Conditionally add is_active only when inserting a new product
  if (!updateId) {
      data.is_active = true;
  }

  try {
    await ctx.deleteMessage().catch(()=>{});
    let query;
    if (updateId) {
      query = supabase.from(table).update(data).eq('id', updateId);
    } else {
      const { data: existing } = await supabase.from(table).select('id').eq('name', prod.name).eq('price', prod.price || ai.price || null).maybeSingle();
      if (existing) {
        await ctx.reply(`⚠️ This product already exists (ID: ${existing.id}). Use \`/update ${existing.id}\` if you want to change it.`, kbAfterTask);
        return;
      }
      query = supabase.from(table).insert([data]);
    }
    
    const { error } = await query;
    if (error) throw error;
    
    await ctx.reply(`✅ Product *${updateId ? 'updated' : 'added'}* successfully.`, {parse_mode: 'Markdown'});
    ctx.session.review = null; ctx.session.await = null; ctx.session.mode = null;
    await ctx.reply('What next?', kbAfterTask);
  } catch (err) {
    console.error(err);
    await ctx.reply(`❌ Error saving: ${err.message}`);
  }
});

function rowToReview(table, row) {
  const isProducts = table === TABLES.products;
  const imgCol = isProducts ? 'image' : 'image_url';
  const prod = {
    name: row.name || '', plan: row.plan ?? null, validity: row.validity ?? null, price: row.price ?? null,
    originalPrice: isProducts ? (row.originalPrice ?? null) : undefined, stock: isProducts ? (row.stock ?? null) : undefined,
    description: row.description || '', tags: Array.isArray(row.tags) ? row.tags : (row.tags ? String(row.tags).split(',').map(s=>s.trim()) : []),
    image: row[imgCol] || null, category: isProducts ? (row.category || null) : undefined, subcategory: isProducts ? (row.subcategory || null) : undefined,
    is_active: row.is_active,
  };
  const ai = {
    description: row.description || '', category: isProducts ? (row.category || 'Download') : 'Download', subcategory: isProducts ? (row.subcategory || 'unknown') : 'unknown',
    tags: Array.isArray(row.tags) ? row.tags : [], features: Array.isArray(row?.features) ? row.features : [],
    name: row.name || '', plan: row.plan ?? 'unknown', validity: row.validity ?? 'unknown', price: row.price ?? null,
  };
  return { prod, ai };
}

bot.command('update', async (ctx) => {
  if (!isAdmin(ctx) || !ctx.session.table) return ctx.reply('Choose table first with /table');
  const id = (ctx.message?.text || '').trim().split(/\s+/)[1];
  if (!id) return ctx.reply('Usage: /update <id>');

  const selCols = ctx.session.table === TABLES.products
    ? 'id,name,plan,validity,price,originalPrice,description,category,subcategory,stock,tags,features,image,is_active'
    : 'id,name,plan,validity,description,price,tags,features,image_url,is_active';

  const { data: row, error } = await supabase.from(ctx.session.table).select(selCols).eq('id', id).maybeSingle();
  if (error) return ctx.reply(`DB error: ${error.message}`);
  if (!row) return ctx.reply('Not found.');
  const { prod, ai } = rowToReview(ctx.session.table, row);
  ctx.session.review = { prod, ai, table: ctx.session.table, updateId: row.id, ogImageFromPage: null };
  ctx.session.mode = null; ctx.session.await = null;
  await presentReview(ctx);
});

bot.catch((err, ctx) => {
  console.error(`Bot error for user ${ctx.from?.id}:`, err);
  try { if (isAdmin(ctx)) ctx.reply(`❌ ${err.message || "Unexpected error"}`); } catch {}
});

const setBotCommands = async () => {
    try {
        await bot.telegram.setMyCommands([
            { command: 'start', description: 'Restart the bot & choose table' },
            { command: 'smartadd', description: '⚡️ Add a product with AI' },
            { command: 'list', description: '📄 List recent products' },
            { command: 'update', description: '✏️ Update a product (e.g., /update 123)' },
            { command: 'toggle', description: '✅/⛔️ Toggle product status (e.g., /toggle 123)' }
        ]);
        console.log('Bot commands have been set successfully.');
    } catch (e) {
        console.error('Failed to set bot commands:', e);
    }
};

bot.command('setcommands', async (ctx) => {
    if (!isAdmin(ctx)) return;
    await setBotCommands();
    await ctx.reply('✅ Bot commands have been manually refreshed.');
});

if (process.env.MODE === "webhook") {
  console.log("Bot running in webhook mode...");
  setBotCommands();
  module.exports = async (req, res) => {
    try {
      await bot.handleUpdate(req.body, res);
    } catch (e) {
      console.error("Error handling webhook:", e);
      res.statusCode = 500;
      res.end("Error processing update");
    }
  };
} else {
  // Local or Railway: long-polling
  (async () => {
    try {
      await bot.launch({ dropPendingUpdates: true });
      await setBotCommands();
      console.log("🤖 Bot running in long-polling mode...");
    } catch (e) {
      console.error("Failed to launch bot in long-polling mode:", e);
    }
  })();

  const shutdown = async (signal) => {
    console.log(`\n${signal} received. Stopping bot...`);
    try {
      await bot.stop();
      process.exit(0);
    } catch (e) {
      console.error("Error on shutdown:", e);
      process.exit(1);
    }
  };
  process.once("SIGINT", () => shutdown("SIGINT"));
  process.once("SIGTERM", () => shutdown("SIGTERM"));
}
