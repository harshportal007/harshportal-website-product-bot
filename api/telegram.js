'use strict';

const { Telegraf } = require('telegraf');
const bot = require('../bot'); // this is your instance from bot.js

module.exports = async (req, res) => {
  if (req.method === 'GET') return res.status(200).send('OK');
  if (req.method !== 'POST') return res.status(405).send('Method Not Allowed');

  try {
    if (typeof req.body === 'string') req.body = JSON.parse(req.body);

    // Telegraf webhook callback expects (ctx, next) style
    const webhookCb = bot.webhookCallback('/api/telegram');
    await webhookCb(req, res);

    if (!res.writableEnded) res.status(200).end();
  } catch (err) {
    console.error('❌ webhook error:', err);
    if (!res.writableEnded) res.status(200).end();
  }
};
