// api/telegram.js
'use strict';

const bot = require('../bot');
const { webhookCallback } = require('telegraf');

const handleUpdate = webhookCallback(bot, 'http'); // Node req/res style

module.exports = async (req, res) => {
  if (req.method === 'GET') return res.status(200).send('OK');
  if (req.method !== 'POST') return res.status(405).send('Method Not Allowed');

  try {
    await handleUpdate(req, res); // Telegraf processes the update
    // If Telegraf already wrote the response, just ensure the request ends:
    if (!res.writableEnded) res.status(200).end();
  } catch (err) {
    console.error('webhook error', err);
    // Always 200 to keep Telegram happy; your bot can log failures
    if (!res.writableEnded) res.status(200).end();
  }
};
