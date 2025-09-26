'use strict';

const bot = require('../bot');  // import the Telegraf bot

// Use Telegraf's webhookCallback directly
const handleUpdate = bot.webhookCallback('/api/telegram');

module.exports = async (req, res) => {
  if (req.method === 'GET') return res.status(200).send('OK');
  if (req.method !== 'POST') return res.status(405).send('Method Not Allowed');

  try {
    if (typeof req.body === 'string') req.body = JSON.parse(req.body);
    await handleUpdate(req, res);
    if (!res.writableEnded) res.status(200).end();
  } catch (err) {
    console.error('[telegram webhook error]', err);
    if (!res.writableEnded) res.status(500).send('Error processing update');
  }
};
