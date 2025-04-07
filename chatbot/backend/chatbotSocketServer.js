const WebSocket = require('ws');
const express = require('express');
const cors = require('cors');
const fetch = (...args) => import('node-fetch').then(({ default: fetch }) => fetch(...args)); // Node 18+

const app = express();
app.use(cors());

const server = app.listen(4000, () => {
  console.log('🔌 WebSocket Server running on http://localhost:4000');
});

const wss = new WebSocket.Server({ server });

wss.on('connection', (ws) => {
  console.log('🔁 New client connected');

  ws.on('message', async (message) => {
    try {
      const data = JSON.parse(message);
      console.log('📩 Received from client:', data.message);

      const response = await fetch('http://localhost:8000/predict-nlp', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: data.message }),
      });

      if (!response.ok) {
        const text = await response.text(); // Get plain error text
        console.error('❌ FastAPI error:', text);
        ws.send(JSON.stringify({ message: '⚠️ Error: FastAPI NLP prediction failed' }));
        return;
      }

      const result = await response.json();
      console.log('📦 FastAPI returned:', result);

      // Use new keys from FastAPI response
      const reply = result.message || `⚠️ Threat Detected: ${result.class_name} (Code: ${result.class_code})`;

      ws.send(JSON.stringify({
        message: reply,
        class_name: result.class_name,
        class_code: result.class_code,
        confidence: result.confidence
      }));

    } catch (err) {
      console.error('❌ Error:', err);
      ws.send(JSON.stringify({ message: '⚠️ Error processing your message' }));
    }
  });
});
