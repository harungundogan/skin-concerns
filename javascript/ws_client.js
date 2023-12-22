//Web Socket client
const WebSocket = require('ws');

const wss = new WebSocket.Server({ port: 8000 });
const socket = new WebSocket('ws://localhost:8000/video_feed');
wss.on('connection', (ws) => {
  ws.on('message', (message) => {
    console.log(`Received: ${message}`);
  });

  setInterval(() => {
    ws.send('Frame data...');
  }, 1000);
});

socket.onclose = () => {
  console.log('WebSocket connection closed.');
};
