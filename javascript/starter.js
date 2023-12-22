const WebSocket = require('ws');

const wss = new WebSocket.Server({ port: 8000 });
const socket = new WebSocket('ws://localhost:8000/video_feed');
wss.on('connection', (ws) => {
  ws.on('message', (message) => {
    console.log(`Received: ${message}`);
  });

  // Simulate sending frames (replace this with your logic to send video frames)
  setInterval(() => {
    ws.send('Frame data...'); // Send frame data as a string or buffer
  }, 1000); // Adjust the interval as needed for video frame rate
});

socket.onclose = () => {
  console.log('WebSocket connection closed.');
};

/*const videoElement = document.getElementById('videoElement');
const ws = new WebSocket('ws://localhost:8000/video_feed');

ws.onmessage = async (event) => {
    const blob = event.data;
    const url = URL.createObjectURL(blob);
    videoElement.src = url;
};*/