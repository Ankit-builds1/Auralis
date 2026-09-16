import { useState, useRef } from 'react';

function App() {
  const [status, setStatus] = useState('idle');
  const streamRef = useRef(null);

  const startMic = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      setStatus('mic active');
      console.log('Got audio tracks:', stream.getAudioTracks());
    } catch (err) {
      console.error('Mic error:', err);
      setStatus('mic denied/error');
    }
  };

  const stopMic = () => {
    streamRef.current?.getTracks().forEach(track => track.stop());
    setStatus('idle');
  };

  return (
    <div style={{ padding: 40, fontFamily: 'sans-serif' }}>
      <h1>Auralis — Frontend Day 1</h1>
      <p>Status: {status}</p>
      <button onClick={startMic}>Start Mic</button>
      <button onClick={stopMic}>Stop Mic</button>
    </div>
  );
}

export default App;