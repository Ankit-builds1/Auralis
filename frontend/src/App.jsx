import { useState, useRef } from 'react';

function App() {
  const [status, setStatus] = useState('idle');
  const streamRef = useRef(null);
  const pcRef = useRef(null);

  const startMic = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      setStatus('mic active');
    } catch (err) {
      console.error('Mic error:', err);
      setStatus('mic denied/error');
    }
  };

  const connectWebRTC = async () => {
    if (!streamRef.current) {
      setStatus('start mic first');
      return;
    }

    const pc = new RTCPeerConnection();
    pcRef.current = pc;

    streamRef.current.getTracks().forEach(track => {
      pc.addTrack(track, streamRef.current);
    });

    pc.onconnectionstatechange = () => {
      console.log('Connection state:', pc.connectionState);
      setStatus(`webrtc: ${pc.connectionState}`);
    };

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    console.log('Generated SDP offer:', offer);

    setStatus('offer created — check console');

    // TODO Day 3: send this offer to Siddhant's backend endpoint
  };

  const stopMic = () => {
    streamRef.current?.getTracks().forEach(track => track.stop());
    pcRef.current?.close();
    setStatus('idle');
  };

  return (
    <div style={{ padding: 40, fontFamily: 'sans-serif' }}>
      <h1>Auralis — Frontend Day 2</h1>
      <p>Status: {status}</p>
      <button onClick={startMic}>Start Mic</button>
      <button onClick={connectWebRTC}>Create WebRTC Offer</button>
      <button onClick={stopMic}>Stop</button>
    </div>
  );
}

export default App;