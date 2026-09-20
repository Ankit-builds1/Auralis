import { useState, useRef } from 'react';

const BACKEND_URL = `${import.meta.env.VITE_BACKEND_URL}/webrtc/offer`;

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

    const pc = new RTCPeerConnection({
      iceServers: [
        { urls: 'stun:stun.l.google.com:19302' },
        {
          urls: 'turn:openrelay.metered.ca:80',
          username: 'openrelayproject',
          credential: 'openrelayproject'
        }
      ]
    });
    pcRef.current = pc;

    streamRef.current.getTracks().forEach(track => {
      pc.addTrack(track, streamRef.current);
    });

    pc.onconnectionstatechange = () => {
      console.log('Connection state:', pc.connectionState);
      setStatus(`webrtc: ${pc.connectionState}`);
    };

    pc.ontrack = (event) => {
      console.log('Received remote track:', event.track);
      setStatus('receiving remote audio');
    };

    try {
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      console.log('Generated SDP offer:', offer);
      setStatus('sending offer to backend...');

      const res = await fetch(BACKEND_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sdp: offer.sdp,
          type: offer.type,
        }),
      });

      if (!res.ok) {
        throw new Error(`Backend responded with ${res.status}`);
      }

      const answer = await res.json();
      console.log('Received SDP answer:', answer);

      await pc.setRemoteDescription(answer);
      setStatus('handshake complete — connected');
    } catch (err) {
      console.error('WebRTC handshake failed:', err);
      setStatus(`error: ${err.message}`);
    }
  };

  const stopMic = () => {
    streamRef.current?.getTracks().forEach(track => track.stop());
    pcRef.current?.close();
    setStatus('idle');
  };

  return (
    <div style={{ padding: 40, fontFamily: 'sans-serif' }}>
      <h1>Auralis — Frontend Day 3</h1>
      <p>Status: {status}</p>
      <button onClick={startMic}>Start Mic</button>
      <button onClick={connectWebRTC}>Connect to Backend</button>
      <button onClick={stopMic}>Stop</button>
    </div>
  );
}

export default App;