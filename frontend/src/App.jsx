import { useState, useRef } from 'react';

const BACKEND_URL = `${import.meta.env.VITE_BACKEND_URL}/webrtc/offer`;

function App() {
  const [status, setStatus] = useState('idle');
  const [error, setError] = useState(null);
  const [isConnecting, setIsConnecting] = useState(false);
  const [vadState, setVadState] = useState('idle'); // idle | listening | speaking

  const streamRef = useRef(null);
  const pcRef = useRef(null);
  const audioRef = useRef(null);
  const dataChannelRef = useRef(null);

  const startMic = async () => {
    setError(null);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: true,
      });

      streamRef.current = stream;
      setStatus('mic active');
    } catch (err) {
      console.error('Mic error:', err);

      if (err.name === 'NotAllowedError') {
        setError(
          'Microphone permission denied. Please allow mic access in your browser settings.'
        );
      } else if (err.name === 'NotFoundError') {
        setError(
          'No microphone found. Please connect a microphone and try again.'
        );
      } else {
        setError(`Could not access microphone: ${err.message}`);
      }

      setStatus('mic unavailable');
    }
  };

  const connectWebRTC = async () => {
    setError(null);

    if (!streamRef.current) {
      setError('Start the microphone before connecting.');
      return;
    }

    if (!import.meta.env.VITE_BACKEND_URL) {
      setError('Backend URL is not configured. Check your .env file.');
      return;
    }

    setIsConnecting(true);

    const pc = new RTCPeerConnection({
      iceServers: [
        {
          urls: 'stun:stun.l.google.com:19302',
        },
        {
          urls: 'turn:openrelay.metered.ca:80',
          username: 'openrelayproject',
          credential: 'openrelayproject',
        },
      ],
    });

    pcRef.current = pc;

    // --------------------------------------------------
    // ICE STATE LOGGING
    // --------------------------------------------------

    pc.oniceconnectionstatechange = () => {
      console.log(
        'ICE connection state:',
        pc.iceConnectionState
      );
    };

    pc.onicegatheringstatechange = () => {
      console.log(
        'ICE gathering state:',
        pc.iceGatheringState
      );
    };

    // --------------------------------------------------
    // DATA CHANNEL FOR VAD
    // --------------------------------------------------

    const dataChannel = pc.createDataChannel('vad-state');
    dataChannelRef.current = dataChannel;

    dataChannel.onopen = () => {
      console.log('Data channel open');
    };

    dataChannel.onmessage = (event) => {
      console.log('VAD state message:', event.data);

      try {
        const data = JSON.parse(event.data);

        if (data.vad_state) {
          setVadState(data.vad_state);
        }
      } catch (err) {
        console.error('Failed to parse VAD message:', err);
      }
    };

    dataChannel.onerror = (err) => {
      console.error('Data channel error:', err);
    };

    // Also accept a data channel opened by the backend
    pc.ondatachannel = (event) => {
      const channel = event.channel;

      dataChannelRef.current = channel;

      channel.onmessage = (msgEvent) => {
        console.log(
          'VAD state message (backend channel):',
          msgEvent.data
        );

        try {
          const data = JSON.parse(msgEvent.data);

          if (data.vad_state) {
            setVadState(data.vad_state);
          }
        } catch (err) {
          console.error('Failed to parse VAD message:', err);
        }
      };
    };

    // --------------------------------------------------
    // ADD MICROPHONE TRACK
    // --------------------------------------------------

    streamRef.current.getTracks().forEach((track) => {
      pc.addTrack(track, streamRef.current);
    });

    // --------------------------------------------------
    // CONNECTION STATE
    // --------------------------------------------------

    pc.onconnectionstatechange = () => {
      const state = pc.connectionState;

      console.log('Connection state:', state);

      setStatus(`webrtc: ${state}`);

      if (state === 'failed') {
        setError(
          'Media connection failed. Peers may be on different networks, or a TURN relay is needed.'
        );

        setIsConnecting(false);
      } else if (state === 'disconnected') {
        setError('Connection lost. Try reconnecting.');

        setIsConnecting(false);
      } else if (state === 'connected') {
        setIsConnecting(false);
      }
    };

    // --------------------------------------------------
    // REMOTE AUDIO TRACK
    // --------------------------------------------------

    pc.ontrack = (event) => {
      console.log('Received remote track:', event.track);

      if (audioRef.current) {
        audioRef.current.srcObject = event.streams[0];

        audioRef.current.play().catch((err) => {
          console.error('Audio playback failed:', err);

          setError(
            'Audio autoplay was blocked by the browser. Click anywhere on the page and reconnect.'
          );
        });
      }

      setStatus('playing remote audio');
    };

    // --------------------------------------------------
    // WEBRTC HANDSHAKE
    // --------------------------------------------------

    try {
      const offer = await pc.createOffer();

      await pc.setLocalDescription(offer);

      console.log(
        'Initial ICE gathering state:',
        pc.iceGatheringState
      );

      // ------------------------------------------------
      // IMPORTANT:
      // Wait until ICE gathering is complete before
      // sending the SDP offer to the backend.
      // ------------------------------------------------

      await new Promise((resolve) => {
        if (pc.iceGatheringState === 'complete') {
          resolve();
          return;
        }

        const checkIceGathering = () => {
          console.log(
            'ICE gathering state:',
            pc.iceGatheringState
          );

          if (pc.iceGatheringState === 'complete') {
            pc.removeEventListener(
              'icegatheringstatechange',
              checkIceGathering
            );

            resolve();
          }
        };

        pc.addEventListener(
          'icegatheringstatechange',
          checkIceGathering
        );
      });

      console.log('Browser ICE gathering complete');

      console.log(
        'Generated SDP offer:',
        pc.localDescription
      );

      setStatus('sending offer to backend...');

      // ------------------------------------------------
      // SEND THE FINAL SDP AFTER ICE GATHERING
      // ------------------------------------------------

      const res = await fetch(BACKEND_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          sdp: pc.localDescription.sdp,
          type: pc.localDescription.type,
        }),
      });

      if (!res.ok) {
        if (res.status === 404) {
          throw new Error(
            'Signaling endpoint not found (404). Check the backend URL and route.'
          );
        } else if (res.status === 403) {
          throw new Error(
            'Request blocked (403). The backend may not allow this origin (CORS).'
          );
        }

        throw new Error(
          `Backend responded with ${res.status}`
        );
      }

      const answer = await res.json();

      console.log('Received SDP answer:', answer);

      await pc.setRemoteDescription(answer);

      console.log('Remote SDP description set successfully');

      setStatus('handshake complete — connected');
    } catch (err) {
      console.error('WebRTC handshake failed:', err);

      setIsConnecting(false);

      if (err.message === 'Failed to fetch') {
        setError(
          'Could not reach the backend. Is the server running and the URL correct?'
        );
      } else {
        setError(err.message);
      }

      setStatus('connection failed');
    }
  };

  // --------------------------------------------------
  // STOP / CLEANUP
  // --------------------------------------------------

  const stopMic = () => {
    streamRef.current
      ?.getTracks()
      .forEach((track) => track.stop());

    pcRef.current?.close();

    if (audioRef.current) {
      audioRef.current.srcObject = null;
    }

    streamRef.current = null;
    pcRef.current = null;
    dataChannelRef.current = null;

    setError(null);
    setIsConnecting(false);
    setVadState('idle');
    setStatus('idle');
  };

  // --------------------------------------------------
  // VAD UI
  // --------------------------------------------------

  const vadLabel = {
    idle: 'Idle',
    listening: '🎧 Listening',
    speaking: '🗣️ Speaking',
  };

  const vadColor = {
    idle: '#888',
    listening: '#4a9eff',
    speaking: '#ff9f43',
  };

  return (
    <div
      style={{
        padding: 40,
        fontFamily: 'sans-serif',
      }}
    >
      <h1>Auralis — Frontend</h1>

      <p>Status: {status}</p>

      <div
        style={{
          display: 'inline-block',
          padding: '8px 16px',
          marginBottom: 16,
          borderRadius: 20,
          border: `2px solid ${vadColor[vadState]}`,
          color: vadColor[vadState],
          fontWeight: 'bold',
        }}
      >
        {vadLabel[vadState] || vadState}
      </div>

      {error && (
        <div
          style={{
            padding: '12px 16px',
            marginBottom: 16,
            border: '1px solid #c33',
            borderRadius: 4,
            color: '#c33',
            maxWidth: 520,
          }}
        >
          {error}
        </div>
      )}

      <div>
        <button onClick={startMic}>
          Start Mic
        </button>

        <button
          onClick={connectWebRTC}
          disabled={isConnecting}
        >
          {isConnecting
            ? 'Connecting...'
            : 'Connect to Backend'}
        </button>

        <button onClick={stopMic}>
          Stop
        </button>
      </div>

      <audio
        ref={audioRef}
        autoPlay
        playsInline
      />
    </div>
  );
}

export default App;