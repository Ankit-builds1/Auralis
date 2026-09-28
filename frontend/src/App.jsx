import { useState, useRef, useEffect } from 'react';
import './App.css';

const BACKEND_URL = `${import.meta.env.VITE_BACKEND_URL}/webrtc/offer`;

const VAD_STATES = ['idle', 'listening', 'speaking', 'processing'];

function App() {
  const [status, setStatus] = useState('idle');
  const [error, setError] = useState(null);
  const [isConnecting, setIsConnecting] = useState(false);
  const [iceState, setIceState] = useState('new');
  const [vadState, setVadState] = useState('idle');
  const [transcripts, setTranscripts] = useState([]);
  const [events, setEvents] = useState([]);
  const [frameCount, setFrameCount] = useState(0);

  const streamRef = useRef(null);
  const pcRef = useRef(null);
  const audioRef = useRef(null);
  const dataChannelRef = useRef(null);
  const transcriptEndRef = useRef(null);

  // Auto-scroll transcript to the newest line
  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [transcripts]);

  // ---------------------------------------------------------
  // EVENT LOGGER
  // ---------------------------------------------------------

  const addEvent = (message) => {
    const time = new Date().toLocaleTimeString();
    setEvents((previous) =>
      [{ id: `${Date.now()}-${Math.random()}`, time, message }, ...previous].slice(0, 30)
    );
  };

  // ---------------------------------------------------------
  // TRANSCRIPT HISTORY
  // ---------------------------------------------------------

  const addTranscript = (text) => {
    const time = new Date().toLocaleTimeString();
    setTranscripts((previous) =>
      [...previous, { id: `${Date.now()}-${Math.random()}`, time, text }].slice(-50)
    );
  };

  // ---------------------------------------------------------
  // MICROPHONE
  // ---------------------------------------------------------

  const startMic = async () => {
    setError(null);

    // Guard: do not create a second mic stream while one is running.
    // A new stream would not be attached to the existing connection.
    if (streamRef.current) {
      addEvent('Microphone already active');
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 48000,
          sampleSize: 16,
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
        video: false,
      });

      streamRef.current = stream;
      const track = stream.getAudioTracks()[0];

      console.log('[MIC] Settings:', track.getSettings());
      addEvent(`Microphone started: ${track.label}`);
      setStatus('mic active');
    } catch (err) {
      console.error('[MIC] Microphone error:', err);

      if (err.name === 'NotAllowedError') {
        setError('Microphone permission denied. Allow microphone access in the browser.');
      } else if (err.name === 'NotFoundError') {
        setError('No microphone found. Connect a microphone and try again.');
      } else {
        setError(`Could not access microphone: ${err.message}`);
      }

      setStatus('mic unavailable');
    }
  };

  // ---------------------------------------------------------
  // DATACHANNEL MESSAGE HANDLER
  // Matches the backend's audio_track.py message types
  // ---------------------------------------------------------

  const handleDataChannelMessage = (rawMessage) => {
    let data;

    try {
      data = typeof rawMessage === 'string' ? JSON.parse(rawMessage) : rawMessage;
    } catch (err) {
      addEvent(`Raw message: ${rawMessage}`);
      return;
    }

    switch (data.type) {
      // ~50 per second: update the counter, but never log these
      case 'audio_frame':
        setFrameCount(data.frame ?? 0);
        return;

      case 'vad':
        if (VAD_STATES.includes(data.vad_state)) {
          setVadState(data.vad_state);
          addEvent(`VAD: ${data.vad_state}`);
          if (data.vad_state === 'processing') {
            setStatus('transcribing...');
          }
        }
        return;

      case 'transcript':
        if (typeof data.text === 'string' && data.text.trim()) {
          addTranscript(data.text.trim());
          addEvent(`Transcript: ${data.text.trim()}`);
          setStatus('transcript received');
        }
        return;

      case 'audio_error':
        console.error('[Backend] Audio error:', data);
        addEvent(`Backend audio error (frame ${data.frame}): ${data.error}`);
        return;

      default:
        if (typeof data.status === 'string') {
          setStatus(data.status);
          addEvent(`Backend status: ${data.status}`);
        } else {
          addEvent(`Unknown message: ${rawMessage}`);
        }
    }
  };

  // ---------------------------------------------------------
  // SETUP DATA CHANNEL
  // ---------------------------------------------------------

  const setupDataChannel = (channel, source = 'DataChannel') => {
    if (!channel) return;

    dataChannelRef.current = channel;

    channel.onopen = () => addEvent(`${source} opened: ${channel.label}`);
    channel.onmessage = (event) => handleDataChannelMessage(event.data);
    channel.onerror = (event) => {
      console.error(`${source} error:`, event);
      addEvent(`${source} error`);
    };
    channel.onclose = () => addEvent(`${source} closed`);
  };

  // ---------------------------------------------------------
  // WEBRTC CONNECTION
  // ---------------------------------------------------------

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

    if (
      pcRef.current &&
      (pcRef.current.connectionState === 'connected' ||
        pcRef.current.connectionState === 'connecting')
    ) {
      addEvent('WebRTC connection already active');
      return;
    }

    setIsConnecting(true);
    setVadState('idle');
    setFrameCount(0);

    const pc = new RTCPeerConnection({
      iceServers: [
        { urls: 'stun:stun.l.google.com:19302' },
        {
          urls: 'turn:openrelay.metered.ca:80',
          username: 'openrelayproject',
          credential: 'openrelayproject',
        },
      ],
    });

    pcRef.current = pc;
    addEvent('PeerConnection created');

    pc.oniceconnectionstatechange = () => {
      const state = pc.iceConnectionState;
      setIceState(state);
      addEvent(`ICE: ${state}`);
      if (state === 'failed') {
        setError('ICE connection failed. The backend or TURN connection may need attention.');
      }
    };

    const dataChannel = pc.createDataChannel('vad-state');
    setupDataChannel(dataChannel, 'Frontend DataChannel');

    pc.ondatachannel = (event) => setupDataChannel(event.channel, 'Backend DataChannel');

    const audioTracks = streamRef.current.getAudioTracks();

    if (audioTracks.length === 0) {
      setError('No audio track found in microphone stream.');
      setIsConnecting(false);
      pc.close();
      return;
    }

    audioTracks.forEach((track) => pc.addTrack(track, streamRef.current));
    addEvent('Microphone audio track added');

    pc.onconnectionstatechange = () => {
      const state = pc.connectionState;
      setStatus(`webrtc: ${state}`);
      addEvent(`Connection: ${state}`);

      if (state === 'connected') {
        setIsConnecting(false);
        addEvent('WebRTC connection established');
      }

      if (state === 'failed') {
        setIsConnecting(false);
        setError('Media connection failed. Check the backend WebRTC connection and TURN configuration.');
      }

      if (state === 'disconnected' || state === 'closed') {
        setIsConnecting(false);
      }
    };

    // Backend no longer echoes mic audio. This stays for Week 3,
    // when the backend will stream TTS audio back.
    pc.ontrack = (event) => {
      addEvent('Remote audio track received');

      if (audioRef.current && event.streams && event.streams[0]) {
        audioRef.current.srcObject = event.streams[0];
        audioRef.current.play().catch((err) => {
          console.error('[Audio] Playback failed:', err);
          setError('Browser blocked remote audio playback. Click the page and reconnect.');
        });
      }
    };

    try {
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      addEvent('SDP offer created');

      // Wait for ICE gathering (max 10s)
      await new Promise((resolve) => {
        if (pc.iceGatheringState === 'complete') {
          resolve();
          return;
        }

        let settled = false;

        const finish = () => {
          if (settled) return;
          settled = true;
          clearTimeout(timeout);
          pc.removeEventListener('icegatheringstatechange', checkIceGathering);
          resolve();
        };

        const timeout = setTimeout(() => {
          addEvent('ICE gathering timeout — continuing');
          finish();
        }, 10000);

        const checkIceGathering = () => {
          if (pc.iceGatheringState === 'complete') finish();
        };

        pc.addEventListener('icegatheringstatechange', checkIceGathering);
      });

      addEvent('Browser ICE gathering complete');

      if (!pc.localDescription) {
        throw new Error('Local SDP description was not created.');
      }

      setStatus('sending offer to backend...');

      const response = await fetch(BACKEND_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sdp: pc.localDescription.sdp,
          type: pc.localDescription.type,
        }),
      });

      if (!response.ok) {
        if (response.status === 404) {
          throw new Error('Signaling endpoint not found (404). Check the backend URL and /webrtc/offer route.');
        }
        if (response.status === 403) {
          throw new Error('Request blocked (403). The backend may not allow this origin.');
        }
        throw new Error(`Backend responded with ${response.status}`);
      }

      const answer = await response.json();
      addEvent('SDP answer received');

      if (!answer || !answer.sdp || !answer.type) {
        throw new Error('Backend returned an invalid SDP answer.');
      }

      await pc.setRemoteDescription(answer);
      addEvent('Remote SDP applied');
      setStatus('handshake complete — connecting');
    } catch (err) {
      console.error('[WebRTC] Handshake failed:', err);
      setIsConnecting(false);

      if (err.message === 'Failed to fetch') {
        setError('Could not reach the backend. Check that the backend and ngrok are running.');
      } else {
        setError(err.message || 'WebRTC handshake failed.');
      }

      setStatus('connection failed');
    }
  };

  // ---------------------------------------------------------
  // STOP EVERYTHING
  // ---------------------------------------------------------

  const stopMic = () => {
    if (dataChannelRef.current) {
      try { dataChannelRef.current.close(); } catch (err) { console.warn(err); }
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
    }

    if (pcRef.current) {
      try { pcRef.current.close(); } catch (err) { console.warn(err); }
    }

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
    setIceState('new');
    setFrameCount(0);
    setEvents([]);
    addEvent('Connection stopped');
    // Transcript history is kept on purpose so you can review it after stopping
  };

  const clearTranscripts = () => setTranscripts([]);

  // ---------------------------------------------------------
  // VAD DISPLAY
  // ---------------------------------------------------------

  const getVadIcon = () =>
    ({ speaking: '🗣️', listening: '🎧', processing: '⚙️' }[vadState] || '◉');

  const getVadLabel = () =>
    ({ speaking: 'SPEAKING', listening: 'LISTENING', processing: 'PROCESSING' }[vadState] || 'IDLE');

  // ---------------------------------------------------------
  // UI
  // ---------------------------------------------------------

  return (
    <div className="auralis-app">
      <div className="auralis-container">

        <header className="auralis-header">
          <div className="brand">
            <div className="brand-logo">A</div>
            <div>
              <h1>AURALIS</h1>
              <p>REAL-TIME VOICE INTELLIGENCE ENGINE</p>
            </div>
          </div>
          <div className="system-status">
            <span className="status-dot"></span>
            SYSTEM ONLINE
          </div>
        </header>

        <div className="dashboard-grid">

          <section className="card">
            <div className="card-title">
              <h2>Connection</h2>
              <span>WEBRTC</span>
            </div>

            <div className="connection-grid">
              <div className="metric">
                <div className="metric-label">Status</div>
                <div className="metric-value">{status}</div>
              </div>
              <div className="metric">
                <div className="metric-label">ICE State</div>
                <div className="metric-value">{iceState}</div>
              </div>
              <div className="metric">
                <div className="metric-label">Frames</div>
                <div className="metric-value">{frameCount}</div>
              </div>
            </div>

            <div className="controls">
              <button className="btn" onClick={startMic}>🎙 START MIC</button>
              <button className="btn btn-primary" onClick={connectWebRTC} disabled={isConnecting}>
                {isConnecting ? 'CONNECTING...' : 'CONNECT'}
              </button>
              <button className="btn btn-danger" onClick={stopMic}>STOP</button>
            </div>

            {error && <div className="error-box">⚠ {error}</div>}
          </section>

          <section className="card">
            <div className="card-title">
              <h2>Voice Activity</h2>
              <span>VAD ENGINE</span>
            </div>
            <div className="vad-box">
              <div className="vad-state">
                <div className="vad-icon">{getVadIcon()}</div>
                <div className="vad-name">{getVadLabel()}</div>
              </div>
            </div>
          </section>

          <section className="card full">
            <div className="card-title">
              <h2>Live Transcript</h2>
              <span>
                STT OUTPUT
                {transcripts.length > 0 && (
                  <button className="btn-link" onClick={clearTranscripts}> · CLEAR</button>
                )}
              </span>
            </div>

            <div className="transcript-box">
              {transcripts.length === 0 ? (
                <span className="transcript-placeholder">Waiting for speech input...</span>
              ) : (
                transcripts.map((line) => (
                  <div className="transcript-line" key={line.id}>
                    <span className="transcript-time">{line.time}</span>
                    <span>{line.text}</span>
                  </div>
                ))
              )}
              <div ref={transcriptEndRef} />
            </div>
          </section>

          <section className="card full">
            <div className="card-title">
              <h2>System Events</h2>
              <span>LIVE LOG</span>
            </div>
            <div className="events">
              {events.length === 0 ? (
                <div className="transcript-placeholder">No events yet.</div>
              ) : (
                events.map((event) => (
                  <div className="event" key={event.id}>
                    <span className="event-time">{event.time}</span>
                    <span className="event-message">{event.message}</span>
                  </div>
                ))
              )}
            </div>
          </section>

        </div>

        <div className="footer">AURALIS // REAL-TIME MULTI-MODEL VOICE SYSTEM</div>

        <audio ref={audioRef} autoPlay playsInline />
      </div>
    </div>
  );
}

export default App;