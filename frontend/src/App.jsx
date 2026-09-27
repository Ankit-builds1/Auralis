import { useState, useRef } from 'react';
import './App.css';

const BACKEND_URL =
  `${import.meta.env.VITE_BACKEND_URL}/webrtc/offer`;

function App() {
  const [status, setStatus] = useState('idle');
  const [error, setError] = useState(null);
  const [isConnecting, setIsConnecting] = useState(false);
  const [iceState, setIceState] = useState('new');
  const [vadState, setVadState] = useState('idle');
  const [transcript, setTranscript] = useState('');
  const [events, setEvents] = useState([]);

  const streamRef = useRef(null);
  const pcRef = useRef(null);
  const audioRef = useRef(null);
  const dataChannelRef = useRef(null);

  // ---------------------------------------------------------
  // EVENT LOGGER
  // ---------------------------------------------------------

  const addEvent = (message) => {
    const time = new Date().toLocaleTimeString();

    setEvents((previous) => [
      {
        id: `${Date.now()}-${Math.random()}`,
        time,
        message,
      },
      ...previous,
    ].slice(0, 30));
  };

  // ---------------------------------------------------------
  // MICROPHONE
  // ---------------------------------------------------------

  const startMic = async () => {
    setError(null);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 48000,
          sampleSize: 16,

          // Disable browser-side processing temporarily.
          // This lets us debug the raw microphone signal.
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
        video: false,
      });

      streamRef.current = stream;

      const track = stream.getAudioTracks()[0];

      console.log('[MIC] Track:', track);

      console.log(
        '[MIC] Settings:',
        track.getSettings()
      );

      console.log(
        '[MIC] Constraints:',
        track.getConstraints()
      );

      console.log(
        '[MIC] Ready state:',
        track.readyState
      );

      console.log(
        '[MIC] Enabled:',
        track.enabled
      );

      console.log(
        '[MIC] Muted:',
        track.muted
      );

      addEvent(
        `Microphone started: ${track.label}`
      );

      addEvent(
        `Mic settings: ${JSON.stringify(track.getSettings())}`
      );

      setStatus('mic active');

    } catch (err) {
      console.error(
        '[MIC] Microphone error:',
        err
      );

      if (err.name === 'NotAllowedError') {
        setError(
          'Microphone permission denied. Allow microphone access in the browser.'
        );
      } else if (err.name === 'NotFoundError') {
        setError(
          'No microphone found. Connect a microphone and try again.'
        );
      } else {
        setError(
          `Could not access microphone: ${err.message}`
        );
      }

      setStatus('mic unavailable');
    }
  };

  // ---------------------------------------------------------
  // TRANSCRIPT / DATACHANNEL MESSAGE HANDLER
  // ---------------------------------------------------------

  const handleDataChannelMessage = (rawMessage) => {
    console.log(
      '[DataChannel] Message received:',
      rawMessage
    );

    addEvent(
      `DataChannel: ${rawMessage}`
    );

    try {
      const data =
        typeof rawMessage === 'string'
          ? JSON.parse(rawMessage)
          : rawMessage;

      // -------------------------------------------------------
      // VAD STATE
      // -------------------------------------------------------

      if (
        data.vad_state === 'idle' ||
        data.vad_state === 'listening' ||
        data.vad_state === 'speaking' ||
        data.vad_state === 'processing'
      ) {
        setVadState(data.vad_state);

        addEvent(
          `VAD: ${data.vad_state}`
        );

        if (data.vad_state === 'processing') {
          setStatus('processing');
        }
      }

      // -------------------------------------------------------
      // TRANSCRIPT FORMAT 1
      //
      // {
      //   "type": "transcript",
      //   "text": "hello"
      // }
      // -------------------------------------------------------

      if (
        data.type === 'transcript' &&
        typeof data.text === 'string'
      ) {
        setTranscript(data.text);

        addEvent(
          `Transcript: ${data.text}`
        );

        setStatus('transcript received');

        return;
      }

      // -------------------------------------------------------
      // TRANSCRIPT FORMAT 2
      //
      // {
      //   "transcript": "hello"
      // }
      // -------------------------------------------------------

      if (
        typeof data.transcript === 'string'
      ) {
        setTranscript(data.transcript);

        addEvent(
          `Transcript: ${data.transcript}`
        );

        setStatus('transcript received');

        return;
      }

      // -------------------------------------------------------
      // BACKEND STATUS
      // -------------------------------------------------------

      if (
        typeof data.status === 'string'
      ) {
        setStatus(data.status);

        addEvent(
          `Backend status: ${data.status}`
        );
      }

    } catch (err) {
      console.error(
        'DataChannel message is not valid JSON:',
        err
      );

      addEvent(
        `Raw message: ${rawMessage}`
      );
    }
  };

  // ---------------------------------------------------------
  // SETUP DATA CHANNEL
  // ---------------------------------------------------------

  const setupDataChannel = (
    channel,
    source = 'DataChannel'
  ) => {
    if (!channel) {
      return;
    }

    console.log(
      `${source} received:`,
      channel.label
    );

    dataChannelRef.current = channel;

    channel.onopen = () => {
      console.log(
        `${source} opened:`,
        channel.label
      );

      addEvent(
        `${source} opened: ${channel.label}`
      );
    };

    channel.onmessage = (event) => {
      handleDataChannelMessage(
        event.data
      );
    };

    channel.onerror = (event) => {
      console.error(
        `${source} error:`,
        event
      );

      addEvent(
        `${source} error`
      );
    };

    channel.onclose = () => {
      console.log(
        `${source} closed`
      );

      addEvent(
        `${source} closed`
      );
    };
  };

  // ---------------------------------------------------------
  // WEBRTC CONNECTION
  // ---------------------------------------------------------

  const connectWebRTC = async () => {
    setError(null);

    // -------------------------------------------------------
    // MAKE SURE MICROPHONE EXISTS
    // -------------------------------------------------------

    if (!streamRef.current) {
      setError(
        'Start the microphone before connecting.'
      );
      return;
    }

    // -------------------------------------------------------
    // MAKE SURE BACKEND URL EXISTS
    // -------------------------------------------------------

    if (!import.meta.env.VITE_BACKEND_URL) {
      setError(
        'Backend URL is not configured. Check your .env file.'
      );
      return;
    }

    // -------------------------------------------------------
    // PREVENT DUPLICATE CONNECTIONS
    // -------------------------------------------------------

    if (
      pcRef.current &&
      (
        pcRef.current.connectionState === 'connected' ||
        pcRef.current.connectionState === 'connecting'
      )
    ) {
      addEvent(
        'WebRTC connection already active'
      );

      return;
    }

    setIsConnecting(true);
    setError(null);
    setTranscript('');
    setVadState('idle');

    // -------------------------------------------------------
    // CREATE PEER CONNECTION
    // -------------------------------------------------------

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

    console.log(
      '[WebRTC] PeerConnection created'
    );

    addEvent(
      'PeerConnection created'
    );

    // -------------------------------------------------------
    // ICE CONNECTION STATE
    // -------------------------------------------------------

    pc.oniceconnectionstatechange = () => {
      const state =
        pc.iceConnectionState;

      console.log(
        '[WebRTC] ICE connection state:',
        state
      );

      setIceState(state);

      addEvent(
        `ICE: ${state}`
      );

      if (state === 'failed') {
        setError(
          'ICE connection failed. The backend or TURN connection may need attention.'
        );
      }
    };

    // -------------------------------------------------------
    // ICE GATHERING STATE
    // -------------------------------------------------------

    pc.onicegatheringstatechange = () => {
      console.log(
        '[WebRTC] ICE gathering state:',
        pc.iceGatheringState
      );

      addEvent(
        `ICE gathering: ${pc.iceGatheringState}`
      );
    };

    // -------------------------------------------------------
    // FRONTEND DATA CHANNEL
    // -------------------------------------------------------

    const dataChannel =
      pc.createDataChannel('vad-state');

    setupDataChannel(
      dataChannel,
      'Frontend DataChannel'
    );

    // -------------------------------------------------------
    // BACKEND DATA CHANNEL
    // -------------------------------------------------------

    pc.ondatachannel = (event) => {
      console.log(
        '[WebRTC] Backend opened DataChannel:',
        event.channel.label
      );

      setupDataChannel(
        event.channel,
        'Backend DataChannel'
      );
    };

    // -------------------------------------------------------
    // GET MICROPHONE TRACKS
    // -------------------------------------------------------

    const audioTracks =
      streamRef.current.getAudioTracks();

    if (audioTracks.length === 0) {
      setError(
        'No audio track found in microphone stream.'
      );

      setIsConnecting(false);

      pc.close();

      return;
    }

    // -------------------------------------------------------
    // MICROPHONE DEBUG INFORMATION
    // -------------------------------------------------------

    const microphoneTrack =
      audioTracks[0];

    console.log(
      '[WebRTC] Microphone track settings:',
      microphoneTrack.getSettings()
    );

    console.log(
      '[WebRTC] Microphone track constraints:',
      microphoneTrack.getConstraints()
    );

    console.log(
      '[WebRTC] Microphone track enabled:',
      microphoneTrack.enabled
    );

    console.log(
      '[WebRTC] Microphone track muted:',
      microphoneTrack.muted
    );

    console.log(
      '[WebRTC] Microphone track readyState:',
      microphoneTrack.readyState
    );

    addEvent(
      `Mic track ready: ${microphoneTrack.readyState}`
    );

    // -------------------------------------------------------
    // ADD MICROPHONE TRACK
    // -------------------------------------------------------

    audioTracks.forEach((track) => {
      console.log(
        '[WebRTC] Adding microphone track:',
        track
      );

      pc.addTrack(
        track,
        streamRef.current
      );
    });

    addEvent(
      'Microphone audio track added'
    );

    // -------------------------------------------------------
    // CONNECTION STATE
    // -------------------------------------------------------

    pc.onconnectionstatechange = () => {
      const state =
        pc.connectionState;

      console.log(
        '[WebRTC] Connection state:',
        state
      );

      setStatus(
        `webrtc: ${state}`
      );

      addEvent(
        `Connection: ${state}`
      );

      if (state === 'connected') {
        setIsConnecting(false);

        addEvent(
          'WebRTC connection established'
        );
      }

      if (state === 'failed') {
        setIsConnecting(false);

        setError(
          'Media connection failed. Check the backend WebRTC connection and TURN configuration.'
        );
      }

      if (state === 'disconnected') {
        setIsConnecting(false);

        addEvent(
          'WebRTC connection disconnected'
        );
      }

      if (state === 'closed') {
        setIsConnecting(false);

        addEvent(
          'WebRTC connection closed'
        );
      }
    };

    // -------------------------------------------------------
    // REMOTE AUDIO
    // -------------------------------------------------------

    pc.ontrack = (event) => {
      console.log(
        '[WebRTC] Received remote track:',
        event.track
      );

      addEvent(
        'Remote audio track received'
      );

      if (
        audioRef.current &&
        event.streams &&
        event.streams[0]
      ) {
        audioRef.current.srcObject =
          event.streams[0];

        audioRef.current
          .play()
          .then(() => {
            console.log(
              '[Audio] Remote audio playback started'
            );
          })
          .catch((err) => {
            console.error(
              '[Audio] Playback failed:',
              err
            );

            setError(
              'Browser blocked remote audio playback. Click the page and reconnect.'
            );
          });
      }

      setStatus(
        'playing remote audio'
      );
    };

    // -------------------------------------------------------
    // CREATE OFFER
    // -------------------------------------------------------

    try {
      const offer =
        await pc.createOffer();

      await pc.setLocalDescription(
        offer
      );

      console.log(
        '[WebRTC] Initial ICE gathering state:',
        pc.iceGatheringState
      );

      addEvent(
        'SDP offer created'
      );

      // -----------------------------------------------------
      // WAIT FOR ICE GATHERING
      // -----------------------------------------------------

      await new Promise((resolve) => {
        if (
          pc.iceGatheringState ===
          'complete'
        ) {
          resolve();
          return;
        }

        let settled = false;

        const finish = () => {
          if (settled) {
            return;
          }

          settled = true;

          clearTimeout(
            timeout
          );

          pc.removeEventListener(
            'icegatheringstatechange',
            checkIceGathering
          );

          resolve();
        };

        const timeout =
          setTimeout(() => {
            console.warn(
              '[WebRTC] ICE gathering timeout. Continuing with gathered candidates.'
            );

            addEvent(
              'ICE gathering timeout — continuing'
            );

            finish();
          }, 10000);

        const checkIceGathering = () => {
          console.log(
            '[WebRTC] ICE gathering state:',
            pc.iceGatheringState
          );

          if (
            pc.iceGatheringState ===
            'complete'
          ) {
            finish();
          }
        };

        pc.addEventListener(
          'icegatheringstatechange',
          checkIceGathering
        );
      });

      console.log(
        '[WebRTC] Browser ICE gathering complete'
      );

      addEvent(
        'Browser ICE gathering complete'
      );

      // -----------------------------------------------------
      // VERIFY LOCAL DESCRIPTION
      // -----------------------------------------------------

      if (!pc.localDescription) {
        throw new Error(
          'Local SDP description was not created.'
        );
      }

      console.log(
        '[WebRTC] Generated SDP offer:',
        pc.localDescription
      );

      setStatus(
        'sending offer to backend...'
      );

      // -----------------------------------------------------
      // SEND OFFER TO BACKEND
      // -----------------------------------------------------

      const response =
        await fetch(
          BACKEND_URL,
          {
            method: 'POST',

            headers: {
              'Content-Type':
                'application/json',
            },

            body: JSON.stringify({
              sdp:
                pc.localDescription.sdp,

              type:
                pc.localDescription.type,
            }),
          }
        );

      if (!response.ok) {
        if (response.status === 404) {
          throw new Error(
            'Signaling endpoint not found (404). Check the backend URL and /webrtc/offer route.'
          );
        }

        if (response.status === 403) {
          throw new Error(
            'Request blocked (403). The backend may not allow this origin.'
          );
        }

        throw new Error(
          `Backend responded with ${response.status}`
        );
      }

      // -----------------------------------------------------
      // RECEIVE SDP ANSWER
      // -----------------------------------------------------

      const answer =
        await response.json();

      console.log(
        '[WebRTC] Received SDP answer:',
        answer
      );

      addEvent(
        'SDP answer received'
      );

      if (
        !answer ||
        !answer.sdp ||
        !answer.type
      ) {
        throw new Error(
          'Backend returned an invalid SDP answer.'
        );
      }

      // -----------------------------------------------------
      // APPLY REMOTE DESCRIPTION
      // -----------------------------------------------------

      await pc.setRemoteDescription(
        answer
      );

      console.log(
        '[WebRTC] Remote SDP description set successfully'
      );

      addEvent(
        'Remote SDP applied'
      );

      setStatus(
        'handshake complete — connecting'
      );

    } catch (err) {
      console.error(
        '[WebRTC] Handshake failed:',
        err
      );

      setIsConnecting(false);

      if (
        err.message ===
        'Failed to fetch'
      ) {
        setError(
          'Could not reach the backend. Check that the backend and ngrok are running.'
        );
      } else {
        setError(
          err.message ||
          'WebRTC handshake failed.'
        );
      }

      setStatus(
        'connection failed'
      );
    }
  };

  // ---------------------------------------------------------
  // STOP EVERYTHING
  // ---------------------------------------------------------

  const stopMic = () => {
    console.log(
      '[WebRTC] Stopping connection'
    );

    // -------------------------------------------------------
    // CLOSE DATACHANNEL
    // -------------------------------------------------------

    if (
      dataChannelRef.current
    ) {
      try {
        dataChannelRef.current.close();
      } catch (err) {
        console.warn(
          'DataChannel close error:',
          err
        );
      }
    }

    // -------------------------------------------------------
    // STOP MICROPHONE TRACKS
    // -------------------------------------------------------

    if (streamRef.current) {
      streamRef.current
        .getTracks()
        .forEach((track) => {
          track.stop();
        });
    }

    // -------------------------------------------------------
    // CLOSE PEER CONNECTION
    // -------------------------------------------------------

    if (pcRef.current) {
      try {
        pcRef.current.close();
      } catch (err) {
        console.warn(
          'PeerConnection close error:',
          err
        );
      }
    }

    // -------------------------------------------------------
    // CLEAR REMOTE AUDIO
    // -------------------------------------------------------

    if (audioRef.current) {
      audioRef.current.srcObject = null;
    }

    // -------------------------------------------------------
    // RESET REFERENCES
    // -------------------------------------------------------

    streamRef.current = null;
    pcRef.current = null;
    dataChannelRef.current = null;

    // -------------------------------------------------------
    // RESET UI
    // -------------------------------------------------------

    setError(null);
    setIsConnecting(false);
    setVadState('idle');
    setStatus('idle');
    setIceState('new');
    setTranscript('');
    setEvents([]);

    addEvent(
      'Connection stopped'
    );
  };

  // ---------------------------------------------------------
  // VAD ICON
  // ---------------------------------------------------------

  const getVadIcon = () => {
    switch (vadState) {
      case 'speaking':
        return '🗣️';

      case 'listening':
        return '🎧';

      case 'processing':
        return '⚙️';

      default:
        return '◉';
    }
  };

  // ---------------------------------------------------------
  // VAD LABEL
  // ---------------------------------------------------------

  const getVadLabel = () => {
    switch (vadState) {
      case 'speaking':
        return 'SPEAKING';

      case 'listening':
        return 'LISTENING';

      case 'processing':
        return 'PROCESSING';

      default:
        return 'IDLE';
    }
  };

  // ---------------------------------------------------------
  // UI
  // ---------------------------------------------------------

  return (
    <div className="auralis-app">

      <div className="auralis-container">

        {/* HEADER */}

        <header className="auralis-header">

          <div className="brand">

            <div className="brand-logo">
              A
            </div>

            <div>
              <h1>
                AURALIS
              </h1>

              <p>
                REAL-TIME VOICE INTELLIGENCE ENGINE
              </p>
            </div>

          </div>

          <div className="system-status">

            <span className="status-dot"></span>

            SYSTEM ONLINE

          </div>

        </header>

        {/* DASHBOARD */}

        <div className="dashboard-grid">

          {/* CONNECTION CARD */}

          <section className="card">

            <div className="card-title">

              <h2>
                Connection
              </h2>

              <span>
                WEBRTC
              </span>

            </div>

            <div className="connection-grid">

              <div className="metric">

                <div className="metric-label">
                  Status
                </div>

                <div className="metric-value">
                  {status}
                </div>

              </div>

              <div className="metric">

                <div className="metric-label">
                  ICE State
                </div>

                <div className="metric-value">
                  {iceState}
                </div>

              </div>

            </div>

            {/* CONTROLS */}

            <div className="controls">

              <button
                className="btn"
                onClick={startMic}
              >
                🎙 START MIC
              </button>

              <button
                className="btn btn-primary"
                onClick={connectWebRTC}
                disabled={isConnecting}
              >
                {isConnecting
                  ? 'CONNECTING...'
                  : 'CONNECT'}
              </button>

              <button
                className="btn btn-danger"
                onClick={stopMic}
              >
                STOP
              </button>

            </div>

            {/* ERROR */}

            {error && (
              <div className="error-box">
                ⚠ {error}
              </div>
            )}

          </section>

          {/* VAD CARD */}

          <section className="card">

            <div className="card-title">

              <h2>
                Voice Activity
              </h2>

              <span>
                VAD ENGINE
              </span>

            </div>

            <div className="vad-box">

              <div className="vad-state">

                <div className="vad-icon">
                  {getVadIcon()}
                </div>

                <div className="vad-name">
                  {getVadLabel()}
                </div>

              </div>

            </div>

          </section>

          {/* TRANSCRIPT */}

          <section className="card full">

            <div className="card-title">

              <h2>
                Live Transcript
              </h2>

              <span>
                STT OUTPUT
              </span>

            </div>

            <div className="transcript-box">

              {transcript ? (
                transcript
              ) : (
                <span className="transcript-placeholder">
                  Waiting for speech input...
                </span>
              )}

            </div>

          </section>

          {/* SYSTEM EVENTS */}

          <section className="card full">

            <div className="card-title">

              <h2>
                System Events
              </h2>

              <span>
                LIVE LOG
              </span>

            </div>

            <div className="events">

              {events.length === 0 ? (

                <div className="transcript-placeholder">
                  No events yet.
                </div>

              ) : (

                events.map((event) => (

                  <div
                    className="event"
                    key={event.id}
                  >

                    <span className="event-time">
                      {event.time}
                    </span>

                    <span className="event-message">
                      {event.message}
                    </span>

                  </div>

                ))

              )}

            </div>

          </section>

        </div>

        {/* FOOTER */}

        <div className="footer">
          AURALIS // REAL-TIME MULTI-MODEL VOICE SYSTEM
        </div>

        {/* REMOTE AUDIO */}

        <audio
          ref={audioRef}
          autoPlay
          playsInline
        />

      </div>

    </div>
  );
}

export default App;