# Auralis frontend (React + Vite)

This folder is the browser client of Auralis, the real-time voice-to-voice emotion engine.

The full documentation (features, DataChannel protocol, results, setup, deployment) is in the README at the root of the `frontend` branch: [../README.md](../README.md).

## Quick start

```bash
npm install
# .env  ->  VITE_BACKEND_URL=http://localhost:8001
npm run dev        # http://localhost:5173
npm run build      # production build in dist/
```

| Env variable | Purpose |
|---|---|
| `VITE_BACKEND_URL` | Auralis backend (local, or a Cloudflare Tunnel URL); no trailing `/` |
| `VITE_TURN_URL`, `VITE_TURN_USERNAME`, `VITE_TURN_CREDENTIAL` | Optional TURN relay for users on other networks |

Live demo: https://auralis-hazel.vercel.app
