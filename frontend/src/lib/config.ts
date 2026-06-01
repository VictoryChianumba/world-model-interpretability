// Shared runtime endpoints for the v2 frontend. The backend is reachable on the same
// host as the page, port 8000 (matches the v1 hooks' inlined constants).

export const API_BASE =
  typeof window !== "undefined"
    ? `http://${window.location.hostname}:8000`
    : "http://localhost:8000";

export const WS_BASE =
  typeof window !== "undefined"
    ? `ws://${window.location.hostname}:8000`
    : "ws://localhost:8000";
