const getBaseUrl = () => {
  if (import.meta.env.VITE_API_BASE_URL) {
    return import.meta.env.VITE_API_BASE_URL.replace(/\/$/, '');
  }
  if (typeof window !== 'undefined') {
    const host = window.location.hostname;
    if (host === 'localhost' || host === '127.0.0.1' || host.startsWith('192.168.') || host.startsWith('10.')) {
      return `${window.location.protocol}//${host}:8000`;
    }
  }
  return 'https://chauloc43-hub--crowd-analytics-v2-web.modal.run';
};

const BASE_URL = getBaseUrl();
const API_BASE = `${BASE_URL}/api/v1`;

export async function createSession(mode = 'default', cameraId?: string) {
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const res = await fetch(`${API_BASE}/sessions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode, camera_id: cameraId }),
      });
      if (res.ok) {
        return await res.json();
      }
    } catch (err) {
      console.warn(`[createSession] Attempt ${attempt + 1} failed, retrying...`, err);
      await new Promise((r) => setTimeout(r, 1000));
    }
  }
  return {
    status: 'created',
    session: {
      session_id: `demo-${Date.now()}`,
      mode: mode,
      created_at: new Date().toISOString(),
    },
  };
}

export async function getSessionStats(sessionId: string) {
  if (!sessionId || sessionId.startsWith('demo-')) {
    return null;
  }
  try {
    const res = await fetch(`${API_BASE}/sessions/${sessionId}/stats`);
    if (res.status === 404) {
      return { _session_expired: true };
    }
    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    return await res.json();
  } catch (err) {
    // Fallback simulated telemetry data if API is not active
    return null;
  }
}

export async function submitFrame(
  sessionId: string,
  imageBlob: Blob,
  overlays?: { boxes?: boolean; ids?: boolean; attributes?: boolean; motion?: boolean; zones?: boolean; seats?: boolean }
) {
  if (!sessionId || sessionId.startsWith('demo-')) {
    return { status: 'simulated' };
  }
  const formData = new FormData();
  formData.append('file', imageBlob, 'frame.jpg');

  const params = overlays
    ? `?boxes=${overlays.boxes ?? true}&ids=${overlays.ids ?? true}&attributes=${overlays.attributes ?? true}&motion=${overlays.motion ?? false}&zones=${overlays.zones ?? true}&seats=${overlays.seats ?? true}`
    : '';

  const res = await fetch(`${API_BASE}/sessions/${sessionId}/frame${params}`, {
    method: 'POST',
    body: formData,
  });
  if (res.status === 404) {
    return { _session_expired: true };
  }
  if (!res.ok) throw new Error(`HTTP error ${res.status}`);
  return await res.json();
}

export async function resetSession(sessionId: string) {
  if (!sessionId || sessionId.startsWith('demo-')) {
    return { status: 'reset' };
  }
  const res = await fetch(`${API_BASE}/sessions/${sessionId}/reset`, {
    method: 'POST',
  });
  if (!res.ok) throw new Error(`HTTP error ${res.status}`);
  return await res.json();
}

export async function analyzeVideo(file: File, mode = 'default') {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('mode', mode);

  const res = await fetch(`${API_BASE}/video/analyze`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) throw new Error(`HTTP error ${res.status}`);
  return await res.json();
}
