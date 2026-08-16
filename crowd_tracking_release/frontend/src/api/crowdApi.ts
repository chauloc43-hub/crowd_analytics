const API_BASE = '/api/v1';

export async function createSession(mode = 'default', cameraId?: string) {
  try {
    const res = await fetch(`${API_BASE}/sessions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode, camera_id: cameraId }),
    });
    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    return await res.json();
  } catch (err) {
    console.warn('API connection offline, using simulated session', err);
    return {
      status: 'created',
      session: {
        session_id: `demo-${Date.now()}`,
        mode: mode,
        created_at: new Date().toISOString(),
      },
    };
  }
}

export async function getSessionStats(sessionId: string) {
  if (!sessionId || sessionId.startsWith('demo-')) {
    return null;
  }
  try {
    const res = await fetch(`${API_BASE}/sessions/${sessionId}/stats`);
    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    return await res.json();
  } catch (err) {
    // Fallback simulated telemetry data if API is not active
    return null;
  }
}

export async function submitFrame(sessionId: string, imageBlob: Blob) {
  if (!sessionId || sessionId.startsWith('demo-')) {
    return { status: 'simulated' };
  }
  const formData = new FormData();
  formData.append('file', imageBlob, 'frame.jpg');

  const res = await fetch(`${API_BASE}/sessions/${sessionId}/frame`, {
    method: 'POST',
    body: formData,
  });
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
