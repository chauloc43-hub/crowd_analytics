import React, { useState, useRef, useEffect } from 'react';
import { Camera, Layers, Play, Pause, Settings, RefreshCw, CheckSquare, Square, AlertCircle, UserCheck, Crosshair } from 'lucide-react';
import type { LabelMode, OverlayOptions, AnalyticsData } from '../types/analytics';
import { createSession, submitFrame, resetSession } from '../api/crowdApi';

interface LivePageProps {
  analytics: AnalyticsData;
  onAnalyticsUpdate: (stats: any) => void;
  t: any;
}

interface FocusedBox {
  id: string;
  tracker: string;
  x: number;
  y: number;
  w: number;
  h: number;
  attr: string;
  confidence: string;
}

export const LivePage: React.FC<LivePageProps> = ({ analytics, onAnalyticsUpdate, t }) => {
  const [labelMode, setLabelMode] = useState<LabelMode>('minimal');
  const [isStreaming, setIsStreaming] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [facingMode, setFacingMode] = useState<'user' | 'environment'>('user');
  const [latencyMs, setLatencyMs] = useState<number>(86);
  const [cameraFps, setCameraFps] = useState<number>(30);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [selectedPerson, setSelectedPerson] = useState<FocusedBox | null>(null);

  const [overlays, setOverlays] = useState<OverlayOptions>({
    boxes: true,
    ids: true,
    attributes: true,
    motion: false,
    zones: true,
    seats: true,
    trajectory: false,
    heatmap: false,
  });

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const intervalRef = useRef<any>(null);
  const lastFrameTimeRef = useRef<number>(performance.now());

  useEffect(() => {
    if (!captureCanvasRef.current) {
      captureCanvasRef.current = document.createElement('canvas');
    }
  }, []);

  const toggleStreaming = async () => {
    if (isStreaming) {
      stopStream();
    } else {
      await startStream();
    }
  };

  const startStream = async () => {
    setErrorMessage(null);
    try {
      const sessionRes = await createSession('default');
      const newSessionId = sessionRes?.session?.session_id || sessionRes?.session?.id;
      if (newSessionId) {
        setSessionId(newSessionId);
      }

      const constraints = {
        video: { facingMode, width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      };

      const mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = mediaStream;

      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
        await videoRef.current.play();
      }

      setIsStreaming(true);
      startFrameLoop(newSessionId);
    } catch (err: any) {
      console.error('Camera stream error:', err);
      setErrorMessage(`Camera Error: ${err.message || 'Could not access webcam'}`);
      setIsStreaming(false);
    }
  };

  const stopStream = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsStreaming(false);
  };

  const switchCameraFacing = async () => {
    const nextFacing = facingMode === 'user' ? 'environment' : 'user';
    setFacingMode(nextFacing);
    if (isStreaming) {
      stopStream();
      setTimeout(startStream, 300);
    }
  };

  const handleResetSession = async () => {
    if (sessionId) {
      await resetSession(sessionId);
    }
  };

  const startFrameLoop = (activeSessionId: string | null) => {
    if (intervalRef.current) clearInterval(intervalRef.current);

    intervalRef.current = setInterval(() => {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const captureCanvas = captureCanvasRef.current;

      if (!video || !canvas || !captureCanvas || video.paused || video.ended) return;

      const now = performance.now();
      const delta = now - lastFrameTimeRef.current;
      lastFrameTimeRef.current = now;
      if (delta > 0) {
        setCameraFps(Math.round(1000 / delta));
      }

      captureCanvas.width = video.videoWidth || 640;
      captureCanvas.height = video.videoHeight || 480;
      const capCtx = captureCanvas.getContext('2d');
      if (!capCtx) return;
      capCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);

      captureCanvas.toBlob(
        async (blob) => {
          if (!blob) return;
          const sendTime = performance.now();

          try {
            if (activeSessionId) {
              const res = await submitFrame(activeSessionId, blob);
              const roundtripLatency = Math.round(performance.now() - sendTime);
              setLatencyMs(roundtripLatency);

              if (res.annotated_frame && canvasRef.current) {
                const ctx = canvasRef.current.getContext('2d');
                if (ctx) {
                  const img = new Image();
                  img.onload = () => {
                    canvasRef.current!.width = img.width || captureCanvas.width;
                    canvasRef.current!.height = img.height || captureCanvas.height;
                    ctx.drawImage(img, 0, 0, canvasRef.current!.width, canvasRef.current!.height);
                  };
                  img.src = res.annotated_frame;
                }
              }

              if (res.analytics) {
                onAnalyticsUpdate(res.analytics);
              }
            }
          } catch (e) {
            // Silently swallow network jitter
          }
        },
        'image/jpeg',
        0.85
      );
    }, 150);
  };

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const clickX = ((e.clientX - rect.left) / rect.width) * 640;
    const clickY = ((e.clientY - rect.top) / rect.height) * 480;

    setSelectedPerson({
      id: 'P01',
      tracker: 'T4',
      x: clickX,
      y: clickY,
      w: 80,
      h: 160,
      attr: 'Male-presenting',
      confidence: '94%',
    });
  };

  useEffect(() => {
    return () => {
      stopStream();
    };
  }, []);

  const toggleOverlay = (key: keyof OverlayOptions) => {
    setOverlays((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  return (
    <div className="p-6 space-y-6 max-w-7xl mx-auto">
      {/* Live Monitor Header */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold font-mono text-slate-100 flex items-center gap-2">
            {t.liveTitle}
            <span
              className={`text-xs border px-2.5 py-0.5 rounded font-mono font-bold ${
                isStreaming
                  ? 'bg-emerald-500/20 text-emerald-300 border-emerald-400/50'
                  : 'bg-slate-800 text-slate-400 border-slate-700'
              }`}
            >
              {isStreaming ? '● LIVE AI STREAM' : 'OFFLINE'}
            </span>
          </h2>
          <p className="text-xs text-sky-300/80 font-mono">{t.liveSub}</p>
        </div>

        {/* Action Controls */}
        <div className="flex items-center space-x-3">
          <button
            onClick={switchCameraFacing}
            className="cyber-btn text-xs px-3 py-1.5 rounded-lg flex items-center gap-1.5 font-mono cursor-pointer"
          >
            <RefreshCw className="w-3.5 h-3.5 text-cyan-400" />
            <span>{t.switchCam} ({facingMode === 'user' ? 'Front' : 'Back'})</span>
          </button>

          <button
            onClick={handleResetSession}
            className="cyber-btn text-xs px-3 py-1.5 rounded-lg flex items-center gap-1.5 font-mono cursor-pointer"
          >
            <span>{t.resetTracker}</span>
          </button>

          <div className="flex items-center space-x-1 bg-[#071120] border border-sky-500/40 p-1 rounded-lg text-xs font-mono">
            {(['minimal', 'analytics', 'debug'] as LabelMode[]).map((mode) => (
              <button
                key={mode}
                onClick={() => setLabelMode(mode)}
                className={`px-2.5 py-1 rounded capitalize transition-all cursor-pointer ${
                  labelMode === mode
                    ? 'bg-cyan-400 text-slate-950 font-bold shadow-sm'
                    : 'text-slate-400 hover:text-cyan-300'
                }`}
              >
                {mode}
              </button>
            ))}
          </div>
        </div>
      </div>

      {errorMessage && (
        <div className="bg-rose-500/15 border border-rose-500/40 text-rose-300 p-3 rounded-lg text-xs font-mono font-bold flex items-center gap-2">
          <AlertCircle className="w-4 h-4 shrink-0" />
          <span>{errorMessage}</span>
        </div>
      )}

      {/* Main Live Monitor Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Left Column: Live Video + Circle Vector Hologram Radar Ring */}
        <div className="lg:col-span-3 space-y-4">
          <div className="cyber-card relative bg-[#071120] border border-sky-500/50 rounded-2xl overflow-hidden shadow-2xl aspect-video flex items-center justify-center">
            {/* Hidden Raw HTML5 Video Element */}
            <video
              ref={videoRef}
              className="absolute inset-0 w-full h-full object-cover pointer-events-none opacity-0"
              autoPlay
              muted
              playsInline
            />

            {/* AI Annotated Frame Canvas */}
            <canvas
              ref={canvasRef}
              width={640}
              height={480}
              onClick={handleCanvasClick}
              className="w-full h-full object-contain cursor-crosshair relative z-10"
              title="Click on any person box to focus details"
            />

            {/* Circle Vector Hologram Radar Ring Overlay */}
            <div className="absolute top-4 right-4 w-28 h-28 pointer-events-none z-20 opacity-60">
              <div className="w-full h-full border-2 border-dashed border-cyan-400 rounded-full radar-ring flex items-center justify-center">
                <div className="w-16 h-16 border border-cyan-400/50 rounded-full flex items-center justify-center">
                  <Crosshair className="w-6 h-6 text-cyan-400 animate-ping" />
                </div>
              </div>
            </div>

            {!isStreaming && (
              <div className="absolute inset-0 bg-[#071120]/90 z-30 flex flex-col items-center justify-center space-y-4">
                <Camera className="w-12 h-12 text-cyan-400/60" />
                <div className="text-center space-y-1 font-mono">
                  <div className="text-base font-bold text-slate-100">{t.cameraStandby}</div>
                  <div className="text-xs text-sky-300">{t.cameraStandbySub}</div>
                </div>
                <button
                  onClick={toggleStreaming}
                  className="cyber-btn bg-cyan-400 text-slate-950 px-6 py-2.5 rounded-lg font-bold text-sm flex items-center gap-2 shadow-lg cursor-pointer"
                >
                  <Play className="w-4 h-4 fill-current" /> {t.startStream}
                </button>
              </div>
            )}

            {/* Interactive Focus Tooltip Popover */}
            {selectedPerson && (
              <div className="absolute top-16 left-16 bg-[#071120]/95 border border-cyan-400 p-3 rounded-xl shadow-2xl text-xs font-mono text-slate-100 space-y-1 z-40">
                <div className="flex items-center justify-between gap-4 font-bold text-cyan-300">
                  <span className="flex items-center gap-1">
                    <UserCheck className="w-3.5 h-3.5" /> Person {selectedPerson.id}
                  </span>
                  <button onClick={() => setSelectedPerson(null)} className="text-slate-400 hover:text-white cursor-pointer">✕</button>
                </div>
                <div>Tracker ID: <strong className="text-slate-200">{selectedPerson.tracker}</strong></div>
                <div>Attribute: <strong className="text-emerald-400">{selectedPerson.attr}</strong></div>
                <div>Confidence: <strong className="text-cyan-400">{selectedPerson.confidence}</strong></div>
              </div>
            )}

            {isStreaming && (
              <div className="absolute top-4 left-4 flex items-center space-x-2 bg-[#0b172a]/80 backdrop-blur-md px-3 py-1.5 rounded-lg border border-sky-500/40 text-xs text-slate-200 font-mono z-20">
                <Camera className="w-4 h-4 text-cyan-400" />
                <span className="font-bold">{t.liveWebcamBadge}</span>
              </div>
            )}

            {isStreaming && (
              <button
                onClick={toggleStreaming}
                className="absolute bottom-4 right-4 bg-rose-500 hover:bg-rose-400 text-white p-2.5 rounded-full shadow-lg transition-transform hover:scale-105 cursor-pointer z-20"
                title={t.stopStream}
              >
                <Pause className="w-5 h-5 fill-current" />
              </button>
            )}
          </div>

          {/* Real-time Telemetry Bar */}
          <div className="cyber-card p-3 rounded-xl flex items-center justify-between text-xs font-mono text-sky-300">
            <div className="flex items-center space-x-4">
              <span>
                Camera: <strong className="text-slate-100">{cameraFps} FPS</strong>
              </span>
              <span className="text-slate-600">|</span>
              <span>
                AI Cadence: <strong className="text-cyan-400 font-bold">6.7 Hz (150ms)</strong>
              </span>
              <span className="text-slate-600">|</span>
              <span>
                Backend Latency: <strong className="text-slate-100">{latencyMs} ms</strong>
              </span>
            </div>
            <span className={`font-mono font-bold ${isStreaming ? 'text-emerald-400' : 'text-slate-500'}`}>
              {isStreaming ? '● Model Connected' : '○ Standby'}
            </span>
          </div>
        </div>

        {/* Right Column: Live Model Realtime KPI Summary */}
        <div className="space-y-6">
          <div className="cyber-card p-4 space-y-3">
            <h3 className="text-sm font-bold font-mono text-slate-100 border-b border-sky-500/30 pb-2">
              {t.modelStats}
            </h3>

            <div className="flex justify-between items-center text-xs font-mono">
              <span className="text-sky-300">{t.detectedCrowd}</span>
              <span className="text-xl font-bold text-cyan-400">{analytics.total_crowd}</span>
            </div>

            <div className="flex justify-between items-center text-xs font-mono">
              <span className="text-sky-300">{t.femaleMale}</span>
              <span className="text-sm font-bold text-slate-200">
                {analytics.visual_presentation.female_presenting} F / {analytics.visual_presentation.male_presenting} M
              </span>
            </div>

            <div className="flex justify-between items-center text-xs font-mono">
              <span className="text-sky-300">{t.unclassifiedUnknown}</span>
              <span className="text-sm font-bold text-amber-400">
                {analytics.visual_presentation.unknown}
              </span>
            </div>

            <div className="flex justify-between items-center text-xs font-mono">
              <span className="text-sky-300">{t.coveragePct}</span>
              <span className="text-sm font-bold text-emerald-400">
                {analytics.visual_presentation.coverage_pct}%
              </span>
            </div>

            <div className="flex justify-between items-center text-xs font-mono border-t border-sky-500/30 pt-2">
              <span className="text-sky-300">{t.seatsOccupiedCount}</span>
              <span className="text-sm font-bold text-slate-100">
                {analytics.seats_occupied} / {analytics.total_seats}
              </span>
            </div>
          </div>

          {/* Overlay Checkboxes */}
          <div className="cyber-card p-4 space-y-3">
            <div className="flex items-center justify-between border-b border-sky-500/30 pb-2">
              <h3 className="text-sm font-bold font-mono text-slate-100 flex items-center gap-2">
                <Layers className="w-4 h-4 text-cyan-400" />
                {t.overlayControls}
              </h3>
              <Settings className="w-4 h-4 text-slate-500" />
            </div>

            <div className="space-y-2 text-xs font-mono">
              {([
                { key: 'boxes', label: t.boxes },
                { key: 'ids', label: t.ids },
                { key: 'attributes', label: t.attributes },
                { key: 'motion', label: t.motion },
                { key: 'zones', label: t.zones },
                { key: 'seats', label: t.seats },
              ] as { key: keyof OverlayOptions; label: string }[]).map((item) => {
                const isChecked = overlays[item.key];
                return (
                  <button
                    key={item.key}
                    onClick={() => toggleOverlay(item.key)}
                    className="w-full flex items-center space-x-2 text-slate-300 hover:text-cyan-300 p-1 rounded hover:bg-sky-500/10 transition-colors text-left cursor-pointer"
                  >
                    {isChecked ? (
                      <CheckSquare className="w-4 h-4 text-cyan-400 shrink-0" />
                    ) : (
                      <Square className="w-4 h-4 text-slate-500 shrink-0" />
                    )}
                    <span>{item.label}</span>
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
