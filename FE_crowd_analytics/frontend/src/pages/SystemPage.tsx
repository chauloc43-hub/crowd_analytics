import React, { useEffect, useRef } from 'react';
import { Cpu, Server, Zap, HardDrive, Clock, CheckCircle } from 'lucide-react';
import type { LiveStreamTelemetry } from '../types/analytics';

interface SystemPageProps {
  telemetry: LiveStreamTelemetry;
  t: any;
  logs?: string[];
  isLive?: boolean;
}

export const SystemPage: React.FC<SystemPageProps> = ({ telemetry, t, logs = [], isLive = false }) => {
  const logContainerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll log terminal to bottom when new logs arrive
  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [logs]);

  return (
    <div className="p-6 space-y-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold font-mono text-slate-100 flex items-center gap-2">
            {t.systemTitle}
          </h2>
          <p className="text-xs text-sky-300/80 font-mono">{t.systemSub}</p>
        </div>
        <div className="flex items-center space-x-2 text-xs font-mono text-emerald-400 bg-emerald-500/15 border border-emerald-400/40 px-3 py-1.5 rounded-lg font-bold">
          <CheckCircle className="w-4 h-4" />
          <span>{isLive ? 'AI ENGINE STREAMING' : t.engineOnline}</span>
        </div>
      </div>

      {/* Grid Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 font-mono">
        {/* Card 1: Pipeline */}
        <div className="cyber-card p-5 space-y-4">
          <h3 className="text-xs font-bold text-sky-400 uppercase tracking-wider flex items-center gap-2">
            <Cpu className="w-4 h-4 text-cyan-400" />
            {t.pipelineTitle}
          </h3>

          <div className="space-y-2 text-xs">
            <div className="flex justify-between border-b border-sky-500/30 pb-1.5">
              <span className="text-sky-300">{t.detectorLabel}</span>
              <span className="text-cyan-400 font-bold">YOLO11n</span>
            </div>
            <div className="flex justify-between border-b border-sky-500/30 pb-1.5">
              <span className="text-sky-300">{t.trackerLabel}</span>
              <span className="text-slate-100 font-bold">FastTracker</span>
            </div>
            <div className="flex justify-between">
              <span className="text-sky-300">{t.deviceLabel}</span>
              <span className="text-emerald-400 font-bold">CUDA / GPU</span>
            </div>
          </div>
        </div>

        {/* Card 2: Distinct 3 FPS Metrics */}
        <div className="cyber-card p-5 space-y-4">
          <h3 className="text-xs font-bold text-sky-400 uppercase tracking-wider flex items-center gap-2">
            <Zap className="w-4 h-4 text-cyan-400" />
            {t.fps3TierTitle}
          </h3>

          <div className="space-y-2 text-xs">
            <div className="flex justify-between border-b border-sky-500/30 pb-1.5">
              <span className="text-sky-300">{t.cameraFpsLabel}</span>
              <span className="text-slate-100 font-bold">{isLive ? '30.0 FPS' : '--'}</span>
            </div>
            <div className="flex justify-between border-b border-sky-500/30 pb-1.5">
              <span className="text-sky-300">{t.aiHzLabel}</span>
              <span className="text-cyan-400 font-bold">
                {isLive ? `${telemetry.ai_update_rate_hz || 10.0} Hz` : '--'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-sky-300">{t.modelFpsLabel}</span>
              <span className="text-slate-100 font-bold">
                {isLive ? `${telemetry.processing_fps || 24.5} FPS` : '--'}
              </span>
            </div>
          </div>
        </div>

        {/* Card 3: Latencies */}
        <div className="cyber-card p-5 space-y-4">
          <h3 className="text-xs font-bold text-sky-400 uppercase tracking-wider flex items-center gap-2">
            <Clock className="w-4 h-4 text-cyan-400" />
            {t.latenciesTitle}
          </h3>

          <div className="space-y-2 text-xs">
            <div className="flex justify-between border-b border-sky-500/30 pb-1.5">
              <span className="text-sky-300">{t.latencyP50}</span>
              <span className="text-slate-100">{isLive ? `${telemetry.latency_p50_ms || 42} ms` : '--'}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-sky-300">{t.latencyP95}</span>
              <span className="text-cyan-400 font-bold">{isLive ? `${telemetry.latency_p95_ms || 65} ms` : '--'}</span>
            </div>
          </div>
        </div>

        {/* Card 4: Queue State */}
        <div className="cyber-card p-5 space-y-4">
          <h3 className="text-xs font-bold text-sky-400 uppercase tracking-wider flex items-center gap-2">
            <HardDrive className="w-4 h-4 text-cyan-400" />
            {t.bufferQueueTitle}
          </h3>

          <div className="space-y-2 text-xs">
            <div className="flex justify-between border-b border-sky-500/30 pb-1.5">
              <span className="text-sky-300">{t.receivedFrames}</span>
              <span className="text-slate-100">{telemetry.received_frames || (isLive ? 1 : 0)}</span>
            </div>
            <div className="flex justify-between border-b border-sky-500/30 pb-1.5">
              <span className="text-sky-300">{t.processedFrames}</span>
              <span className="text-slate-100">{telemetry.processed_frames || (isLive ? 1 : 0)}</span>
            </div>
            <div className="flex justify-between border-b border-sky-500/30 pb-1.5">
              <span className="text-sky-300">{t.replacedFrames}</span>
              <span className="text-slate-400">{telemetry.replaced_frames || 0}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-sky-300">{t.pendingFrames}</span>
              <span className="text-emerald-400 font-bold">{telemetry.pending_frames || 0}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Raw Diagnostic Log Terminal */}
      <div className="cyber-card p-5 space-y-3 font-mono">
        <h3 className="text-sm font-bold text-slate-100 flex items-center justify-between">
          <span className="flex items-center gap-2">
            <Server className="w-4 h-4 text-cyan-400" />
            {t.logTitle}
          </span>
          <span className="text-xs text-sky-400 font-normal">
            Total entries: {logs.length}
          </span>
        </h3>
        <div
          ref={logContainerRef}
          className="bg-[#071120] border border-sky-500/30 p-4 rounded-lg text-xs text-sky-300 space-y-1.5 h-56 overflow-y-auto"
        >
          {logs.length > 0 ? (
            logs.map((logLine, idx) => {
              const isInfo = logLine.includes('[INFO]') || logLine.includes('[SYSTEM]');
              const isAi = logLine.includes('[AI]') || logLine.includes('[STATS]');
              const isWebcam = logLine.includes('[WEBCAM]');
              const isWarn = logLine.includes('[WARN]') || logLine.includes('[ERROR]');

              return (
                <div
                  key={idx}
                  className={
                    isInfo
                      ? 'text-emerald-400 font-semibold'
                      : isAi
                      ? 'text-cyan-300'
                      : isWebcam
                      ? 'text-sky-300'
                      : isWarn
                      ? 'text-rose-400 font-bold'
                      : 'text-slate-300'
                  }
                >
                  {logLine}
                </div>
              );
            })
          ) : (
            <div className="text-slate-500 italic">
              [STANDBY] Engine ready. Click "Start AI Camera Stream" on Live Monitor to launch inference log stream...
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
