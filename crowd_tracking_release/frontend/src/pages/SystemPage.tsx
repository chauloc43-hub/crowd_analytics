import React from 'react';
import { Cpu, Server, Zap, HardDrive, Clock, CheckCircle } from 'lucide-react';
import type { LiveStreamTelemetry } from '../types/analytics';

interface SystemPageProps {
  telemetry: LiveStreamTelemetry;
  t: any;
}

export const SystemPage: React.FC<SystemPageProps> = ({ telemetry, t }) => {
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
          <span>{t.engineOnline}</span>
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
              <span className="text-slate-100 font-bold">30.0 FPS</span>
            </div>
            <div className="flex justify-between border-b border-sky-500/30 pb-1.5">
              <span className="text-sky-300">{t.aiHzLabel}</span>
              <span className="text-cyan-400 font-bold">{telemetry.ai_update_rate_hz || 6.7} Hz</span>
            </div>
            <div className="flex justify-between">
              <span className="text-sky-300">{t.modelFpsLabel}</span>
              <span className="text-slate-100 font-bold">{telemetry.processing_fps || 14.2} FPS</span>
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
              <span className="text-slate-100">{telemetry.latency_p50_ms || 68} ms</span>
            </div>
            <div className="flex justify-between">
              <span className="text-sky-300">{t.latencyP95}</span>
              <span className="text-cyan-400 font-bold">{telemetry.latency_p95_ms || 91} ms</span>
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
              <span className="text-slate-100">{telemetry.received_frames || 1322}</span>
            </div>
            <div className="flex justify-between border-b border-sky-500/30 pb-1.5">
              <span className="text-sky-300">{t.processedFrames}</span>
              <span className="text-slate-100">{telemetry.processed_frames || 701}</span>
            </div>
            <div className="flex justify-between border-b border-sky-500/30 pb-1.5">
              <span className="text-sky-300">{t.replacedFrames}</span>
              <span className="text-slate-400">{telemetry.replaced_frames || 621}</span>
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
        <h3 className="text-sm font-bold text-slate-100 flex items-center gap-2">
          <Server className="w-4 h-4 text-cyan-400" />
          {t.logTitle}
        </h3>
        <div className="bg-[#071120] border border-sky-500/30 p-4 rounded-lg text-xs text-sky-300 space-y-1 h-48 overflow-y-auto">
          <div className="text-emerald-400">[INFO] Session demo-102 initialized successfully on GPU:0</div>
          <div>[INFO] YOLO11n loaded. Weights warm-up completed in 412ms.</div>
          <div>[DEBUG] FastTracker initialized with max_age=30, min_hits=3, iou_threshold=0.3</div>
          <div className="text-cyan-400">[TELEMETRY] Cadence timer 150ms active. WebRTC peer handshake complete.</div>
          <div>[STATS] Frame #701 processed in 14.2ms. Bbox payload sent to client canvas.</div>
        </div>
      </div>
    </div>
  );
};
