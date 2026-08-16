import React, { useState } from 'react';
import { FileVideo, Upload, CheckCircle, AlertCircle, Play, Download, Users, Cpu, Activity, Eye, Code } from 'lucide-react';
import { analyzeVideo } from '../api/crowdApi';

export const VideoPage: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [showRawJson, setShowRawJson] = useState(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setError(null);
      setResult(null);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setAnalyzing(true);
    setError(null);

    try {
      const res = await analyzeVideo(file);
      setResult(res);
    } catch (err: any) {
      setError(err.message || 'Video analysis failed or demo GPU busy.');
    } finally {
      setAnalyzing(false);
    }
  };

  // Helper getters for analytics
  const analytics = result?.analytics || {};
  const crowd = analytics.crowd || {};
  const genderCounts = crowd.gender_counts || analytics.attributes?.visual_presentation?.gender_counts || {};
  const femaleCount = genderCounts.female || 0;
  const maleCount = genderCounts.male || 0;
  const unknownCount = genderCounts.unknown || 0;
  const totalGender = femaleCount + maleCount + unknownCount || 1;
  const femalePct = Math.round((femaleCount / totalGender) * 100);
  const malePct = Math.round((maleCount / totalGender) * 100);
  const unknownPct = Math.round((unknownCount / totalGender) * 100);

  const trajectory = analytics.trajectory || {};
  const performance = result?.performance || {};
  const inputInfo = result?.input || {};
  const annotatedUrl = result?.artifacts?.annotated_video_url;

  return (
    <div className="p-6 space-y-6 max-w-5xl mx-auto font-mono">
      <div>
        <h2 className="text-xl font-bold text-slate-100 flex items-center gap-2">
          <FileVideo className="w-6 h-6 text-cyan-400" />
          OFFLINE VIDEO ANALYSIS // UPLOAD & REVIEW
        </h2>
        <p className="text-xs text-sky-300/80">
          Upload short video clip (MP4 / WebM / AVI) for AI object detection, FastTracker tracking, gender classification & heatmap rendering
        </p>
      </div>

      <div className="cyber-card p-6 space-y-6">
        {/* Drop Zone */}
        <div className="border-2 border-dashed border-sky-500/40 hover:border-cyan-400 bg-[#071120] rounded-xl p-8 transition-all flex flex-col items-center justify-center space-y-3 cursor-pointer relative">
          <input
            type="file"
            accept="video/*"
            onChange={handleFileChange}
            className="absolute inset-0 opacity-0 cursor-pointer"
          />
          <FileVideo className="w-12 h-12 text-cyan-400 animate-pulse" />
          <div className="text-center">
            <div className="text-sm font-bold text-slate-100">
              {file ? file.name : 'Click or Drag & Drop Video File'}
            </div>
            <div className="text-xs text-sky-400 mt-1">
              {file ? `${(file.size / (1024 * 1024)).toFixed(2)} MB` : 'MP4, WebM, AVI up to 64MB'}
            </div>
          </div>
        </div>

        {/* Upload Button */}
        {file && (
          <div className="text-center">
            <button
              onClick={handleUpload}
              disabled={analyzing}
              className="cyber-btn bg-cyan-400 text-slate-950 hover:bg-cyan-300 px-8 py-3 rounded-lg font-bold text-sm inline-flex items-center gap-2 shadow-lg cursor-pointer transition-all disabled:opacity-50"
            >
              {analyzing ? (
                <>
                  <span className="w-4 h-4 rounded-full border-2 border-slate-950 border-t-transparent animate-spin" />
                  Processing Video AI Pipeline...
                </>
              ) : (
                <>
                  <Upload className="w-4 h-4" /> Start Video Analysis & Render Overlay
                </>
              )}
            </button>
          </div>
        )}

        {/* Error Alert */}
        {error && (
          <div className="bg-rose-500/15 border border-rose-500/40 text-rose-300 p-4 rounded-lg text-xs font-bold flex items-center gap-3">
            <AlertCircle className="w-5 h-5 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Results Container */}
        {result && (
          <div className="space-y-6 pt-4 border-t border-sky-500/20">
            <div className="flex items-center justify-between bg-emerald-500/10 border border-emerald-500/40 text-emerald-300 p-4 rounded-lg text-xs font-bold">
              <div className="flex items-center gap-2 text-sm">
                <CheckCircle className="w-5 h-5 text-emerald-400" />
                <span>Video Processing Completed Successfully!</span>
              </div>
              {annotatedUrl && (
                <a
                  href={annotatedUrl}
                  download={`annotated_${file?.name || 'video.mp4'}`}
                  className="bg-emerald-400 text-slate-950 px-3 py-1.5 rounded hover:bg-emerald-300 font-bold inline-flex items-center gap-1 text-xs cursor-pointer transition-colors"
                >
                  <Download className="w-3.5 h-3.5" /> Download Result Video
                </a>
              )}
            </div>

            {/* Video Player Display */}
            {annotatedUrl ? (
              <div className="space-y-2">
                <div className="flex items-center justify-between text-xs text-sky-300 font-bold">
                  <span className="flex items-center gap-1.5">
                    <Play className="w-4 h-4 text-cyan-400" /> ANNOTATED AI OVERLAY PLAYBACK
                  </span>
                  <span className="text-[11px] text-slate-400">
                    {inputInfo.duration_seconds}s @ {inputInfo.fps} FPS
                  </span>
                </div>
                <div className="relative rounded-xl overflow-hidden border border-cyan-500/40 bg-black shadow-2xl">
                  <video
                    src={annotatedUrl}
                    controls
                    autoPlay
                    loop
                    className="w-full max-h-[500px] object-contain mx-auto"
                  />
                </div>
              </div>
            ) : (
              <div className="bg-amber-500/10 border border-amber-500/30 text-amber-300 p-4 rounded-lg text-xs">
                Video playback preview unavailable, displaying analytical metrics breakdown below.
              </div>
            )}

            {/* Analytics Dashboard Grid */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-slate-200">
              {/* Crowd Count Card */}
              <div className="bg-[#081528] p-4 rounded-xl border border-sky-500/30 space-y-2">
                <div className="flex items-center gap-2 text-sky-400 text-xs font-bold">
                  <Users className="w-4 h-4" /> CROWD DETECTION
                </div>
                <div className="text-3xl font-bold text-slate-100">
                  {crowd.current_count ?? crowd.confirmed_active_count ?? 0}
                  <span className="text-xs text-slate-400 font-normal ml-2">People detected</span>
                </div>
                <div className="text-xs text-slate-400 space-y-1 pt-1 border-t border-sky-500/20">
                  <div>Confirmed Tracks: <span className="text-cyan-300 font-bold">{crowd.confirmed_active_count ?? 0}</span></div>
                  <div>Unique Track IDs: <span className="text-cyan-300 font-bold">{crowd.unique_track_count ?? 0}</span></div>
                </div>
              </div>

              {/* Gender Breakdown Card */}
              <div className="bg-[#081528] p-4 rounded-xl border border-sky-500/30 space-y-2">
                <div className="flex items-center gap-2 text-pink-400 text-xs font-bold">
                  <Eye className="w-4 h-4" /> VISUAL PRESENTATION
                </div>
                <div className="space-y-1 text-xs">
                  <div className="flex justify-between text-slate-300">
                    <span>Female: {femaleCount} ({femalePct}%)</span>
                    <span>Male: {maleCount} ({malePct}%)</span>
                  </div>
                  <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden flex">
                    <div style={{ width: `${femalePct}%` }} className="bg-pink-500 h-full" />
                    <div style={{ width: `${malePct}%` }} className="bg-cyan-400 h-full" />
                    <div style={{ width: `${unknownPct}%` }} className="bg-slate-500 h-full" />
                  </div>
                  <div className="text-[11px] text-slate-400 pt-1">
                    Unknown/Uncertain: {unknownCount} ({unknownPct}%)
                  </div>
                </div>
              </div>

              {/* Execution Telemetry Card */}
              <div className="bg-[#081528] p-4 rounded-xl border border-sky-500/30 space-y-2">
                <div className="flex items-center gap-2 text-emerald-400 text-xs font-bold">
                  <Cpu className="w-4 h-4" /> AI PERFORMANCE
                </div>
                <div className="text-2xl font-bold text-slate-100">
                  {performance.average_processing_fps ?? 0} <span className="text-xs text-slate-400 font-normal">FPS</span>
                </div>
                <div className="text-xs text-slate-400 space-y-1 pt-1 border-t border-sky-500/20">
                  <div>Processing Time: <span className="text-emerald-300 font-bold">{performance.wall_time_seconds ?? 0}s</span></div>
                  <div>Frames Processed: <span className="text-emerald-300 font-bold">{inputInfo.frames_processed ?? 0}</span></div>
                </div>
              </div>
            </div>

            {/* Motion & Spatial Details */}
            <div className="bg-[#081528] p-4 rounded-xl border border-sky-500/30 text-xs space-y-3">
              <div className="flex items-center gap-2 text-cyan-400 font-bold">
                <Activity className="w-4 h-4" /> MOTION & SPATIAL DYNAMICS
              </div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-slate-300">
                <div>
                  <span className="text-slate-400">Moving Tracks:</span>
                  <div className="text-base font-bold text-slate-100">{trajectory.moving_count ?? 0}</div>
                </div>
                <div>
                  <span className="text-slate-400">Stationary Tracks:</span>
                  <div className="text-base font-bold text-slate-100">{trajectory.stationary_count ?? 0}</div>
                </div>
                <div>
                  <span className="text-slate-400">Avg Speed:</span>
                  <div className="text-base font-bold text-slate-100">
                    {trajectory.mean_speed_reference_px_per_second ? trajectory.mean_speed_reference_px_per_second.toFixed(1) : 0} px/s
                  </div>
                </div>
                <div>
                  <span className="text-slate-400">Input FPS:</span>
                  <div className="text-base font-bold text-slate-100">{inputInfo.fps ?? 0} FPS</div>
                </div>
              </div>
            </div>

            {/* Toggle Raw Specs JSON */}
            <div className="pt-2">
              <button
                onClick={() => setShowRawJson(!showRawJson)}
                className="text-xs text-sky-400 hover:text-cyan-300 inline-flex items-center gap-1.5 cursor-pointer transition-colors"
              >
                <Code className="w-3.5 h-3.5" />
                {showRawJson ? 'Hide Raw Telemetry Data' : 'View Raw Telemetry Data (JSON)'}
              </button>

              {showRawJson && (
                <pre className="mt-3 bg-[#050c18] p-4 rounded-xl text-slate-300 font-mono text-[11px] overflow-x-auto border border-sky-500/30 max-h-96">
                  {JSON.stringify(result, null, 2)}
                </pre>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default VideoPage;
