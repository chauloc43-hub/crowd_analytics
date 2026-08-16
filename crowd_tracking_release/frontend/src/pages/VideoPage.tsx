import React, { useState } from 'react';
import { FileVideo, Upload, CheckCircle, AlertCircle } from 'lucide-react';
import { analyzeVideo } from '../api/crowdApi';

export const VideoPage: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

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

  return (
    <div className="p-6 space-y-6 max-w-4xl mx-auto font-mono">
      <div>
        <h2 className="text-xl font-bold text-slate-100 flex items-center gap-2">
          OFFLINE VIDEO ANALYSIS // UPLOAD
        </h2>
        <p className="text-xs text-sky-300/80">
          Upload short video clip MP4/WebM (max 60 seconds / 64 MB) for offline crowd analytics processing
        </p>
      </div>

      <div className="cyber-card p-8 space-y-6 text-center">
        <div className="border-2 border-dashed border-sky-500/40 hover:border-cyan-400 bg-[#071120] rounded-xl p-8 transition-colors flex flex-col items-center justify-center space-y-3 cursor-pointer relative">
          <input
            type="file"
            accept="video/*"
            onChange={handleFileChange}
            className="absolute inset-0 opacity-0 cursor-pointer"
          />
          <FileVideo className="w-12 h-12 text-cyan-400 animate-pulse" />
          <div>
            <div className="text-sm font-bold text-slate-100">
              {file ? file.name : 'Click or Drag & Drop Video File'}
            </div>
            <div className="text-xs text-sky-400 mt-1">
              {file ? `${(file.size / (1024 * 1024)).toFixed(2)} MB` : 'MP4, WebM up to 64MB'}
            </div>
          </div>
        </div>

        {file && (
          <button
            onClick={handleUpload}
            disabled={analyzing}
            className="cyber-btn bg-cyan-400 text-slate-950 px-6 py-2.5 rounded-lg font-bold text-sm inline-flex items-center gap-2 shadow-lg cursor-pointer"
          >
            {analyzing ? (
              <>
                <span className="w-4 h-4 rounded-full border-2 border-slate-950 border-t-transparent animate-spin" />
                Processing Offline Video...
              </>
            ) : (
              <>
                <Upload className="w-4 h-4" /> Start Offline Video Analysis
              </>
            )}
          </button>
        )}

        {error && (
          <div className="bg-rose-500/15 border border-rose-500/40 text-rose-300 p-3 rounded-lg text-xs font-bold flex items-center justify-center gap-2">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {result && (
          <div className="bg-emerald-500/15 border border-emerald-500/40 text-emerald-300 p-4 rounded-lg text-xs space-y-2 text-left">
            <div className="flex items-center gap-2 font-bold text-sm">
              <CheckCircle className="w-4 h-4" /> Analysis Completed Successfully!
            </div>
            <pre className="bg-[#071120] p-3 rounded text-slate-200 font-mono text-[11px] overflow-x-auto border border-sky-500/30">
              {JSON.stringify(result, null, 2)}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
};
