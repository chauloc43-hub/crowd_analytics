import React, { useState, useRef } from 'react';
import { Armchair, Save, Crosshair, Plus, Minus } from 'lucide-react';
import type { SeatData } from '../types/analytics';

interface RoomSetupPageProps {
  t: any;
}

export const RoomSetupPage: React.FC<RoomSetupPageProps> = ({ t }) => {
  const [tab, setTab] = useState<'layout' | 'calibration'>('layout');
  const [rows, setRows] = useState(4);
  const [areaM2, setAreaM2] = useState(64);
  const [calibrationSaved, setCalibrationSaved] = useState(false);

  const initialSeats: SeatData[] = [];
  for (let r = 1; r <= 4; r++) {
    for (let c = 1; c <= 2; c++) {
      initialSeats.push({ id: `L-${r}-${c}`, row: r, col: c, section: 'left', status: (r === 2 && c === 1) || (r === 3 && c === 2) ? 'occupied' : 'vacant' });
    }
    for (let c = 1; c <= 4; c++) {
      initialSeats.push({ id: `C-${r}-${c}`, row: r, col: c, section: 'center', status: (r === 2 && c <= 3) || (r === 3 && c >= 2) ? 'occupied' : 'vacant' });
    }
    for (let c = 1; c <= 2; c++) {
      initialSeats.push({ id: `R-${r}-${c}`, row: r, col: c, section: 'right', status: (r === 2 && c === 2) ? 'occupied' : 'vacant' });
    }
  }

  const [seats, setSeats] = useState<SeatData[]>(initialSeats);

  const toggleSeatStatus = (id: string) => {
    setSeats((prev) =>
      prev.map((s) => {
        if (s.id !== id) return s;
        if (s.status === 'vacant') return { ...s, status: 'disabled' };
        if (s.status === 'disabled') return { ...s, status: 'vacant' };
        return s;
      })
    );
  };

  const canvasRef = useRef<HTMLCanvasElement>(null);

  return (
    <div className="p-6 space-y-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold font-mono text-slate-100 flex items-center gap-2">
            {t.roomTitle}
          </h2>
          <p className="text-xs text-sky-300/80 font-mono">{t.roomSub}</p>
        </div>

        <div className="flex space-x-2 bg-[#071120] border border-sky-500/40 p-1 rounded-lg text-xs font-mono">
          <button
            onClick={() => setTab('layout')}
            className={`px-3 py-1.5 rounded-md font-bold flex items-center gap-1.5 transition-all cursor-pointer ${
              tab === 'layout'
                ? 'bg-cyan-400 text-slate-950 shadow-sm'
                : 'text-slate-400 hover:text-cyan-300'
            }`}
          >
            <Armchair className="w-3.5 h-3.5" />
            {t.layoutTab}
          </button>

          <button
            onClick={() => setTab('calibration')}
            className={`px-3 py-1.5 rounded-md font-bold flex items-center gap-1.5 transition-all cursor-pointer ${
              tab === 'calibration'
                ? 'bg-cyan-400 text-slate-950 shadow-sm'
                : 'text-slate-400 hover:text-cyan-300'
            }`}
          >
            <Crosshair className="w-3.5 h-3.5" />
            {t.calibTab}
          </button>
        </div>
      </div>

      {tab === 'layout' && (
        <div className="space-y-6">
          {/* Room Specs Control Bar */}
          <div className="cyber-card p-5 rounded-xl flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center space-x-8 font-mono">
              <div>
                <label className="text-xs text-sky-300 block mb-1">{t.roomArea}</label>
                <div className="flex items-center space-x-2">
                  <input
                    type="number"
                    value={areaM2}
                    onChange={(e) => setAreaM2(Number(e.target.value))}
                    className="w-20 bg-[#071120] border border-sky-500/40 text-cyan-300 font-bold text-center px-2 py-1 rounded text-sm focus:outline-none focus:border-cyan-400"
                  />
                  <span className="text-xs text-slate-300">m²</span>
                </div>
              </div>

              <div>
                <label className="text-xs text-sky-300 block mb-1">{t.seatingPattern}</label>
                <div className="text-sm font-bold text-slate-100">2 - 4 - 2</div>
              </div>

              <div>
                <label className="text-xs text-sky-300 block mb-1">{t.rowsCount}</label>
                <div className="flex items-center space-x-2">
                  <button
                    onClick={() => setRows(Math.max(1, rows - 1))}
                    className="cyber-btn p-1 rounded cursor-pointer"
                  >
                    <Minus className="w-3.5 h-3.5" />
                  </button>
                  <span className="font-bold text-slate-100 w-6 text-center">{rows}</span>
                  <button
                    onClick={() => setRows(rows + 1)}
                    className="cyber-btn p-1 rounded cursor-pointer"
                  >
                    <Plus className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
            </div>

            <div className="text-right font-mono">
              <div className="text-xs text-sky-300">{t.configuredSeats}</div>
              <div className="text-xl font-bold text-cyan-400">
                {seats.filter((s) => s.status !== 'disabled').length} / {seats.length}
              </div>
            </div>
          </div>

          {/* Interactive Seat Map Grid */}
          <div className="cyber-card p-6 rounded-xl space-y-6">
            {/* FRONT / BOARD */}
            <div className="w-full py-2 bg-cyan-400/10 border border-cyan-400/40 rounded-lg text-center text-xs font-mono font-bold text-cyan-300 uppercase tracking-widest">
              {t.frontLectern}
            </div>

            {/* Grid Rows */}
            <div className="space-y-4 max-w-3xl mx-auto py-4 font-mono">
              {Array.from({ length: rows }).map((_, rIdx) => {
                const rowNum = rIdx + 1;
                const leftSeats = seats.filter((s) => s.row === rowNum && s.section === 'left');
                const centerSeats = seats.filter((s) => s.row === rowNum && s.section === 'center');
                const rightSeats = seats.filter((s) => s.row === rowNum && s.section === 'right');

                return (
                  <div key={rowNum} className="flex items-center justify-center space-x-8">
                    <span className="text-xs text-sky-300 w-12 font-bold">Dãy {rowNum}</span>

                    {/* Left Section (2) */}
                    <div className="flex space-x-2">
                      {leftSeats.map((s) => (
                        <button
                          key={s.id}
                          onClick={() => toggleSeatStatus(s.id)}
                          className={`w-8 h-8 rounded-full border flex items-center justify-center text-xs font-mono transition-transform hover:scale-125 cursor-pointer shadow-sm ${
                            s.status === 'occupied'
                              ? 'bg-cyan-400 text-slate-950 border-cyan-300 font-bold shadow-cyan-400/50 shadow-md'
                              : s.status === 'vacant'
                              ? 'bg-[#071120] border-sky-500/40 text-sky-300 hover:border-cyan-400'
                              : 'bg-rose-500/15 border-rose-500/30 text-rose-400 opacity-40 line-through'
                          }`}
                        >
                          {s.status === 'occupied' ? '●' : s.status === 'disabled' ? '✕' : '○'}
                        </button>
                      ))}
                    </div>

                    <div className="w-6 border-r border-dashed border-sky-500/40 h-6" />

                    {/* Center Section (4) */}
                    <div className="flex space-x-2">
                      {centerSeats.map((s) => (
                        <button
                          key={s.id}
                          onClick={() => toggleSeatStatus(s.id)}
                          className={`w-8 h-8 rounded-full border flex items-center justify-center text-xs font-mono transition-transform hover:scale-125 cursor-pointer shadow-sm ${
                            s.status === 'occupied'
                              ? 'bg-cyan-400 text-slate-950 border-cyan-300 font-bold shadow-cyan-400/50 shadow-md'
                              : s.status === 'vacant'
                              ? 'bg-[#071120] border-sky-500/40 text-sky-300 hover:border-cyan-400'
                              : 'bg-rose-500/15 border-rose-500/30 text-rose-400 opacity-40 line-through'
                          }`}
                        >
                          {s.status === 'occupied' ? '●' : s.status === 'disabled' ? '✕' : '○'}
                        </button>
                      ))}
                    </div>

                    <div className="w-6 border-r border-dashed border-sky-500/40 h-6" />

                    {/* Right Section (2) */}
                    <div className="flex space-x-2">
                      {rightSeats.map((s) => (
                        <button
                          key={s.id}
                          onClick={() => toggleSeatStatus(s.id)}
                          className={`w-8 h-8 rounded-full border flex items-center justify-center text-xs font-mono transition-transform hover:scale-125 cursor-pointer shadow-sm ${
                            s.status === 'occupied'
                              ? 'bg-cyan-400 text-slate-950 border-cyan-300 font-bold shadow-cyan-400/50 shadow-md'
                              : s.status === 'vacant'
                              ? 'bg-[#071120] border-sky-500/40 text-sky-300 hover:border-cyan-400'
                              : 'bg-rose-500/15 border-rose-500/30 text-rose-400 opacity-40 line-through'
                          }`}
                        >
                          {s.status === 'occupied' ? '●' : s.status === 'disabled' ? '✕' : '○'}
                        </button>
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>

            {/* REAR */}
            <div className="w-full py-2 bg-[#071120] border border-sky-500/30 rounded-lg text-center text-xs font-mono text-sky-400 uppercase tracking-widest">
              {t.rearDoor}
            </div>
          </div>
        </div>
      )}

      {tab === 'calibration' && (
        <div className="cyber-card p-6 space-y-6">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-base font-bold font-mono text-slate-100">{t.homographyTitle}</h3>
              <p className="text-xs text-sky-300/80 font-mono">
                Draw seating plane polygon to transform pixel density into physical people/m²
              </p>
            </div>
            <button
              onClick={() => setCalibrationSaved(true)}
              className="cyber-btn bg-cyan-400 text-slate-950 px-4 py-2 rounded-lg font-bold text-xs flex items-center gap-1.5 cursor-pointer"
            >
              <Save className="w-4 h-4" />
              {t.saveCalibBtn}
            </button>
          </div>

          {calibrationSaved && (
            <div className="bg-emerald-500/15 border border-emerald-500/40 text-emerald-300 p-3 rounded-lg text-xs font-mono font-bold">
              {t.calibSavedMsg}
            </div>
          )}

          {/* Interactive Calibration Canvas */}
          <div className="relative bg-[#071120] border border-sky-500/40 rounded-xl overflow-hidden aspect-video flex items-center justify-center">
            <canvas ref={canvasRef} width={800} height={450} className="w-full h-full object-contain" />
            <div className="absolute inset-0 p-8 pointer-events-none flex flex-col justify-between">
              <div className="border-2 border-dashed border-cyan-400/50 bg-cyan-400/5 rounded-lg flex items-center justify-center">
                <span className="bg-[#0b172a] text-cyan-300 font-mono text-xs px-3 py-1 rounded border border-cyan-400/40 font-bold">
                  {t.seatingPlaneLabel}
                </span>
              </div>
            </div>
          </div>

          {/* Tool Buttons */}
          <div className="flex flex-wrap gap-3 text-xs font-mono">
            <button className="cyber-btn px-3 py-2 rounded-lg cursor-pointer">[ Vẽ Polygon Phòng ]</button>
            <button className="cyber-btn px-3 py-2 rounded-lg cursor-pointer">[ Vẽ Vùng ROI Cửa ]</button>
            <button className="cyber-btn px-3 py-2 rounded-lg cursor-pointer">[ Vẽ Vạch Đếm ]</button>
          </div>
        </div>
      )}
    </div>
  );
};
