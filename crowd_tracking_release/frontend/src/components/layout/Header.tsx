import React from 'react';
import { Radio, ChevronDown, Clock, ShieldCheck, Globe } from 'lucide-react';
import type { Language } from '../../i18n/translations';

interface HeaderProps {
  currentRoom: string;
  onRoomChange: (room: string) => void;
  sessionDuration: string;
  lang: Language;
  onToggleLanguage: () => void;
  t: any;
}

export const Header: React.FC<HeaderProps> = ({
  currentRoom,
  onRoomChange,
  sessionDuration,
  lang,
  onToggleLanguage,
  t,
}) => {
  return (
    <header className="h-16 bg-[#0b172a]/95 backdrop-blur-md border-b border-sky-500/40 px-4 flex items-center justify-between sticky top-0 z-50 shadow-xl">
      {/* Brand & Title - Cyber Frame Style */}
      <div className="flex items-center space-x-3">
        <div className="w-10 h-10 rounded-lg bg-sky-500/15 border border-cyan-400/50 flex items-center justify-center text-cyan-400 shadow-md shadow-cyan-500/20">
          <Radio className="w-5 h-5 animate-pulse text-cyan-400" />
        </div>
        <div>
          <h1 className="text-base font-bold text-slate-100 tracking-wide flex items-center gap-2 font-mono">
            {t.brandTitle}
            <span className="text-[10px] uppercase tracking-widest px-2 py-0.5 rounded bg-cyan-400/15 text-cyan-300 border border-cyan-400/30">
              CYBER FRAME
            </span>
          </h1>
          <p className="text-xs text-sky-300/70">{t.brandSub}</p>
        </div>
      </div>

      {/* Center: Room Selector & Live Status */}
      <div className="flex items-center space-x-3">
        <div className="relative">
          <select
            value={currentRoom}
            onChange={(e) => onRoomChange(e.target.value)}
            className="appearance-none bg-[#091526] text-sky-200 text-sm font-mono font-semibold border border-sky-500/40 rounded-lg px-3 py-1.5 pr-8 focus:outline-none focus:border-cyan-400 transition-all cursor-pointer shadow-inner"
          >
            <option value="Classroom A">Classroom A (Main Hall)</option>
            <option value="Classroom B">Classroom B (Lab 102)</option>
            <option value="Auditorium">Auditorium East</option>
          </select>
          <ChevronDown className="w-4 h-4 text-sky-400 absolute right-2.5 top-1/2 -translate-y-1/2 pointer-events-none" />
        </div>

        <div className="flex items-center space-x-2 bg-emerald-500/15 border border-emerald-400/40 px-3 py-1.5 rounded-lg text-emerald-300 text-xs font-mono font-bold shadow-sm shadow-emerald-500/20">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping" />
          <span>{t.liveStreamBadge}</span>
        </div>
      </div>

      {/* Right: Language Switcher & Session Timer */}
      <div className="flex items-center space-x-4">
        {/* Bilingual Switcher Button */}
        <button
          onClick={onToggleLanguage}
          className="px-3 py-1.5 rounded-lg bg-[#091526] border border-sky-500/40 text-cyan-300 hover:border-cyan-400 hover:scale-105 active:scale-95 transition-all text-xs font-mono font-bold flex items-center gap-1.5 cursor-pointer shadow-sm"
          title="Switch Language / Đổi Ngôn Ngữ"
        >
          <Globe className="w-3.5 h-3.5 text-cyan-400" />
          <span>{lang === 'vi' ? '🇻🇳 Tiếng Việt' : '🇬🇧 English'}</span>
        </button>

        <div className="hidden sm:flex items-center space-x-2 text-xs font-mono text-cyan-300 bg-[#091526] border border-sky-500/30 px-3 py-1.5 rounded-lg shadow-sm">
          <Clock className="w-3.5 h-3.5 text-cyan-400" />
          <span>{t.sessionTime} {sessionDuration}</span>
        </div>

        <div className="flex items-center space-x-1.5 text-xs text-slate-300 font-mono">
          <ShieldCheck className="w-4 h-4 text-emerald-400" />
          <span className="hidden md:inline">YOLO11n Ready</span>
        </div>
      </div>
    </header>
  );
};
