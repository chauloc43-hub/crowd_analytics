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
  isLive?: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  currentRoom,
  onRoomChange,
  sessionDuration,
  lang,
  onToggleLanguage,
  t,
  isLive = false,
}) => {
  return (
    <header className="min-h-[64px] py-2 md:py-0 bg-[#0b172a]/95 backdrop-blur-md border-b border-sky-500/40 px-3 sm:px-6 flex flex-wrap md:flex-nowrap items-center justify-between sticky top-0 z-50 shadow-xl gap-2">
      {/* Brand & Title - Cyber Frame Style */}
      <div className="flex items-center space-x-2 sm:space-x-3">
        <div className="w-8 h-8 sm:w-10 sm:h-10 rounded-lg bg-sky-500/15 border border-cyan-400/50 flex items-center justify-center text-cyan-400 shadow-md shadow-cyan-500/20 shrink-0">
          <Radio className="w-4 h-4 sm:w-5 sm:h-5 animate-pulse text-cyan-400" />
        </div>
        <div>
          <h1 className="text-xs sm:text-base font-bold text-slate-100 tracking-wide flex items-center gap-1.5 font-mono">
            <span>{t.brandTitle}</span>
            <span className="hidden xs:inline-block text-[9px] uppercase tracking-widest px-1.5 py-0.5 rounded bg-cyan-400/15 text-cyan-300 border border-cyan-400/30">
              CYBER FRAME
            </span>
          </h1>
          <p className="hidden sm:block text-xs text-sky-300/70">{t.brandSub}</p>
        </div>
      </div>

      {/* Center: Room Selector & Live Status */}
      <div className="flex items-center space-x-2">
        <div className="relative">
          <select
            value={currentRoom}
            onChange={(e) => onRoomChange(e.target.value)}
            className="appearance-none bg-[#091526] text-sky-200 text-xs sm:text-sm font-mono font-semibold border border-sky-500/40 rounded-lg px-2.5 py-1 sm:py-1.5 pr-7 focus:outline-none focus:border-cyan-400 transition-all cursor-pointer shadow-inner max-w-[130px] sm:max-w-none truncate"
          >
            <option value="Classroom A">Classroom A (Main Hall)</option>
            <option value="Classroom B">Classroom B (Lab 102)</option>
            <option value="Auditorium">Auditorium East</option>
          </select>
          <ChevronDown className="w-3.5 h-3.5 text-sky-400 absolute right-2 top-1/2 -translate-y-1/2 pointer-events-none" />
        </div>

        {isLive && (
          <div className="flex items-center space-x-1.5 bg-emerald-500/15 border border-emerald-400/40 px-2 sm:px-3 py-1 sm:py-1.5 rounded-lg text-emerald-300 text-[11px] sm:text-xs font-mono font-bold shadow-sm shadow-emerald-500/20">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-ping" />
            <span className="hidden xs:inline">{t.liveStreamBadge}</span>
            <span className="xs:hidden">LIVE</span>
          </div>
        )}
      </div>

      {/* Right: Language Switcher & Session Timer */}
      <div className="flex items-center space-x-2 sm:space-x-4">
        {/* Bilingual Switcher Button */}
        <button
          onClick={onToggleLanguage}
          className="px-2 sm:px-3 py-1 sm:py-1.5 rounded-lg bg-[#091526] border border-sky-500/40 text-cyan-300 hover:border-cyan-400 hover:scale-105 active:scale-95 transition-all text-xs font-mono font-bold flex items-center gap-1 cursor-pointer shadow-sm"
          title="Switch Language / Đổi Ngôn Ngữ"
        >
          <Globe className="w-3.5 h-3.5 text-cyan-400" />
          <span className="hidden sm:inline">{lang === 'vi' ? '🇻🇳 Tiếng Việt' : '🇬🇧 English'}</span>
          <span className="sm:hidden">{lang === 'vi' ? '🇻🇳 VI' : '🇬🇧 EN'}</span>
        </button>

        <div className="hidden sm:flex items-center space-x-2 text-xs font-mono text-cyan-300 bg-[#091526] border border-sky-500/30 px-3 py-1.5 rounded-lg shadow-sm">
          <Clock className="w-3.5 h-3.5 text-cyan-400" />
          <span>{t.sessionTime} {sessionDuration}</span>
        </div>

        <div className="hidden md:flex items-center space-x-1.5 text-xs text-slate-300 font-mono">
          <ShieldCheck className="w-4 h-4 text-emerald-400" />
          <span>YOLO11n Ready</span>
        </div>
      </div>
    </header>
  );
};
