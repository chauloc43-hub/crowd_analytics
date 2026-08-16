import React from 'react';
import {
  LayoutDashboard,
  Radio,
  Activity,
  Cpu,
  FileVideo,
} from 'lucide-react';
import type { PageType } from '../../types/analytics';

interface SidebarProps {
  activePage: PageType;
  onPageChange: (page: PageType) => void;
  t: any;
}

export const Sidebar: React.FC<SidebarProps> = ({ activePage, onPageChange, t }) => {
  const navItems: { id: PageType; label: string; icon: React.FC<{ className?: string }> }[] = [
    { id: 'overview', label: t.overview, icon: LayoutDashboard },
    { id: 'live', label: t.live, icon: Radio },
    { id: 'analytics', label: t.analytics, icon: Activity },
    { id: 'system', label: t.system, icon: Cpu },
    { id: 'video', label: t.video, icon: FileVideo },
  ];

  return (
    <aside className="hidden md:flex w-64 bg-[#0b172a]/95 backdrop-blur-md border-r border-sky-500/40 flex-col justify-between p-3 shrink-0 shadow-2xl">
      <div className="space-y-1.5">
        <div className="px-3 py-2 text-[11px] font-mono font-bold text-sky-400/80 uppercase tracking-widest flex items-center justify-between">
          <span>{t.navSection}</span>
          <span className="text-[9px] text-slate-500 font-mono">■ ■ □ □</span>
        </div>

        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activePage === item.id;
          return (
            <button
              key={item.id}
              onClick={() => onPageChange(item.id)}
              className={`w-full flex items-center space-x-3 px-3.5 py-2.5 rounded-lg text-sm font-mono transition-all duration-200 cursor-pointer ${
                isActive
                  ? 'bg-sky-500/20 text-cyan-300 border border-cyan-400/50 shadow-md shadow-cyan-500/20 font-bold'
                  : 'text-slate-300 hover:text-cyan-300 hover:bg-sky-500/10 hover:border hover:border-sky-500/30'
              }`}
            >
              <Icon className={`w-4 h-4 ${isActive ? 'text-cyan-400' : 'text-slate-400'}`} />
              <span>{item.label}</span>
              {item.id === 'live' && (
                <span className="ml-auto w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              )}
            </button>
          );
        })}
      </div>

      {/* Cyber Panel Footer Info */}
      <div className="mt-auto p-3 bg-[#071120] border border-sky-500/30 rounded-xl text-xs font-mono space-y-2">
        <div className="flex justify-between items-center text-slate-400">
          <span>AI PIPELINE</span>
          <span className="text-cyan-400 font-bold">YOLO11n</span>
        </div>
        <div className="flex justify-between items-center text-slate-400">
          <span>TRACKER</span>
          <span className="text-slate-200">FastTracker</span>
        </div>
        <div className="flex justify-between items-center text-slate-400">
          <span>TARGET AREA</span>
          <span className="text-slate-200">64 m²</span>
        </div>
      </div>
    </aside>
  );
};
