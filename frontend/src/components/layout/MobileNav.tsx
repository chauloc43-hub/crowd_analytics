import React from 'react';
import {
  LayoutDashboard,
  Radio,
  Activity,
  Cpu,
} from 'lucide-react';
import type { PageType } from '../../types/analytics';

interface MobileNavProps {
  activePage: PageType;
  onPageChange: (page: PageType) => void;
  t: any;
}

export const MobileNav: React.FC<MobileNavProps> = ({ activePage, onPageChange, t }) => {
  const navItems: { id: PageType; label: string; icon: React.FC<{ className?: string }> }[] = [
    { id: 'overview', label: t.overview, icon: LayoutDashboard },
    { id: 'live', label: t.live, icon: Radio },
    { id: 'analytics', label: t.analytics, icon: Activity },
    { id: 'system', label: t.system, icon: Cpu },
  ];

  return (
    <div className="md:hidden fixed bottom-0 left-0 right-0 h-14 bg-[#0b172a]/95 backdrop-blur-md border-t border-sky-500/40 px-2 flex items-center justify-around z-50 shadow-2xl">
      {navItems.map((item) => {
        const Icon = item.icon;
        const isActive = activePage === item.id;
        return (
          <button
            key={item.id}
            onClick={() => onPageChange(item.id)}
            className={`flex flex-col items-center justify-center space-y-0.5 py-1 px-3 rounded-lg text-[10px] font-mono font-medium transition-all ${
              isActive ? 'text-cyan-400 font-bold' : 'text-slate-400 hover:text-cyan-300'
            }`}
          >
            <Icon className={`w-5 h-5 ${isActive ? 'text-cyan-400' : 'text-slate-400'}`} />
            <span>{item.label}</span>
          </button>
        );
      })}
    </div>
  );
};
