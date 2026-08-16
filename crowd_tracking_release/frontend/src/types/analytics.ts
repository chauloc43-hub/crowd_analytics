export type PageType = 'overview' | 'live' | 'analytics' | 'room' | 'system' | 'video';

export type LabelMode = 'minimal' | 'analytics' | 'debug';

export interface OverlayOptions {
  boxes: boolean;
  ids: boolean;
  attributes: boolean;
  motion: boolean;
  zones: boolean;
  seats: boolean;
  trajectory: boolean;
  heatmap: boolean;
}

export interface PersonDetection {
  id: string;
  tracker_id?: string;
  bbox: [number, number, number, number]; // [x1, y1, x2, y2]
  label?: string;
  attribute?: 'male' | 'female' | 'unknown';
  confidence?: number;
  motion?: 'moving' | 'stationary';
}

export interface ZoneData {
  name: string;
  peopleCount: number;
  density: number; // people/m2
  avgDwellTime: string; // e.g. "14m 02s"
  percentage: number;
}

export interface SeatData {
  id: string;
  row: number;
  col: number;
  section: 'left' | 'center' | 'right';
  status: 'vacant' | 'occupied' | 'disabled';
  personId?: string;
}

export interface SessionInfo {
  session_id: string;
  mode: string;
  created_at: string;
  camera_id?: string;
}

export interface LiveStreamTelemetry {
  received_frames?: number;
  processed_frames?: number;
  replaced_frames?: number;
  pending_frames?: number;
  current_fps?: number;
  ai_update_rate_hz?: number;
  processing_fps?: number;
  latency_p50_ms?: number;
  latency_p95_ms?: number;
}

export interface AnalyticsData {
  total_crowd: number;
  occupancy_rate: number; // e.g., 0.81 for 81%
  density_per_m2: number | null; // e.g. 0.41 or null if uncalibrated
  seats_occupied: number;
  total_seats: number;
  moving_count: number;
  stationary_count: number;
  flow_in: number;
  flow_out: number;
  net_flow: number;
  visual_presentation: {
    female_presenting: number;
    male_presenting: number;
    unknown: number;
    coverage_pct: number;
  };
  space_distribution: {
    front_pct: number;
    middle_pct: number;
    back_pct: number;
  };
  zones: ZoneData[];
}
