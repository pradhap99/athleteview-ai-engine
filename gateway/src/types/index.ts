import { z } from 'zod';

export const SportEnum = z.enum(['cricket', 'football', 'kabaddi', 'basketball', 'tennis']);
export type Sport = z.infer<typeof SportEnum>;

export const CameraPosition = z.enum(['chest', 'shoulder', 'back', 'helmet']);
export type CameraPos = z.infer<typeof CameraPosition>;

export const StreamCreateSchema = z.object({
  athlete_id: z.string().min(1),
  camera_position: CameraPosition,
  sport: SportEnum,
  match_id: z.string().min(1),
  resolution: z.enum(['720p', '1080p', '4K']).default('4K'),
  ai_features: z.array(z.enum([
    'super_resolution', 'tracking', 'pose_estimation',
    'biometrics', 'highlight_detection', 'stabilization', 'audio_enhancement'
  ])).default(['super_resolution', 'tracking', 'biometrics']),
});
export type StreamCreateInput = z.infer<typeof StreamCreateSchema>;

export const AthleteSchema = z.object({
  id: z.string(),
  name: z.string(),
  sport: SportEnum,
  team: z.string(),
  jersey_number: z.number().int().positive(),
  position: z.string().optional(),
  devices: z.array(z.object({
    device_id: z.string(),
    camera_position: CameraPosition,
    firmware_version: z.string(),
  })).default([]),
});
export type Athlete = z.infer<typeof AthleteSchema>;

export const BiometricReading = z.object({
  athlete_id: z.string(),
  timestamp: z.number(),
  heart_rate: z.number().min(30).max(250),
  spo2: z.number().min(70).max(100),
  body_temp: z.number().min(34).max(42),
  fatigue_index: z.number().min(0).max(100),
  sprint_speed: z.number().min(0).max(45),
  injury_risk: z.enum(['low', 'medium', 'high', 'critical']),
});
export type BiometricData = z.infer<typeof BiometricReading>;

export interface StreamState {
  id: string;
  athlete_id: string;
  status: 'initializing' | 'live' | 'paused' | 'ended';
  started_at: number;
  viewers: number;
  latency_ms: number;
  ai_pipeline_active: boolean;
  distribution: {
    platform: boolean;
    youtube: boolean;
    twitch: boolean;
    tv_broadcast: boolean;
  };
}

export interface HighlightEvent {
  id: string;
  stream_id: string;
  athlete_id: string;
  type: 'goal' | 'wicket' | 'catch' | 'tackle' | 'sprint' | 'fatigue_alert' | 'injury_risk';
  confidence: number;
  timestamp: number;
  metadata: Record<string, unknown>;
}
