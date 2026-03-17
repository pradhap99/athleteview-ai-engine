import { Request, Response, NextFunction } from 'express';
import { redis } from '../index';

const TIERS: Record<string, number> = {
  free: 100,
  pro: 1000,
  enterprise: 10000,
};

export async function rateLimiter(req: Request, res: Response, next: NextFunction) {
  const identifier = (req as any).apiKey?.key || (req as any).user?.sub || req.ip || 'anon';
  const tier = (req as any).apiKey?.tier || 'free';
  const limit = TIERS[tier] || TIERS.free;
  const key = `ratelimit:${identifier}:${new Date().toISOString().slice(0, 13)}`;
  const current = await redis.incr(key);
  if (current === 1) await redis.expire(key, 3600);
  res.setHeader('X-RateLimit-Limit', limit);
  res.setHeader('X-RateLimit-Remaining', Math.max(0, limit - current));
  if (current > limit) return res.status(429).json({ error: 'Rate limit exceeded', limit, reset: 3600 });
  next();
}
