import { Router, Request, Response } from 'express';
import { BiometricReading } from '../types';
import { redis, logger } from '../index';

export const biometricsRouter = Router();

biometricsRouter.get('/:athlete_id/live', async (req: Request, res: Response) => {
  const data = await redis.get(`bio:live:${req.params.athlete_id}`);
  if (!data) return res.status(404).json({ error: 'No live biometric data' });
  res.json(JSON.parse(data));
});

biometricsRouter.get('/:athlete_id/history', async (req: Request, res: Response) => {
  const { from, to, limit } = req.query;
  const key = `bio:history:${req.params.athlete_id}`;
  const fromTs = from ? Number(from) : Date.now() - 3600000;
  const toTs = to ? Number(to) : Date.now();
  const maxItems = limit ? Math.min(Number(limit), 1000) : 100;
  const entries = await redis.zrangebyscore(key, fromTs, toTs, 'LIMIT', 0, maxItems);
  res.json({ athlete_id: req.params.athlete_id, readings: entries.map(e => JSON.parse(e)), count: entries.length });
});

biometricsRouter.post('/:athlete_id', async (req: Request, res: Response) => {
  const parsed = BiometricReading.safeParse({ ...req.body, athlete_id: req.params.athlete_id });
  if (!parsed.success) return res.status(400).json({ error: 'Invalid biometric data', details: parsed.error.issues });
  const data = parsed.data;
  await redis.set(`bio:live:${data.athlete_id}`, JSON.stringify(data), 'EX', 30);
  await redis.zadd(`bio:history:${data.athlete_id}`, data.timestamp, JSON.stringify(data));
  if (data.heart_rate > 190 || data.injury_risk === 'critical') {
    logger.warn('Biometric alert', { athlete_id: data.athlete_id, hr: data.heart_rate, risk: data.injury_risk });
  }
  res.status(201).json({ status: 'recorded' });
});
