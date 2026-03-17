import { Router, Request, Response } from 'express';
import { v4 as uuid } from 'uuid';
import { redis, logger } from '../index';

export const apikeysRouter = Router();

apikeysRouter.post('/', async (req: Request, res: Response) => {
  const { name, tier } = req.body;
  const key = `av_${uuid().replace(/-/g, '')}`;
  const apiKey = { key, name: name || 'default', tier: tier || 'free', created_at: Date.now(), requests_today: 0 };
  await redis.set(`apikey:${key}`, JSON.stringify(apiKey));
  logger.info('API key created', { name: apiKey.name, tier: apiKey.tier });
  res.status(201).json(apiKey);
});

apikeysRouter.get('/', async (req: Request, res: Response) => {
  const keys = await redis.keys('apikey:*');
  const apiKeys = await Promise.all(keys.map(async (k) => JSON.parse((await redis.get(k)) || '{}')));
  res.json({ api_keys: apiKeys, total: apiKeys.length });
});
