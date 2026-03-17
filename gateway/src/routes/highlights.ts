import { Router, Request, Response } from 'express';
import { redis } from '../index';

export const highlightsRouter = Router();

highlightsRouter.get('/:match_id', async (req: Request, res: Response) => {
  const keys = await redis.keys(`highlight:${req.params.match_id}:*`);
  const highlights = await Promise.all(keys.map(async (k) => JSON.parse((await redis.get(k)) || '{}')));
  highlights.sort((a: any, b: any) => b.timestamp - a.timestamp);
  res.json({ match_id: req.params.match_id, highlights, total: highlights.length });
});

highlightsRouter.get('/:match_id/:id', async (req: Request, res: Response) => {
  const data = await redis.get(`highlight:${req.params.match_id}:${req.params.id}`);
  if (!data) return res.status(404).json({ error: 'Highlight not found' });
  res.json(JSON.parse(data));
});
