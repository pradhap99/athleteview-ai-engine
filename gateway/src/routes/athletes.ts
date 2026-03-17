import { Router, Request, Response } from 'express';
import { AthleteSchema } from '../types';
import { redis, logger } from '../index';

export const athletesRouter = Router();

athletesRouter.post('/', async (req: Request, res: Response) => {
  const parsed = AthleteSchema.safeParse(req.body);
  if (!parsed.success) return res.status(400).json({ error: 'Invalid input', details: parsed.error.issues });
  await redis.set(`athlete:${parsed.data.id}`, JSON.stringify(parsed.data));
  logger.info('Athlete registered', { id: parsed.data.id, name: parsed.data.name });
  res.status(201).json(parsed.data);
});

athletesRouter.get('/:id', async (req: Request, res: Response) => {
  const data = await redis.get(`athlete:${req.params.id}`);
  if (!data) return res.status(404).json({ error: 'Athlete not found' });
  res.json(JSON.parse(data));
});

athletesRouter.get('/', async (req: Request, res: Response) => {
  const keys = await redis.keys('athlete:*');
  const athletes = await Promise.all(keys.map(async (k) => JSON.parse((await redis.get(k)) || '{}')));
  res.json({ athletes, total: athletes.length });
});
