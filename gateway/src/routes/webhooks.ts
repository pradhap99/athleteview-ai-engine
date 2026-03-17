import { Router, Request, Response } from 'express';
import { v4 as uuid } from 'uuid';
import { redis, logger } from '../index';

export const webhooksRouter = Router();

webhooksRouter.post('/', async (req: Request, res: Response) => {
  const { url, events } = req.body;
  if (!url || !events?.length) return res.status(400).json({ error: 'url and events required' });
  const id = `wh_${uuid().slice(0, 8)}`;
  const webhook = { id, url, events, active: true, created_at: Date.now(), last_triggered: null };
  await redis.set(`webhook:${id}`, JSON.stringify(webhook));
  logger.info('Webhook created', { id, url, events });
  res.status(201).json(webhook);
});

webhooksRouter.get('/', async (req: Request, res: Response) => {
  const keys = await redis.keys('webhook:*');
  const webhooks = await Promise.all(keys.map(async (k) => JSON.parse((await redis.get(k)) || '{}')));
  res.json({ webhooks, total: webhooks.length });
});

webhooksRouter.delete('/:id', async (req: Request, res: Response) => {
  await redis.del(`webhook:${req.params.id}`);
  res.status(204).send();
});
