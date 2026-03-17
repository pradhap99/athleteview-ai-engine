import { Router, Request, Response } from 'express';
import { v4 as uuid } from 'uuid';
import { StreamCreateSchema, StreamState } from '../types';
import { redis, kafkaProducer, logger } from '../index';

export const streamsRouter = Router();

streamsRouter.post('/', async (req: Request, res: Response) => {
  const parsed = StreamCreateSchema.safeParse(req.body);
  if (!parsed.success) {
    return res.status(400).json({ error: 'Invalid input', details: parsed.error.issues });
  }
  const { athlete_id, camera_position, sport, match_id, resolution, ai_features } = parsed.data;
  const streamId = `stream_${uuid().slice(0, 8)}`;
  const stream: StreamState = {
    id: streamId,
    athlete_id,
    status: 'initializing',
    started_at: Date.now(),
    viewers: 0,
    latency_ms: 0,
    ai_pipeline_active: ai_features.length > 0,
    distribution: { platform: true, youtube: false, twitch: false, tv_broadcast: false },
  };
  await redis.set(`stream:${streamId}`, JSON.stringify(stream), 'EX', 86400);
  await redis.sadd(`match:${match_id}:streams`, streamId);

  try {
    await kafkaProducer.send({
      topic: 'stream.events',
      messages: [{ key: streamId, value: JSON.stringify({ event: 'stream.created', stream, sport, camera_position, resolution, ai_features }) }],
    });
  } catch { /* kafka optional */ }

  logger.info('Stream created', { streamId, athlete_id, sport });
  res.status(201).json({ stream_id: streamId, srt_endpoint: `srt://ingest.athleteview.ai:9000?streamid=${streamId}`, rtmp_endpoint: `rtmp://ingest.athleteview.ai:1935/live/${streamId}`, status: 'initializing' });
});

streamsRouter.get('/:id', async (req: Request, res: Response) => {
  const data = await redis.get(`stream:${req.params.id}`);
  if (!data) return res.status(404).json({ error: 'Stream not found' });
  res.json(JSON.parse(data));
});

streamsRouter.get('/', async (req: Request, res: Response) => {
  const keys = await redis.keys('stream:stream_*');
  const streams = await Promise.all(keys.map(async (k) => JSON.parse((await redis.get(k)) || '{}')));
  res.json({ streams, total: streams.length });
});

streamsRouter.patch('/:id/status', async (req: Request, res: Response) => {
  const { status } = req.body;
  if (!['live', 'paused', 'ended'].includes(status)) {
    return res.status(400).json({ error: 'Invalid status' });
  }
  const data = await redis.get(`stream:${req.params.id}`);
  if (!data) return res.status(404).json({ error: 'Stream not found' });
  const stream: StreamState = { ...JSON.parse(data), status };
  await redis.set(`stream:${req.params.id}`, JSON.stringify(stream), 'EX', 86400);
  res.json(stream);
});

streamsRouter.delete('/:id', async (req: Request, res: Response) => {
  await redis.del(`stream:${req.params.id}`);
  res.status(204).send();
});
