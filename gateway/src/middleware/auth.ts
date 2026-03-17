import { Request, Response, NextFunction } from 'express';
import jwt from 'jsonwebtoken';
import { redis } from '../index';

const JWT_SECRET = process.env.JWT_SECRET || 'athleteview-dev-secret';

export async function authMiddleware(req: Request, res: Response, next: NextFunction) {
  const authHeader = req.headers.authorization;
  if (!authHeader) return res.status(401).json({ error: 'Authorization header required' });

  // Support both Bearer JWT and API key
  if (authHeader.startsWith('Bearer ')) {
    try {
      const token = authHeader.slice(7);
      const decoded = jwt.verify(token, JWT_SECRET) as { sub: string; role: string };
      (req as any).user = decoded;
      return next();
    } catch {
      return res.status(401).json({ error: 'Invalid token' });
    }
  }

  if (authHeader.startsWith('ApiKey ')) {
    const key = authHeader.slice(7);
    const data = await redis.get(`apikey:${key}`);
    if (!data) return res.status(401).json({ error: 'Invalid API key' });
    (req as any).apiKey = JSON.parse(data);
    return next();
  }

  res.status(401).json({ error: 'Invalid authorization format' });
}
