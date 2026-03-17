import express from 'express';
import { createServer } from 'http';
import { Server as SocketServer } from 'socket.io';
import cors from 'cors';
import helmet from 'helmet';
import { Kafka, logLevel } from 'kafkajs';
import Redis from 'ioredis';
import { createLogger, format, transports } from 'winston';
import { Registry, collectDefaultMetrics, Counter, Histogram } from 'prom-client';
import { streamsRouter } from './routes/streams';
import { athletesRouter } from './routes/athletes';
import { biometricsRouter } from './routes/biometrics';
import { highlightsRouter } from './routes/highlights';
import { apikeysRouter } from './routes/apikeys';
import { webhooksRouter } from './routes/webhooks';
import { authMiddleware } from './middleware/auth';
import { rateLimiter } from './middleware/rateLimit';
import { setupWebSocket } from './websocket/handler';

const PORT = parseInt(process.env.API_PORT || '3000');
const REDIS_URL = process.env.REDIS_URL || 'redis://localhost:6379';
const KAFKA_BROKERS = (process.env.KAFKA_BROKERS || 'localhost:9092').split(',');

// Logger
export const logger = createLogger({
  level: process.env.LOG_LEVEL || 'info',
  format: format.combine(format.timestamp(), format.json()),
  transports: [new transports.Console()],
});

// Metrics
export const metricsRegistry = new Registry();
collectDefaultMetrics({ register: metricsRegistry });

export const httpRequestsTotal = new Counter({
  name: 'http_requests_total',
  help: 'Total HTTP requests',
  labelNames: ['method', 'route', 'status'],
  registers: [metricsRegistry],
});

export const httpRequestDuration = new Histogram({
  name: 'http_request_duration_seconds',
  help: 'HTTP request duration',
  labelNames: ['method', 'route'],
  buckets: [0.01, 0.05, 0.1, 0.5, 1, 5],
  registers: [metricsRegistry],
});

// Redis client
export const redis = new Redis(REDIS_URL, { maxRetriesPerRequest: 3 });

// Kafka producer
const kafka = new Kafka({
  clientId: 'athleteview-gateway',
  brokers: KAFKA_BROKERS,
  logLevel: logLevel.WARN,
});
export const kafkaProducer = kafka.producer();

// Express app
const app = express();
const httpServer = createServer(app);

// Socket.io
const io = new SocketServer(httpServer, {
  cors: { origin: '*', methods: ['GET', 'POST'] },
  transports: ['websocket', 'polling'],
});

// Middleware
app.use(helmet());
app.use(cors());
app.use(express.json({ limit: '10mb' }));

// Metrics middleware
app.use((req, res, next) => {
  const start = Date.now();
  res.on('finish', () => {
    const duration = (Date.now() - start) / 1000;
    httpRequestsTotal.inc({ method: req.method, route: req.path, status: res.statusCode.toString() });
    httpRequestDuration.observe({ method: req.method, route: req.path }, duration);
  });
  next();
});

// Health check
app.get('/health', (_req, res) => {
  res.json({ status: 'healthy', service: 'gateway', timestamp: Date.now() });
});

// Metrics endpoint
app.get('/metrics', async (_req, res) => {
  res.set('Content-Type', metricsRegistry.contentType);
  res.end(await metricsRegistry.metrics());
});

// API routes
app.use('/api/v1/streams', rateLimiter, authMiddleware, streamsRouter);
app.use('/api/v1/athletes', rateLimiter, authMiddleware, athletesRouter);
app.use('/api/v1/biometrics', rateLimiter, authMiddleware, biometricsRouter);
app.use('/api/v1/highlights', rateLimiter, authMiddleware, highlightsRouter);
app.use('/api/v1/apikeys', rateLimiter, authMiddleware, apikeysRouter);
app.use('/api/v1/webhooks', rateLimiter, authMiddleware, webhooksRouter);

// WebSocket
setupWebSocket(io);

// Start
async function start() {
  try {
    await kafkaProducer.connect();
    logger.info('Kafka producer connected');
  } catch (err) {
    logger.warn('Kafka not available, running without event bus', { error: err });
  }

  httpServer.listen(PORT, () => {
    logger.info(`AthleteView Gateway running on port ${PORT}`);
  });
}

start();

export { app, io };
