import { Server, Socket } from 'socket.io';
import { logger } from '../index';
import { RoomManager } from './rooms';

export function setupWebSocket(io: Server) {
  const rooms = new RoomManager();

  io.on('connection', (socket: Socket) => {
    logger.info('WebSocket connected', { id: socket.id });

    socket.on('subscribe', (data: { channels: string[] }) => {
      for (const channel of data.channels) {
        socket.join(channel);
        rooms.addClient(channel, socket.id);
        logger.debug('Client subscribed', { socket: socket.id, channel });
      }
      socket.emit('subscribed', { channels: data.channels });
    });

    socket.on('unsubscribe', (data: { channels: string[] }) => {
      for (const channel of data.channels) {
        socket.leave(channel);
        rooms.removeClient(channel, socket.id);
      }
    });

    socket.on('disconnect', () => {
      rooms.removeAllClient(socket.id);
      logger.info('WebSocket disconnected', { id: socket.id });
    });
  });

  // Expose broadcast function for other services
  (io as any).broadcastToChannel = (channel: string, event: string, data: unknown) => {
    io.to(channel).emit(event, data);
  };
}
