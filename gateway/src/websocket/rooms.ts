export class RoomManager {
  private rooms: Map<string, Set<string>> = new Map();

  addClient(room: string, clientId: string) {
    if (!this.rooms.has(room)) this.rooms.set(room, new Set());
    this.rooms.get(room)!.add(clientId);
  }

  removeClient(room: string, clientId: string) {
    this.rooms.get(room)?.delete(clientId);
    if (this.rooms.get(room)?.size === 0) this.rooms.delete(room);
  }

  removeAllClient(clientId: string) {
    for (const [room, clients] of this.rooms) {
      clients.delete(clientId);
      if (clients.size === 0) this.rooms.delete(room);
    }
  }

  getRoomSize(room: string): number {
    return this.rooms.get(room)?.size || 0;
  }

  getAllRooms(): { room: string; clients: number }[] {
    return Array.from(this.rooms.entries()).map(([room, clients]) => ({ room, clients: clients.size }));
  }
}
