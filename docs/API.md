# AthleteView API Documentation

Base URL: `https://api.athleteview.ai/v1`

## Authentication

All requests require authentication via Bearer JWT or API Key:
```
Authorization: Bearer <jwt_token>
Authorization: ApiKey <api_key>
```

## Endpoints

### Streams
| Method | Path | Description |
|--------|------|-------------|
| POST | /streams | Create new stream |
| GET | /streams | List all streams |
| GET | /streams/:id | Get stream details |
| PATCH | /streams/:id/status | Update stream status |
| DELETE | /streams/:id | End stream |

### Athletes
| Method | Path | Description |
|--------|------|-------------|
| POST | /athletes | Register athlete |
| GET | /athletes | List athletes |
| GET | /athletes/:id | Get athlete profile |

### Biometrics
| Method | Path | Description |
|--------|------|-------------|
| GET | /biometrics/:athlete_id/live | Get live vitals |
| GET | /biometrics/:athlete_id/history | Get historical data |
| POST | /biometrics/:athlete_id | Submit reading |

### Highlights
| Method | Path | Description |
|--------|------|-------------|
| GET | /highlights/:match_id | List match highlights |

### WebSocket
Connect: `ws://api.athleteview.ai/ws`
Subscribe: `{"type": "subscribe", "channels": ["biometrics:kohli_18", "highlights:match_123"]}`
