import { describe, test, expect } from '@jest/globals';
import { StreamCreateSchema } from '../src/types';

describe('Stream Schema Validation', () => {
  test('valid stream creation', () => {
    const result = StreamCreateSchema.safeParse({
      athlete_id: 'kohli_18',
      camera_position: 'chest',
      sport: 'cricket',
      match_id: 'ipl_2026_mi_csk',
    });
    expect(result.success).toBe(true);
  });

  test('invalid sport rejected', () => {
    const result = StreamCreateSchema.safeParse({
      athlete_id: 'kohli_18',
      camera_position: 'chest',
      sport: 'quidditch',
      match_id: 'test',
    });
    expect(result.success).toBe(false);
  });
});
