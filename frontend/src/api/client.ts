export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000';

export function apiUrl(path: string): string {
  if (path.startsWith('http://') || path.startsWith('https://')) {
    return path;
  }

  if (path.startsWith('/')) {
    return `${API_BASE_URL}${path}`;
  }

  return `${API_BASE_URL}/${path}`;
}

function readApiMessage(payload: unknown): string | null {
  if (!payload || typeof payload !== 'object') {
    return null;
  }

  const record = payload as Record<string, unknown>;
  if (typeof record.detail === 'string') {
    return record.detail;
  }
  if (typeof record.message === 'string') {
    return record.message;
  }

  return null;
}

export async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(apiUrl(path), init);
  const contentType = response.headers.get('content-type') ?? '';
  const payload = contentType.includes('application/json') ? (await response.json()) as unknown : null;

  if (!response.ok) {
    throw new Error(readApiMessage(payload) ?? `Request failed with status ${response.status}.`);
  }

  return payload as T;
}
