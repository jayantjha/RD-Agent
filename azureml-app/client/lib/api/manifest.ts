/**
 * API client for manifest-related endpoints
 */
import { API_CONFIG } from '../config';

// Constants
const SESSION_ID = "a40ea1b4-22bc-43a9-975d-78bffb0c1d43";

/**
 * Fetches manifest data for a specific version (loop)
 */
export async function fetchManifestData(version: string) {
  const response = await fetch(
    `${API_CONFIG.BASE_URL}/data/manifest/${SESSION_ID}?loop=${version}`,
    {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    }
  );

  if (!response.ok) {
    throw new Error(`API request failed with status: ${response.status}`);
  }
  console.log(response);
  return response.json();
}
