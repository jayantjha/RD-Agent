/**
 * API client for manifest-related endpoints
 * This file provides functions for interacting with manifest API endpoints
 */
import { config } from '../config/index';

// Constants
// const SESSION_ID = "0ff86955-de91-4f59-b21a-cb4367cd912e";

/**
 * Configuration for manifest API
 */
export const MANIFEST_API_CONFIG = {
  BASE_URL: config.apiBaseUrl
};

/**
 * Fetches manifest data for a specific version (loop)
 * @param version The version identifier (loop)
 * @returns Promise with the manifest data
 */
export async function fetchManifestData(version: string, sessionId: string) {
  const response = await fetch(
    `${MANIFEST_API_CONFIG.BASE_URL}/data/manifest/${sessionId}?loop=${version}`,
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
  
  return response.json();
}
