import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

/**
 * Safely parses a JSON string
 * @param jsonString The JSON string to parse
 * @returns The parsed object or null if parsing fails
 */
export function parseJSON(jsonString: string) {
  try {
    return JSON.parse(jsonString);
  } catch (error) {
    console.error('Error parsing JSON:', error);
    return null;
  }
}

// Constants for localStorage keys
export const STORAGE_KEYS = {
  METRICS: 'RDAGENT_METRIC'
};

// Function to extract metrics from manifest data
export function extractMetricsFromManifest(manifestData: any): number | null {
  if (!manifestData || !manifestData.metrics) return null;
  
  // Get the first metric key
  const metricKeys = Object.keys(manifestData.metrics);
  if (metricKeys.length === 0) return null;
  
  const firstMetricKey = metricKeys[0];
  const metricData = manifestData.metrics[firstMetricKey];
  
  // Return the ensemble metric if available
  if (metricData && 'ensemble' in metricData) {
    return metricData.ensemble;
  }
  
  return null;
}

// Function to extract metrics from manifest data
export function extractMetricKeyFromManifest(manifestData: any): string | null {
  if (!manifestData || !manifestData.metrics) return null;
  
  // Get the first metric key
  const metricKeys = Object.keys(manifestData.metrics);
  if (metricKeys.length === 0) return null;
  
  const firstMetricKey = metricKeys[0];
  return firstMetricKey;
}

// Function to store metrics in localStorage
export function storeMetric(sessionId: string, version: string, metricValue: number): void {
  try {
    // Get existing metrics from localStorage
    const existingData = localStorage.getItem(STORAGE_KEYS.METRICS);
    let metricsData: Record<string, { id: string, metrics: Record<string, number> }> = {};
    
    if (existingData) {
      metricsData = JSON.parse(existingData);
    }
    
    // Update or add the session metrics
    if (!metricsData[sessionId]) {
      metricsData[sessionId] = { id: sessionId, metrics: {} };
    }
    
    metricsData[sessionId].metrics[version] = metricValue;
    
    // Save back to localStorage
    localStorage.setItem(STORAGE_KEYS.METRICS, JSON.stringify(metricsData));
  } catch (error) {
    console.error("Error storing metrics in localStorage:", error);
  }
}

// Function to delete metrics from localStorage
export function deleteMetric(): void {
  try {
    localStorage.removeItem(STORAGE_KEYS.METRICS);
  } catch (error) {
    console.error("Error deleting metrics from localStorage:", error);
  }
}

// Function to get all metrics for a session
export function getSessionMetrics(sessionId: string): number[] {
  try {
    const existingData = localStorage.getItem(STORAGE_KEYS.METRICS);
    if (!existingData) return [];
    
    const metricsData = JSON.parse(existingData);
    if (!metricsData[sessionId]) return [];
    
    const sessionMetrics = metricsData[sessionId].metrics;
    
    // Convert to array sorted by version (numeric sort)
    return Object.keys(sessionMetrics)
      .sort((a, b) => parseInt(a) - parseInt(b))
      .map(version => sessionMetrics[version]);
  } catch (error) {
    console.error("Error retrieving metrics from localStorage:", error);
    return [];
  }
}

// Function to get accumulated metrics up to a specific version
export function getAccumulatedMetrics(sessionId: string, upToVersion: string): number[] {
  try {
    const allMetrics = getSessionMetrics(sessionId);
    const versionNum = parseInt(upToVersion);
    
    // If version is invalid, return empty array
    if (isNaN(versionNum)) return [];
    
    // Return metrics up to and including the specified version
    return allMetrics.slice(0, versionNum + 1);
  } catch (error) {
    console.error("Error getting accumulated metrics:", error);
    return [];
  }
}
