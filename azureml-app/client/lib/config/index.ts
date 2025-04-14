/**
 * Application configuration
 * This file provides environment-specific configuration settings
 */

// Determine the current environment
const environment = process.env.NEXT_PUBLIC_ENV || 'development';

// Configuration for different environments
const configurations = {
  development: {
    apiBaseUrl: 'http://localhost:8000',
    dataEndpoint: '/data/file',
    updatesEndpoint: '/updates/saved',
  },
  production: {
    apiBaseUrl: process.env.NEXT_PUBLIC_API_BASE_URL || 'https://api.azureml-app.com',
    dataEndpoint: '/data/file',
    updatesEndpoint: '/updates/saved',
  },
  test: {
    apiBaseUrl: 'http://localhost:8001',
    dataEndpoint: '/data/file',
    updatesEndpoint: '/updates/saved',
  }
};

// Export the configuration for the current environment
export const config = configurations[environment as keyof typeof configurations];

/**
 * Helper function to get a full API URL
 * @param path The API endpoint path
 * @returns The complete API URL
 */
export function getApiUrl(path: string): string {
  return `${config.apiBaseUrl}${path}`;
}
