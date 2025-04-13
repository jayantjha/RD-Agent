import { config, getApiUrl } from '@/lib/config/index';

/**
 * Fetches file content from the server
 * @param runId The run ID
 * @param filePath The full path to the file
 * @returns The file content as a string
 */
export async function fetchFileContent(runId: string, filePath: string): Promise<string> {
  try {
    const url = getApiUrl(`${config.dataEndpoint}/${runId}?path=${encodeURIComponent(filePath)}`);
    const response = await fetch(url);
    
    if (!response.ok) {
      throw new Error(`Error fetching file content: ${response.status} ${response.statusText}`);
    }
    
    return await response.text();
  } catch (error) {
    console.error("Failed to fetch file content:", error);
    throw error;
  }
}
