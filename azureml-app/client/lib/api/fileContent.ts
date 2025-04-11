/**
 * Fetches file content from the server
 * @param runId The run ID
 * @param filePath The full path to the file
 * @returns The file content as a string
 */
export async function fetchFileContent(runId: string, filePath: string): Promise<string> {
  try {
    const response = await fetch(`http://127.0.0.1:8000/data/file/${runId}?path=${encodeURIComponent(filePath)}`);
    
    if (!response.ok) {
      throw new Error(`Error fetching file content: ${response.status} ${response.statusText}`);
    }
    
    return await response.text();
  } catch (error) {
    console.error("Failed to fetch file content:", error);
    throw error;
  }
}
