import { useQuery } from "@tanstack/react-query";
import { fetchFileContent } from "../api/fileContent";

/**
 * React Query hook to fetch and cache file content
 */
export function useFileContent(runId: string | undefined, filePath: string | undefined) {
  return useQuery({
    queryKey: ['fileContent', runId, filePath],
    enabled: !!runId && !!filePath, // Only fetch when both runId and filePath are available
    staleTime: 1000 * 60 * 5, // Consider data fresh for 5 minutes
    queryFn: async () => {
      if (!runId || !filePath) {
        throw new Error('Run ID and file path are required');
      }
      return await fetchFileContent(runId, filePath);
    },
  });
}
