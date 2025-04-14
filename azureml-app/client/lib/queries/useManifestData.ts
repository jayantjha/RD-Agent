import { useQuery } from "@tanstack/react-query";
import { fetchManifestData } from "../api/manifest";

/**
 * React Query hook to fetch and cache manifest data
 */
export function useManifestData(version: string, sessionId: string) {
  // Make sure we handle the case when the hook is used outside a QueryClientProvider
  try {
    return useQuery({
      queryKey: ['manifest', version],
      enabled: !!version && !!sessionId, // Only fetch when version is available
      staleTime: 1000 * 60 * 5, // Consider data fresh for 5 minutes
      // Handle errors in the query function or globally via React Query's error boundaries
      queryFn: async () => {
        try {
          return await fetchManifestData(version, sessionId);
        } catch (error) {
          console.error('Error fetching manifest data:', error);
          throw error; // Ensure the error is propagated
        }
      },
    });
  } catch (error) {
    // If the hook is used outside of a QueryClientProvider, return a default structure
    console.error('QueryClient error:', error);
    return {
      data: undefined,
      error: new Error('QueryClient not found. Make sure to wrap your application with QueryClientProvider.'),
      isLoading: false,
      isError: true,
    };
  }
}
