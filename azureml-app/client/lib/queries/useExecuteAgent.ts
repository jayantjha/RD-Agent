import { useMutation } from "@tanstack/react-query";
import { config } from '@/lib/config/index';

interface AgentRequestParams {
  user_prompt: string;
  data_uri: string;
  chat_thread_id: string;
}

interface AgentResponse {
  status: string;
  agent_id: string;
  thread_id: string;
}

/**
 * Function to execute an agent task on the server
 */
const executeAgent = async (params: AgentRequestParams): Promise<AgentResponse> => {
  const response = await fetch(`${config.apiBaseUrl}/execute/agents`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(params),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `Failed to execute agent: ${response.statusText}`);
  }

  return response.json();
};

/**
 * React Query hook to execute an agent task
 */
export function useExecuteAgent() {
  try {
    return useMutation({
      mutationFn: (params: AgentRequestParams) => executeAgent(params),
      onError: (error) => {
        console.error('Error executing agent task:', error);
      },
    });
  } catch (error) {
    console.error('QueryClient error:', error);
    // Return a default structure if used outside QueryClientProvider
    return {
      mutate: () => {
        console.error('QueryClient not found. Make sure to wrap your application with QueryClientProvider.');
      },
      error: new Error('QueryClient not found. Make sure to wrap your application with QueryClientProvider.'),
      isLoading: false,
      isError: true,
    } as any;
  }
}