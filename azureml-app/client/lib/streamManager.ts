import { config, getApiUrl } from '@/lib/config/index';

let eventSource: EventSource | null = null;

/**
 * Starts an event stream connection to the server
 * @param threadId The thread ID to connect to
 * @param onMessage Callback function for handling incoming messages
 * @param onError Optional callback function for handling errors
 */
export function startEventStream(
  threadId: string, 
  onMessage: (data: any) => void,
  onError?: (error: any) => void
): void {
  // Close any existing connections
  if (eventSource) {
    stopEventStream();
  }

  const url = getApiUrl(`${config.updatesEndpoint}/${threadId}`);
  
  try {
    eventSource = new EventSource(url);
    
    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onMessage(data);
      } catch (error) {
        console.error('Error parsing event data:', error);
        if (onError) onError(error);
      }
    };
    
    eventSource.onerror = (error) => {
      console.error('EventSource error:', error);
      if (onError) onError(error);
      stopEventStream();
    };
  } catch (error) {
    console.error('Failed to start event stream:', error);
    if (onError) onError(error);
  }
}

/**
 * Stops the current event stream connection
 */
export function stopEventStream(): void {
  if (eventSource) {
    eventSource.close();
    eventSource = null;
  }
}
