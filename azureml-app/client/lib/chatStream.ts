import { config, getApiUrl } from '@/lib/config/index';

export async function chatStream(
  userMessage: string,
  onMessage: (data: any) => void,
  threadId?: string,
  onError?: (error: any) => void
): Promise<{thread_id: string, run_id: string}> {


const url = getApiUrl(`${config.chatEndPoint}`);
const res = await fetch(url, {
    method: "POST",
    headers: {
        "Content-Type": "application/json",
    },
    body: JSON.stringify({
        message: userMessage,
        ...(threadId ? { thread_id: threadId } : {})
    }),
})
  const {thread_id, run_id} = await res.json()
  const streamUrl = getApiUrl(`${config.chatEndPoint}/stream/${thread_id}/${run_id}`);
  const eventSource = new EventSource(streamUrl);
  eventSource.onmessage = (event) => {
    try {
      const data = event.data;
      onMessage(data);
      eventSource.close();
    } catch (error) {
      console.error('Error parsing event data:', error);
      if (onError) onError(error);
    }
  };
  eventSource.onerror = (error) => {
    console.error('EventSource error:', error);
    if (onError) onError(error);
    eventSource.close();
  };
    // Return the thread_id and run_id for further use  
  return {thread_id, run_id};
}