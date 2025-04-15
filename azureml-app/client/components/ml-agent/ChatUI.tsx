"use client"

import React, { useEffect, useRef } from "react"
import { FileText, Paperclip, Send } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { useExecuteAgent } from "@/lib/queries/useExecuteAgent"

interface ChatUIProps {
  messages: {
    role: "user" | "agent"
    content: string
    timestamp: Date
    showFiles?: boolean
  }[]
  userMessage: string
  setUserMessage: (message: string) => void
  handleSendMessage: (threadId: string) => void
  readyToStart: boolean
  setReadyToStart: (ready: boolean) => void
  files: File[]
  startAgent: () => void
  handleFileChange: (e: React.ChangeEvent<HTMLInputElement>) => void
  messagesEndRef: React.RefObject<HTMLDivElement>
  chatThreadId: string | undefined;
}

export function ChatUI({
  messages,
  userMessage,
  setUserMessage,
  handleSendMessage,
  readyToStart,
  setReadyToStart,
  startAgent,
  files,
  handleFileChange,
  messagesEndRef,
  chatThreadId,
}: ChatUIProps) {

  const { 
    mutate, 
    isLoading, 
    isError, 
    error, 
    data, 
    isSuccess 
  } = useExecuteAgent();

  const handleExecuteClick = () => {
    // Call mutate with the required parameters
    mutate({
      user_prompt: "this is a user_prompt",
      data_uri: "https://google.com",
      chat_thread_id: chatThreadId,
    });
  };

  useEffect(() => {
    if (isSuccess && data.thread_id) {
      console.log("thread_id", data.thread_id);
      handleSendMessage(data.thread_id);
    }
  }, [isSuccess, data, handleSendMessage])

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-white">
      <div className="w-full max-w-3xl flex flex-col items-center">
        <div className="flex flex-col items-center justify-center mb-12 mt-20">
          <Avatar className="h-16 w-16 mb-4">
            <AvatarFallback className="bg-azure-blue text-white text-xl">ML</AvatarFallback>
          </Avatar>
          <h2 className="text-xl text-gray-600 mb-2">ML Agent</h2>
          <h1 className="text-3xl font-semibold text-gray-800 text-center">How can I help you today?</h1>
        </div>

        <div className="w-full">
          <div className="space-y-6 mb-8">
            {messages.map((message, index) => (
              <div key={index} className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}>
                {message.role === "agent" && (
                  <div className="flex h-8 w-8 shrink-0 mr-2">
                    <Avatar>
                      <AvatarFallback className="bg-azure-blue text-white">ML</AvatarFallback>
                    </Avatar>
                  </div>
                )}
                <div
                  className={`max-w-[80%] rounded-lg p-4 ${
                    message.role === "user" ? "bg-azure-blue text-white" : "bg-gray-100 text-gray-800"
                  }`}
                >
                  <div className="whitespace-pre-line">{message.content}</div>
                  {message.role === "user" && message.showFiles && files.length > 0 && (
                    <div className="mt-2 bg-white/20 rounded p-2 text-white flex items-center gap-2">
                      <FileText className="h-4 w-4" />
                      <span>
                        {files[0].name} ({(files[0].size / 1024).toFixed(1)} KB)
                      </span>
                    </div>
                  )}
                </div>
              </div>
            ))}
            {readyToStart && (
              <div className="flex justify-center gap-4 mt-6 mb-6">
                <Button
                  variant="outline"
                  className="border-azure-blue text-azure-blue hover:bg-azure-blue-5"
                  onClick={() => {
                    setReadyToStart(false)
                    setUserMessage("")
                  }}
                >
                  Continue Chat
                </Button>
                <Button
                  className="bg-azure-blue text-white hover:bg-azure-dark-blue"
                  onClick={() => handleExecuteClick()}
                >
                  Start Agent Run
                </Button>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {!readyToStart && (
            <div className="flex items-center gap-2 bg-gray-50 p-2 rounded-lg border border-gray-200">
              <Input
                placeholder="Chat with your agent..."
                value={userMessage}
                onChange={(e) => setUserMessage(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault()
                    handleSendMessage()
                  }
                }}
                className="flex-1 border-0 focus-visible:ring-0 focus-visible:ring-offset-0"
              />
              <Button
                variant="ghost"
                size="icon"
                onClick={() => document.getElementById("datasets")?.click()}
                className="text-gray-500 hover:text-azure-blue hover:bg-transparent"
              >
                <Paperclip className="h-5 w-5" />
                <span className="sr-only">Attach file</span>
              </Button>
              <Input id="datasets" type="file" multiple onChange={handleFileChange} className="hidden" />
              <Button
                variant="ghost"
                size="icon"
                onClick = {() => handleSendMessage()}
                disabled={!userMessage.trim()}
                className="text-gray-500 hover:text-azure-blue hover:bg-transparent"
              >
                <Send className="h-5 w-5" />
                <span className="sr-only">Send message</span>
              </Button>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}


