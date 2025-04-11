"use client"

import React, { useState } from "react"
import { ChevronUp, ChevronDown, Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"

interface AgentHeaderProps {
  taskDescription: string
  filesCount: number
  progress: number
  isStreaming: boolean
  stopAgent: () => void
}

export function AgentHeader({ taskDescription, filesCount, progress, isStreaming, stopAgent }: AgentHeaderProps) {
  const [expandedTaskDescription, setExpandedTaskDescription] = useState(false)

  return (
    <header className="border-b border-azure-border bg-white py-3 px-4 shadow-sm">
      <div className="container mx-auto">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="max-w-2xl">
            <h1 className="text-xl font-semibold text-azure-dark-blue">ML Agent</h1>
            <Collapsible>
              <div className="flex items-center gap-2">
                <p className={`text-sm text-gray-600 mt-1 ${!expandedTaskDescription ? "line-clamp-2" : ""}`}>
                  {taskDescription}
                </p>
                {taskDescription.length > 80 && (
                  <CollapsibleTrigger asChild>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-6 w-6 p-0 text-azure-blue flex-shrink-0"
                      onClick={() => setExpandedTaskDescription(!expandedTaskDescription)}
                    >
                      {expandedTaskDescription ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
                    </Button>
                  </CollapsibleTrigger>
                )}
              </div>
              <CollapsibleContent>
                <p className="text-sm text-gray-600 mt-1">{taskDescription}</p>
              </CollapsibleContent>
            </Collapsible>
          </div>

          <div className="flex items-center gap-4 hidden-view">
            <div className="flex items-center gap-2">
              <span className="text-sm text-gray-600">Datasets:</span>
              <Badge className="bg-azure-gray text-gray-700 font-normal">{filesCount} files</Badge>
            </div>

            <Separator orientation="vertical" className="h-6 bg-azure-border" />

            <div className="flex items-center gap-2">
              <span className="text-sm text-gray-600">Progress:</span>
              <div className="w-24 h-2 bg-azure-gray rounded-full overflow-hidden">
                <div className="h-full bg-azure-blue" style={{ width: `${progress}%` }}></div>
              </div>
              <span className="text-sm font-medium text-gray-700">{progress}%</span>
            </div>

            {isStreaming && (
              <Button
                variant="destructive"
                size="sm"
                onClick={stopAgent}
                className="bg-red-600 hover:bg-red-700 text-white"
              >
                Stop Agent
              </Button>
            )}
          </div>
        </div>
      </div>
    </header>
  )
}
