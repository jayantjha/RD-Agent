"use client"

import React, { useState, useRef } from "react"
import { Bot, FileCode, BarChart3, Database, ArrowRight, CheckCircle2, XCircle, Loader2, Clock } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Collapsible, CollapsibleContent } from "@/components/ui/collapsible"

interface AgentProgressProps {
  agentActivities: any[],
  startAgent(): void,
  expandedActivities: Record<string, boolean>
  toggleActivityExpand: (activityId: string) => void
  handleArtifactLink: (artifactId: string | undefined, version: string) => void
  isStreaming: boolean
  currentActivityIndex: number
  activityStreamEndRef: React.RefObject<HTMLDivElement>
}

export function AgentProgress({
  agentActivities,
  expandedActivities,
  toggleActivityExpand,
  handleArtifactLink,
  isStreaming,
  startAgent,
  currentActivityIndex,
  activityStreamEndRef,
}: AgentProgressProps) {
  const getActivityIcon = (type: string, status: string, index: number) => {
    // If this is the current activity and we're streaming, show loading icon
    if (isStreaming && index === currentActivityIndex) {
      return <Loader2 className="h-4 w-4 animate-spin text-azure-blue" />
    }

    // Otherwise, show status icon
    if (status === "failed") {
      return <XCircle className="h-4 w-4 text-red-500" />
    }

    if (status === "done") {
      return <CheckCircle2 className="h-4 w-4 text-green-500" />
    }

    // Default icons based on type
    switch (type) {
      case "info":
        return <Bot className="h-4 w-4" />
      case "code":
        return <FileCode className="h-4 w-4" />
      case "model":
        return <Database className="h-4 w-4" />
      case "metrics":
        return <BarChart3 className="h-4 w-4" />
      case "complete":
        return <ArrowRight className="h-4 w-4" />
      default:
        return <Bot className="h-4 w-4" />
    }
  }

  return (
    <Card className="flex-1 flex flex-col border-azure-border shadow-sm">
      <CardHeader className="border-b border-azure-border bg-white pb-3">
        <CardTitle className="text-lg font-semibold text-azure-dark-blue">Agent Activity</CardTitle>
        <CardDescription className="text-gray-600">Live stream of the agent's actions and progress</CardDescription>
      </CardHeader>
      {/* <Button
        className="bg-azure-blue text-white hover:bg-azure-dark-blue"
        onClick={() => startAgent()}
    >
        Start Agent Run
    </Button> */}
      <CardContent className="flex-1 overflow-hidden bg-white p-4">
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-sm font-medium text-gray-700">Activity Stream</h3>
          <Badge className="bg-azure-gray text-gray-700 font-normal flex items-center gap-1">
            {isStreaming ? (
              <>
                <Loader2 className="h-3 w-3 animate-spin text-azure-blue" />
                <span>Streaming</span>
              </>
            ) : (
              <>
                <Clock className="h-3 w-3" />
                <span>Complete</span>
              </>
            )}
          </Badge>
        </div>
        <ScrollArea className="h-[calc(100vh-240px)]">
          <div className="space-y-3 p-1">
            {agentActivities.map((activity, index) => (
              <div
                key={activity.id}
                className={`group border border-azure-border rounded-md p-3 hover:bg-azure-gray/30 transition-colors cursor-pointer bg-white ${
                  isStreaming && index === currentActivityIndex ? "border-azure-blue" : ""
                }`}
                onClick={() => toggleActivityExpand(activity.id)}
              >
                <div className="flex gap-3">
                  <div
                    className={`flex h-6 w-6 shrink-0 items-center justify-center rounded-full ${
                      isStreaming && index === currentActivityIndex
                        ? "bg-azure-blue/10"
                        : activity.status === "done"
                        ? "bg-green-50"
                        : activity.status === "failed"
                        ? "bg-red-50"
                        : "bg-azure-gray"
                    }`}
                  >
                    {getActivityIcon(activity.type, activity.status, index)}
                  </div>
                  <div className="flex flex-col gap-1 flex-1">
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center gap-2">
                          <p className="text-sm font-medium text-gray-800">{activity.message}</p>
                          <Badge className="bg-azure-gray text-gray-700 text-xs font-normal">{activity.version}</Badge>

                          {/* Artifact link as text */}
                          {activity.artifactId && (
                            <span
                              className="text-xs text-azure-blue hover:underline cursor-pointer"
                              onClick={(e) => {
                                e.stopPropagation()
                                handleArtifactLink(activity.artifactId, activity.version)
                              }}
                            >
                              {activity.artifactName}
                            </span>
                          )}
                        </div>

                        {/* Short description always visible */}
                        <p className="text-sm text-gray-600 mt-1">{activity.shortDescription}</p>

                        {/* Expandable details */}
                        <Collapsible
                          open={expandedActivities[activity.id]}
                          onOpenChange={() => toggleActivityExpand(activity.id)}
                        >
                          <CollapsibleContent className="mt-2 pt-2 border-t border-azure-border">
                            <p className="text-sm text-gray-700">{activity.details}</p>
                          </CollapsibleContent>
                        </Collapsible>
                      </div>
                    </div>
                    <p className="text-xs text-gray-500 mt-1">{activity.timestamp.toLocaleTimeString()}</p>
                  </div>
                </div>
              </div>
            ))}

            {/* Streaming indicator at the end */}
            {isStreaming && (
              <div className="flex items-center justify-center py-2">
                <Loader2 className="h-5 w-5 text-azure-blue animate-spin mr-2" />
                <span className="text-sm text-gray-700">Agent is processing...</span>
              </div>
            )}

            {/* Invisible div for auto-scrolling */}
            <div ref={activityStreamEndRef} />
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  )
}
