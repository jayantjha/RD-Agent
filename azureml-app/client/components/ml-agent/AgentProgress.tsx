"use client"

import React, { useEffect, useState } from "react"
import { Bot, FileCode, BarChart3, Database, Play, RefreshCcw, ArrowRight, CheckCircle2, XCircle, Loader2, Clock } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Collapsible, CollapsibleContent } from "@/components/ui/collapsible"
import { startEventStream, stopEventStream } from "@/lib/streamManager"
import { getApiUrl } from "@/lib/config/index"

interface AgentProgressProps {
  threadId: string,
  agentActivities: any[],
  setAgentActivities: React.Dispatch<React.SetStateAction<any[]>>,
  startAgent(): void,
  expandedActivities: Record<string, boolean>
  toggleActivityExpand: (activityId: string) => void
  handleArtifactLink: (artifactId: string | undefined, version: string) => void
  isStreaming: boolean
  setIsStreaming: React.Dispatch<React.SetStateAction<boolean>>
  currentActivityIndex: number
  setCurrentActivityIndex: React.Dispatch<React.SetStateAction<number>>
  activityStreamEndRef: React.RefObject<HTMLDivElement>
  setProgress: React.Dispatch<React.SetStateAction<number>>
  setReadyArtifacts: React.Dispatch<React.SetStateAction<{
    code: string[];
    models: string[];
    metrics: string[];
  }>>
  setAvailableLoopCounts: React.Dispatch<React.SetStateAction<number[]>>
  setSessionId: React.Dispatch<React.SetStateAction<string | undefined>>
  setSelectedVersion: React.Dispatch<React.SetStateAction<string>>
}

export function AgentProgress({
  threadId,
  agentActivities,
  setAgentActivities,
  expandedActivities,
  toggleActivityExpand,
  handleArtifactLink,
  isStreaming,
  setIsStreaming,
  startAgent,
  currentActivityIndex,
  setSelectedVersion,
  setCurrentActivityIndex,
  activityStreamEndRef,
  setProgress,
  setReadyArtifacts,
  setAvailableLoopCounts,
  setSessionId,
}: AgentProgressProps) {
  // Maintain a local state for tracking unique loop_count values
  const [loopCounts, setLoopCounts] = useState<Set<number>>(new Set());

  // Event mappings
  const event_mappings: Record<string, string> = {
    "DS_LOOP": "Starting ML agent for task",
    "DS_SCENARIO": "Loading and analyzing requirements and datasets",
    "RDLOOP": "Hypothesis generation and coding loop",
    "CODING": "Generating the pipeline code",
    "EXPERIMENT_GENERATION": "Generating experiment for the loop",
    "HYPOTHESIS_GENERATION": "Generating new hypothesis",
    "DATA_LOADING": "Code for loading data",
    "FEATURE_TASK": "Code for feature engineering",
    "MODEL_TASK": "Code for hypothesized model",
    "ENSEMBLE_TASK": "Generating ensemble model",
    "WORKFLOW_TASK": "Developing workflow",
    "FEEDBACK": "Gathering feedback for the loop",
    "RECORD": "Recording results",
    "DS_UPLOADED": "Generated main.py pipeline code",
    "PIPELINE_TASK": "Running pipeline",
    "RUNNING": "Executing and evaluating model"
  }

  const processAgentActivity = (data: any) => {
    
    // Ignore list for specific task types
    const ignoreList = ["FILE_MODIFIED", "MANIFEST_CREATED", "PIPELINE_TASK"];
    const childTasks = ["DS_UPLOADED"];
    
    // Skip processing if the task is in the ignore list
    if (ignoreList.includes(data.task)) {
      return;
    }
    
    // Check if this is a DS_SCENARIO with STARTED status to extract session ID
    if (data.task === "DS_SCENARIO" && data.status === "STARTED" && data.session_id) {
      setSessionId(data.session_id);
    }

    if (data.task === "DS_UPLOADED" && data.status === "COMPLETED" && data.loop_count !== undefined) {
      setSelectedVersion(data.loop_count.toString());
    }
    
    const activity = {
      id: data.id || `activity-${Date.now()}`,
      timestamp: new Date(data.createdAt * 1000),
      message: `${event_mappings[data.task] || data.task}`,
      shortDescription: data.message || "",
      details: data.description || "",
      type: data.type || "info",
      artifactId: data.artifactId,
      artifactName: data.artifactName,
      version: data.loop_count !== undefined ? `v${data.loop_count}` : undefined,
      sessionId: data.session_id || undefined,
      status: data.status.toLowerCase() || "done",
      loop_count: data.loop_count,
      indent: childTasks.includes(data.task) ? 1 : 0,
      highlight: data.task == "EXPERIMENT_GENERATION",
      previousEvents: Array<any>(),
    }

    // Update activities state
    setAgentActivities((prev) => {
      if (prev.length == 0)
        return [activity]

      let lastActivity = prev[prev.length - 1]
      let remaining = prev.slice(0, -1)
    
      if (lastActivity.message == activity.message) {
        // activity.indent = lastActivity.indent
        activity.previousEvents = [...lastActivity.previousEvents, lastActivity]

        return [...remaining, activity]
      }

      return [...prev, activity]
    })

    // Track unique loop_count values if present
    if (data.loop_count !== undefined) {
      setLoopCounts(prevLoopCounts => {
        const newLoopCounts = new Set(prevLoopCounts);
        newLoopCounts.add(data.loop_count);
        
        // Update parent component with the current set of loop counts
        const sortedLoopCounts = Array.from(newLoopCounts).sort((a, b) => a - b);
        setAvailableLoopCounts(sortedLoopCounts);
        
        return newLoopCounts;
      });
    }

    // Update current activity index
    setCurrentActivityIndex((prev) => prev + 1)
    
    // Calculate progress (approximation since we don't know total number)
    setProgress((prevProgress) => Math.min(100, prevProgress + 5))

    // Update ready artifacts when an artifact is created
    // if (activity.artifactId) {
    //   if (activity.artifactId.startsWith("code")) {
    //     setReadyArtifacts((prev) => ({
    //       ...prev,
    //       code: [...prev.code, activity.artifactId!],
    //     }))
    //   } else if (activity.artifactId.startsWith("model")) {
    //     setReadyArtifacts((prev) => ({
    //       ...prev,
    //       models: [...prev.models, activity.artifactId!],
    //     }))
    //   } else if (activity.artifactId.startsWith("metric")) {
    //     setReadyArtifacts((prev) => ({
    //       ...prev,
    //       metrics: [...prev.metrics, activity.artifactId!],
    //     }))
    //   }
    // }

    // // If the message indicates completion, end the streaming
    // if (activity.type === "complete") {
    //   setIsStreaming(false)
    //   setCurrentActivityIndex(-1)
    // }
  }

  // Handle errors from event stream
  const handleStreamError = (error: any) => {
    console.error('Stream connection error:', error);
    
    // Optionally add an error activity to show the user
    setAgentActivities((prev) => [
      ...prev,
      {
        id: `error-${Date.now()}`,
        timestamp: new Date(),
        message: "Connection error",
        shortDescription: "Lost connection to the agent",
        details: "The connection to the agent was lost. It may still be running in the background.",
        type: "info",
        status: "failed",
      }
    ]);
  }

  useEffect(() => {
    if (isStreaming) startEventStream(threadId, processAgentActivity, handleStreamError);
    return () => stopEventStream();
  }, [isStreaming]);

  // Auto-scroll to the bottom of the activity stream when new activities are added
  // useEffect(() => {
  //   if (agentActivities.length > 0 && isStreaming) {
  //     activityStreamEndRef.current?.scrollIntoView({ behavior: "smooth" })
  //   }
  // }, [agentActivities, isStreaming])

  const getActivityIcon = (type: string, status: string, index: number) => {
    // If this is the current activity and we're streaming, show loading icon
    // if (isStreaming && index === currentActivityIndex) {
    //   return <Loader2 className="h-4 w-4 animate-spin text-azure-blue" />
    // }

    // Otherwise, show status icon
    if (status === "failed") {
      return <XCircle className="h-4 w-4 text-red-500" />
    }

    if (status === "completed") {
      return <CheckCircle2 className="h-4 w-4 text-green-500" />
    }

    if (status === "started") {
      return <Play className="h-4 w-4 text-green-500" />
    }

    if (status === "in_progress") {
      return <RefreshCcw className="h-4 w-4 text-green-500" />
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
                style={{marginLeft: `${activity.indent * 20}px`, backgroundColor: `${activity.highlight ? "#e0e0e0" : "inherit"}`}}
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
                          {
                            activity.version && <Badge className="bg-azure-gray text-gray-700 text-xs font-normal">{activity.version}</Badge>
                          }

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
                            { activity.previousEvents.map((prevEvent: any) => (
                                <div key={prevEvent.id} className="flex items-center gap-2 mb-1">
                                  <p className="text-sm text-gray-600 mt-1">{prevEvent.shortDescription}</p>
                                  <p className="text-sm text-gray-600 mt-1">{prevEvent.details}</p>
                                </div>
                              ))
                            }

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
