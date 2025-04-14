"use client"

import React, { useRef, useCallback, useState } from "react"
import { FileCode, BarChart3, Database, ChevronDown } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { ScrollArea } from "@/components/ui/scroll-area"
import { TabsList, TabsTrigger, Tabs, TabsContent } from "@/components/ui/tabs"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { CodeArtifactsTab } from "./tabs/CodeArtifactsTab"
import { MetricsArtifactsTab } from "./tabs/MetricsArtifactsTab"
import { ModelsArtifactsTab } from "./tabs/ModelsArtifactsTab"
import { useManifestData } from "@/lib/queries/useManifestData"
import { parseJSON } from "@/lib/utils" 

// Utility function to format file size to MB or KB depending on size
const formatSizeInMB = (sizeInBytes: number) => {
  if (sizeInBytes < 1024 * 1024) {
    // If size is less than 1MB, show in KB
    const sizeInKB = sizeInBytes / 1024;
    return sizeInKB.toFixed(2) + " KB";
  } else {
    // Otherwise show in MB
    const sizeInMB = sizeInBytes / (1024 * 1024);
    return sizeInMB.toFixed(2) + " MB";
  }
};

// Step type friendly names mapping
const STEP_FRIENDLY_NAMES: Record<string, string> = {
  "direct_exp_gen": "Experiment generation",
  "coding": "Developing code",
  "running": "Training and Generating Model",
  "feedback": "Evaluate the model"
};

// Status rendering helper
const getStatusText = (manifest: any) => {
  if (manifest?.feedback?.decision === undefined) return 'Running';
  return manifest?.feedback?.decision ? 'Successful' : 'Failed';
};

// Summary section component
const SummarySection = ({ manifest }: { manifest: any }) => {
  const stepName = manifest?.step_name || 'N/A';
  
  return (
    <div className="mb-4 p-3 bg-gray-50 rounded-md border border-azure-border">
      <h3 className="text-sm font-medium text-azure-dark-blue mb-2">Summary</h3>
      <div className="grid grid-cols-2 gap-4">
        <div>
          <p className="text-xs text-gray-500">Stage</p>
          <p className="text-sm font-medium">
            {STEP_FRIENDLY_NAMES[stepName] || stepName}
          </p>
        </div>
        <div>
          <p className="text-xs text-gray-500">Status</p>
          <p className="text-sm font-medium">
            {getStatusText(manifest)}
          </p>
        </div>
      </div>
    </div>
  );
};

// Experiment details section component
const ExperimentSection = ({ manifest }: { manifest: any }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  
  return (
    <div className="mb-4 p-3 bg-gray-50 rounded-md border border-azure-border">
      <div 
        className="flex items-center justify-between cursor-pointer"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <h3 className="text-sm font-medium text-azure-dark-blue mb-2">Experiment</h3>
        <ChevronDown 
          className={`h-4 w-4 transition-transform duration-200 ${isExpanded ? 'rotate-180' : ''}`}
        />
      </div>
      
      {isExpanded && (
        manifest && manifest.hypothesis ? (
          <div className="mt-2 space-y-4">
            <ExperimentField label="Hypothesis" value={manifest?.hypothesis?.hypothesis} />
            <ExperimentField label="Problem" value={manifest?.hypothesis?.problem} />
            <ExperimentField label="Evaluation" value={manifest?.feedback?.hypothesis_evaluation} />
            <ExperimentField label="Observation" value={manifest?.feedback?.observations} />
          </div>
        ) : (
          <div className="text-center p-6 text-gray-500 bg-azure-gray/20 rounded-md border border-azure-border flex flex-col items-center">
            <p>No Experiment details yet</p>
          </div>
        )
      )}
    </div>
  );
};

// Helper component for experiment field
const ExperimentField = ({ label, value }: { label: string, value?: string }) => (
  <div>
    <p className="text-xs text-gray-500">{label}</p>
    <p className="text-sm font-medium">{value || 'Unknown'}</p>
  </div>
);

interface AgentArtifactsProps {
  artifacts: {
    code: any[]
    models: any[]
    metrics: any[]
  }
  readyArtifacts: {
    code: string[]
    models: string[]
    metrics: string[]
  }
  selectedVersion: string
  setSelectedVersion: (version: string) => void
  availableVersions: string[]
  openSections: {
    code: boolean
    models: boolean
    metrics: boolean
  }
  toggleSection: (section: any) => void
  selectedCodeFile: string
  setSelectedCodeFile: (fileId: string) => void
  currentMetricData: number[]
  artifactRefs: React.MutableRefObject<{ [key: string]: HTMLDivElement | null }>
  getFilteredArtifacts: (type: "code" | "models" | "metrics") => any[]
  getLatestMetric: () => any
  renderMetricsChart: (accuracies: number[]) => React.JSX.Element
}

export function AgentArtifacts({
  artifacts,
  readyArtifacts,
  selectedVersion,
  setSelectedVersion,
  availableVersions,
  openSections,
  toggleSection,
  selectedCodeFile,
  setSelectedCodeFile,
  currentMetricData,
  artifactRefs,
  getFilteredArtifacts,
  getLatestMetric,
  renderMetricsChart,
}: AgentArtifactsProps) {
  const { data: manifestData, isLoading, error } = useManifestData(selectedVersion);
  const parsedManifest = manifestData ? parseJSON(manifestData) : null;
  const runId = selectedVersion || undefined;

  const handleVersionChange = useCallback((version: string) => {
    setSelectedVersion(version);
  }, [setSelectedVersion]);

  return (
    <Card className="flex-1 flex flex-col border-azure-border shadow-sm">
      <CardHeader className="border-b border-azure-border bg-white pb-3">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-lg font-semibold text-azure-dark-blue">ML Artifacts</CardTitle>
            <CardDescription className="text-gray-600">
              Code, models, and metrics generated by the agent
              {isLoading && " (Loading...)"}
              {error && " (Error loading data)"}
            </CardDescription>
          </div>

          {availableVersions.length > 0 && (
            <div className="flex items-center gap-2">
              <Label htmlFor="version-select" className="text-sm text-gray-600">
                Version:
              </Label>
              <Select value={selectedVersion} onValueChange={handleVersionChange}>
                <SelectTrigger id="version-select" className="w-24 border-azure-border text-gray-700">
                  <SelectValue placeholder="Select version" />
                </SelectTrigger>
                <SelectContent>
                  {availableVersions.map((version) => (
                    <SelectItem key={version} value={version}>
                      {version}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}
        </div>
      </CardHeader>
      <CardContent className="flex-1 overflow-auto bg-white p-4">
        <ScrollArea className="h-[calc(100vh-240px)]">
          <SummarySection manifest={parsedManifest} />
          <ExperimentSection manifest={parsedManifest} />
          
          <Tabs defaultValue="code" className="artifact-tabs">
            <TabsList className="mb-4 w-full grid grid-cols-3">
              <TabsTrigger value="code" className="artifact-tab">
                <FileCode className="h-4 w-4 mr-2" />
                Code
              </TabsTrigger>
              <TabsTrigger value="models" className="artifact-tab">
                <Database className="h-4 w-4 mr-2" />
                Models
              </TabsTrigger>
              <TabsTrigger value="metrics" className="artifact-tab">
                <BarChart3 className="h-4 w-4 mr-2" />
                Metrics
              </TabsTrigger>
            </TabsList>

            <TabsContent value="code" className="mt-0 p-0">
              <CodeArtifactsTab 
                getFilteredArtifacts={getFilteredArtifacts}
                selectedCodeFile={selectedCodeFile}
                setSelectedCodeFile={setSelectedCodeFile}
                artifactRefs={artifactRefs}
                manifestData={parsedManifest}
                runId={runId}
              />
            </TabsContent>

            <TabsContent value="models" className="mt-0 p-0">
              <ModelsArtifactsTab 
                getFilteredArtifacts={getFilteredArtifacts}
                artifactRefs={artifactRefs}
                manifestData={parsedManifest}
                formatSizeInMB={formatSizeInMB}
              />
            </TabsContent>

            <TabsContent value="metrics" className="mt-0 p-0">
              <MetricsArtifactsTab 
                currentMetricData={currentMetricData}
                getLatestMetric={getLatestMetric}
                renderMetricsChart={renderMetricsChart}
                selectedVersion={selectedVersion}
                artifactRefs={artifactRefs}
              />
            </TabsContent>
          </Tabs>
        </ScrollArea>
      </CardContent>
    </Card>
  );
}
