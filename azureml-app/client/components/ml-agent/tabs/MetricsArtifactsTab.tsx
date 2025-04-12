import React from "react"
import { BarChart3 } from "lucide-react"
import { Badge } from "@/components/ui/badge"

interface MetricsArtifactsTabProps {
  currentMetricData: number[]
  getLatestMetric: () => any
  renderMetricsChart: (accuracies: number[]) => React.JSX.Element
  selectedVersion: string
  artifactRefs: React.MutableRefObject<{ [key: string]: HTMLDivElement | null }>
}

export function MetricsArtifactsTab({
  currentMetricData,
  getLatestMetric,
  renderMetricsChart,
  selectedVersion,
  artifactRefs,
}: MetricsArtifactsTabProps) {
  return (
    <div className="space-y-4">
      <div className="text-center p-6 text-gray-500 bg-azure-gray/20 rounded-md border border-azure-border flex flex-col items-center">
          <BarChart3 className="h-12 w-12 text-azure-blue/30 mb-2" />
          <p>No metrics available yet</p>
      </div>
      {/* {currentMetricData.length > 0 ? (
        <div
          className="border border-azure-border rounded-md p-4 transition-all duration-300 bg-white"
          ref={(el) => {
            const metric = getLatestMetric()
            if (metric) artifactRefs.current[metric.id] = el
          }}
        >
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <h3 className="text-sm font-medium text-gray-800">Model Performance</h3>
              <Badge className="bg-azure-gray text-gray-700 text-xs font-normal">{selectedVersion}</Badge>
            </div>
          </div>

          {/* Render the metrics chart with the current data */}
          {/* {renderMetricsChart(currentMetricData)}

          <div className="mt-2 text-xs text-center text-gray-500">
            Model accuracy comparison across versions
          </div>
        </div>
      ) : (
        <div className="text-center p-6 text-gray-500 bg-azure-gray/20 rounded-md border border-azure-border flex flex-col items-center">
          <BarChart3 className="h-12 w-12 text-azure-blue/30 mb-2" />
          <p>No metrics available yet</p>
        </div> 
      )}*/}
    </div>
  )
}
