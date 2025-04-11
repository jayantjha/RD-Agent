import { forwardRef, Ref } from "react"

interface MetricsChartProps {
  accuracies: number[]
  chartRef: Ref<SVGSVGElement>
}

export const MetricsChart = forwardRef<SVGSVGElement, MetricsChartProps>(({ accuracies }, chartRef) => {
  const versions = Array.from({ length: accuracies.length }, (_, i) => `v${i + 1}`)

  // Calculate chart dimensions - make it responsive but smaller
  const chartWidth = 100 // percentage
  const chartHeight = 140 // reduced height
  const padding = 25
  const availableWidth = chartHeight * 1.8 - padding * 2
  const availableHeight = chartHeight - padding * 2

  // Calculate scales
  const xStep = availableWidth / (versions.length - 1 || 1)
  const maxAccuracy = Math.max(...accuracies, 1)
  const yScale = availableHeight / maxAccuracy

  // Generate points for the line
  let points = ""
  accuracies.forEach((acc, i) => {
    const x = padding + i * xStep
    const y = chartHeight - padding - acc * yScale
    points += `${x},${y} `
  })

  return (
    <div className="w-full overflow-hidden">
      <svg
        ref={chartRef}
        width="100%"
        height={chartHeight}
        className="w-full h-auto"
        viewBox={`0 0 ${chartHeight * 1.8} ${chartHeight}`}
        preserveAspectRatio="xMidYMid meet"
      >
        {/* X and Y axes */}
        <line
          x1={padding}
          y1={chartHeight - padding}
          x2={chartHeight * 1.8 - padding}
          y2={chartHeight - padding}
          stroke="#888"
          strokeWidth="1"
        />
        <line x1={padding} y1={padding} x2={padding} y2={chartHeight - padding} stroke="#888" strokeWidth="1" />

        {/* X axis labels */}
        {versions.map((version, i) => (
          <text
            key={version}
            x={padding + i * xStep}
            y={chartHeight - padding + 15}
            textAnchor="middle"
            fontSize="4.5"
            fill="currentColor"
          >
            {version}
          </text>
        ))}

        {/* Y axis labels */}
        <text x={padding - 8} y={padding} textAnchor="end" fontSize="4.5" fill="currentColor">
          1.0
        </text>
        <text x={padding - 8} y={chartHeight - padding} textAnchor="end" fontSize="4.5" fill="currentColor">
          0.0
        </text>
        <text
          x={padding - 8}
          y={(chartHeight - padding + padding) / 2}
          textAnchor="end"
          fontSize="4.5"
          fill="currentColor"
        >
          0.5
        </text>

        {/* Data points */}
        {accuracies.map((acc, i) => (
          <circle
            key={i}
            cx={padding + i * xStep}
            cy={chartHeight - padding - acc * yScale}
            r="3"
            className="fill-azure-blue"
          />
        ))}

        {/* Line connecting points */}
        <polyline points={points} fill="none" stroke="#0078d4" strokeWidth="2" />

        {/* Accuracy labels */}
        {accuracies.map((acc, i) => (
          <text
            key={i}
            x={padding + i * xStep}
            y={chartHeight - padding - acc * yScale - 8}
            textAnchor="middle"
            fontSize="4.5"
            fill="currentColor"
          >
            {acc.toFixed(2)}
          </text>
        ))}
      </svg>
    </div>
  )
})

MetricsChart.displayName = "MetricsChart"
