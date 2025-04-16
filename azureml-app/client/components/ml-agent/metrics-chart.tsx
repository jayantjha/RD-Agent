import { forwardRef, Ref } from "react"

interface MetricsChartProps {
  accuracies: number[]
  chartRef: Ref<SVGSVGElement>
}

export const MetricsChart = forwardRef<SVGSVGElement, MetricsChartProps>(({ accuracies }, chartRef) => {
  const versions = Array.from({ length: accuracies.length }, (_, i) => `v${i}`)

  // Calculate chart dimensions - make it responsive but smaller
  const chartWidth = 100 // percentage
  const chartHeight = 140 // reduced height
  const padding = 25
  const availableWidth = chartHeight * 1.8 - padding * 2
  const availableHeight = chartHeight - padding * 2

  // Determine if we have any negative values and calculate ranges
  const minAccuracy = Math.min(...accuracies)
  const maxAccuracy = Math.max(...accuracies, 1)
  const hasNegativeValues = minAccuracy < 0
  
  // Calculate scales - adjust for negative values if needed
  const xStep = availableWidth / (versions.length - 1 || 1)
  const yRange = hasNegativeValues ? maxAccuracy - minAccuracy : maxAccuracy
  const yScale = availableHeight / yRange

  // Calculate zero position for y-axis when we have negative values
  const zeroY = hasNegativeValues ? chartHeight - padding + (minAccuracy * yScale) : chartHeight - padding

  // Generate points for the line
  let points = ""
  accuracies.forEach((acc, i) => {
    const x = padding + i * xStep
    // For negative values, we need to adjust where zero is
    const y = hasNegativeValues
      ? zeroY - acc * yScale
      : chartHeight - padding - acc * yScale
    points += `${x},${y} `
  })

  return (
    <div className="w-full overflow-hidden">
      <svg
        ref={chartRef}
        width="100%"
        height={chartHeight}
        className="w-full h-fix"
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

        {/* Zero line (x-axis) if we have negative values */}
        {hasNegativeValues && (
          <line
            x1={padding}
            y1={zeroY}
            x2={chartHeight * 1.8 - padding}
            y2={zeroY}
            stroke="#888"
            strokeWidth="1"
            strokeDasharray="2,2"
          />
        )}

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
        {/* <text x={padding - 8} y={padding} textAnchor="end" fontSize="4.5" fill="currentColor">
          {maxAccuracy.toFixed(2)}
        </text>
        
        <text x={padding - 8} y={chartHeight - padding} textAnchor="end" fontSize="4.5" fill="currentColor">
          {hasNegativeValues ? "0.00" : Math.min(...accuracies).toFixed(2)}
        </text> */}
        
        {/* Middle label or negative min value */}
        {hasNegativeValues ? (
          <>
            {/* <text x={padding - 8} y={zeroY} textAnchor="end" fontSize="4.5" fill="currentColor">
              {"0.00"}
            </text>
            <text x={padding - 8} y={chartHeight - padding} textAnchor="end" fontSize="4.5" fill="currentColor">
              {minAccuracy.toFixed(2)}
            </text> */}
          </>
        ) : (
          <text
            x={padding - 8}
            y={(chartHeight - padding + padding) / 2}
            textAnchor="end"
            fontSize="4.5"
            fill="currentColor"
          >
            {((Math.min(...accuracies) + maxAccuracy) / 2).toFixed(2)}
          </text>
        )}

        {/* Data points */}
        {accuracies.map((acc, i) => {
          const cy = hasNegativeValues
            ? zeroY - acc * yScale
            : chartHeight - padding - acc * yScale
          return (
            <circle
              key={i}
              cx={padding + i * xStep}
              cy={cy}
              r="3"
              className="fill-azure-blue"
            />
          )
        })}

        {/* Line connecting points */}
        <polyline points={points} fill="none" stroke="#0078d4" strokeWidth="2" />

        {/* Accuracy labels */}
        {accuracies.map((acc, i) => {
          const y = hasNegativeValues
            ? zeroY - acc * yScale
            : chartHeight - padding - acc * yScale
          return (
            <text
              key={i}
              x={padding + i * xStep}
              y={y - 8}
              textAnchor="middle"
              fontSize="4.5"
              fill="currentColor"
            >
              {acc.toFixed(2)}
            </text>
          )
        })}
      </svg>
    </div>
  )
})

MetricsChart.displayName = "MetricsChart"
