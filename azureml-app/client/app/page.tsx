"use client"
import { Suspense } from "react"
import type React from "react"

// Create a component that uses useSearchParams
import { MLAgentContent } from "../components/ml-agent/MLAgentContent"

export default function MLAgentPage() {
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <MLAgentContent />
    </Suspense>
  )
}
