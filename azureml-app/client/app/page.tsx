"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import {
  Bot,
  FileCode,
  BarChart3,
  Database,
  Clock,
  ArrowRight,
  ChevronRight,
  ChevronDown,
  FileText,
  Loader2,
  Rocket,
  Shield,
  CheckCircle2,
  XCircle,
  Paperclip,
  Send,
  ChevronUp,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Separator } from "@/components/ui/separator"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { TabsList, TabsTrigger, Tabs, TabsContent } from "@/components/ui/tabs"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"

export default function MLAgentPage() {
  const [userMessage, setUserMessage] = useState("")
  const [expandedTaskDescription, setExpandedTaskDescription] = useState(false)
  const [files, setFiles] = useState<File[]>([])
  const [isAgentRunning, setIsAgentRunning] = useState(false)
  const [selectedCodeFile, setSelectedCodeFile] = useState("")
  const [selectedVersion, setSelectedVersion] = useState("")
  const [expandedActivities, setExpandedActivities] = useState<Record<string, boolean>>({})
  const [availableVersions, setAvailableVersions] = useState<string[]>([])
  const [taskDescription, setTaskDescription] = useState("")
  const [progress, setProgress] = useState(0)
  const [openSections, setOpenSections] = useState({
    code: true,
    models: true,
    metrics: true,
  })

  const [highlightedArtifact, setHighlightedArtifact] = useState<string | null>(null)
  const artifactRefs = useRef<{ [key: string]: HTMLDivElement | null }>({})
  const activityStreamEndRef = useRef<HTMLDivElement>(null)
  const [isStreaming, setIsStreaming] = useState(false)
  const chartRef = useRef<SVGSVGElement>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  // Chat messages state
  const [messages, setMessages] = useState<
    {
      role: "user" | "agent"
      content: string
      timestamp: Date
      showFiles?: boolean
    }[]
  >([
    {
      role: "agent",
      content:
        "Hello! I'm your ML assistant. I can help you build and optimize machine learning models. What kind of ML task would you like to work on today?",
      timestamp: new Date(),
    },
  ])

  // Add these new state variables after the existing state declarations:
  const [isSimulating, setIsSimulating] = useState(false)
  const [simulationStep, setSimulationStep] = useState(0)
  const [readyToStart, setReadyToStart] = useState(false)

  // Add a state to track which artifacts are ready to display
  const [readyArtifacts, setReadyArtifacts] = useState<{
    code: string[]
    models: string[]
    metrics: string[]
  }>({
    code: [],
    models: [],
    metrics: [],
  })

  // Mock artifacts data with 4 versions
  const [artifacts, setArtifacts] = useState({
    code: [
      {
        id: "code-1",
        name: "main.py",
        version: "v1",
        content:
          "# ML Pipeline v1\nimport pandas as pd\nimport numpy as np\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.model_selection import train_test_split\n\ndef load_data(file_path):\n    # Load the dataset\n    df = pd.read_csv(file_path)\n    return df\n\ndef preprocess(df):\n    # Handle missing values\n    df = df.fillna(0)\n    \n    # Normalize numerical features\n    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns\n    df[numerical_cols] = (df[numerical_cols] - df[numerical_cols].mean()) / df[numerical_cols].std()\n    \n    return df\n\ndef train_model(X, y):\n    # Split data\n    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n    \n    # Train model\n    model = RandomForestClassifier(n_estimators=100)\n    model.fit(X_train, y_train)\n    \n    # Evaluate\n    accuracy = model.score(X_test, y_test)\n    \n    return model, accuracy\n\n# Main pipeline execution\ndef run_pipeline(data_path, target_col):\n    # Load data\n    df = load_data(data_path)\n    \n    # Preprocess\n    df_processed = preprocess(df)\n    \n    # Split features and target\n    X = df_processed.drop(target_col, axis=1)\n    y = df_processed[target_col]\n    \n    # Train model\n    model, accuracy = train_model(X, y)\n    \n    print(f\"Model trained with accuracy: {accuracy:.4f}\")\n    return model, accuracy",
      },
      {
        id: "code-2",
        name: "main.py",
        version: "v2",
        content:
          "# ML Pipeline v2\nimport pandas as pd\nimport numpy as np\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.impute import KNNImputer\n\ndef load_data(file_path):\n    # Load the dataset\n    df = pd.read_csv(file_path)\n    return df\n\ndef preprocess(df):\n    # Handle missing values with KNN imputation\n    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns\n    categorical_cols = df.select_dtypes(include=['object']).columns\n    \n    # Handle categorical missing values\n    for col in categorical_cols:\n        df[col] = df[col].fillna(df[col].mode()[0])\n    \n    # Handle numerical missing values with KNN\n    if len(numerical_cols) > 0:\n        imputer = KNNImputer(n_neighbors=5)\n        df[numerical_cols] = imputer.fit_transform(df[numerical_cols])\n    \n    # Normalize numerical features\n    df[numerical_cols] = (df[numerical_cols] - df[numerical_cols].mean()) / df[numerical_cols].std()\n    \n    return df\n\ndef engineer_features(df):\n    # Create interaction features\n    for col1 in df.columns[:3]:\n        for col2 in df.columns[3:6]:\n            df[f'{col1}_{col2}_interaction'] = df[col1] * df[col2]\n    \n    # Create polynomial features\n    for col in df.select_dtypes(include=['float64', 'int64']).columns:\n        df[f'{col}_squared'] = df[col] ** 2\n    \n    return df\n\ndef train_model(X, y):\n    # Split data\n    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n    \n    # Train model with optimized hyperparameters\n    model = RandomForestClassifier(n_estimators=200, max_depth=15)\n    model.fit(X_train, y_train)\n    \n    # Evaluate\n    accuracy = model.score(X_test, y_test)\n    \n    return model, accuracy\n\n# Main pipeline execution\ndef run_pipeline(data_path, target_col):\n    # Load data\n    df = load_data(data_path)\n    \n    # Preprocess\n    df_processed = preprocess(df)\n    \n    # Feature engineering\n    df_engineered = engineer_features(df_processed)\n    \n    # Split features and target\n    X = df_engineered.drop(target_col, axis=1)\n    y = df_engineered[target_col]\n    \n    # Train model\n    model, accuracy = train_model(X, y)\n    \n    print(f\"Model trained with accuracy: {accuracy:.4f}\")\n    return model, accuracy",
      },
      {
        id: "code-3",
        name: "main.py",
        version: "v3",
        content:
          "# ML Pipeline v3\nimport pandas as pd\nimport numpy as np\nfrom sklearn.ensemble import GradientBoostingClassifier\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.impute import KNNImputer\nfrom sklearn.decomposition import PCA\n\ndef load_data(file_path):\n    # Load the dataset\n    df = pd.read_csv(file_path)\n    return df\n\ndef preprocess(df):\n    # Handle missing values with KNN imputation\n    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns\n    categorical_cols = df.select_dtypes(include=['object']).columns\n    \n    # Handle categorical missing values\n    for col in categorical_cols:\n        df[col] = df[col].fillna(df[col].mode()[0])\n    \n    # Handle numerical missing values with KNN\n    if len(numerical_cols) > 0:\n        imputer = KNNImputer(n_neighbors=5)\n        df[numerical_cols] = imputer.fit_transform(df[numerical_cols])\n    \n    # Normalize numerical features\n    df[numerical_cols] = (df[numerical_cols] - df[numerical_cols].mean()) / df[numerical_cols].std()\n    \n    return df\n\ndef engineer_features_advanced(df):\n    # Create interaction features\n    for col1 in df.columns[:3]:\n        for col2 in df.columns[3:6]:\n            df[f'{col1}_{col2}_interaction'] = df[col1] * df[col2]\n    \n    # Create polynomial features\n    for col in df.select_dtypes(include=['float64', 'int64']).columns[:5]:\n        df[f'{col}_squared'] = df[col] ** 2\n        df[f'{col}_cubed'] = df[col] ** 3\n    \n    # Apply PCA for dimensionality reduction\n    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns\n    if len(numerical_cols) > 10:\n        pca = PCA(n_components=10)\n        pca_result = pca.fit_transform(df[numerical_cols])\n        for i in range(10):\n            df[f'pca_{i+1}'] = pca_result[:, i]\n    \n    return df\n\ndef train_model(X, y):\n    # Split data\n    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n    \n    # Train model\n    model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1)\n    model.fit(X_train, y_train)\n    \n    # Evaluate\n    accuracy = model.score(X_test, y_test)\n    \n    return model, accuracy\n\n# Main pipeline execution\ndef run_pipeline(data_path, target_col):\n    # Load data\n    df = load_data(data_path)\n    \n    # Preprocess\n    df_processed = preprocess(df)\n    \n    # Advanced feature engineering\n    df_engineered = engineer_features_advanced(df_processed)\n    \n    # Split features and target\n    X = df_engineered.drop(target_col, axis=1)\n    y = df_engineered[target_col]\n    \n    # Train model\n    model, accuracy = train_model(X, y)\n    \n    print(f\"Model trained with accuracy: {accuracy:.4f}\")\n    return model, accuracy",
      },
      {
        id: "code-4",
        name: "main.py",
        version: "v4",
        content:
          "# ML Pipeline v4\nimport pandas as pd\nimport numpy as np\nfrom sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.impute import KNNImputer\nfrom sklearn.decomposition import PCA\nfrom sklearn.feature_selection import SelectFromModel, RFE\n\ndef load_data(file_path):\n    # Load the dataset\n    df = pd.read_csv(file_path)\n    return df\n\ndef preprocess(df):\n    # Handle missing values with KNN imputation\n    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns\n    categorical_cols = df.select_dtypes(include=['object']).columns\n    \n    # Handle categorical missing values\n    for col in categorical_cols:\n        df[col] = df[col].fillna(df[col].mode()[0])\n    \n    # Handle numerical missing values with KNN\n    if len(numerical_cols) > 0:\n        imputer = KNNImputer(n_neighbors=5)\n        df[numerical_cols] = imputer.fit_transform(df[numerical_cols])\n    \n    # Normalize numerical features\n    df[numerical_cols] = (df[numerical_cols] - df[numerical_cols].mean()) / df[numerical_cols].std()\n    \n    return df\n\ndef engineer_features_advanced(df):\n    # Create interaction features\n    for col1 in df.columns[:3]:\n        for col2 in df.columns[3:6]:\n            df[f'{col1}_{col2}_interaction'] = df[col1] * df[col2]\n    \n    # Create polynomial features\n    for col in df.select_dtypes(include=['float64', 'int64']).columns[:5]:\n        df[f'{col}_squared'] = df[col] ** 2\n        df[f'{col}_cubed'] = df[col] ** 3\n    \n    # Apply PCA for dimensionality reduction\n    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns\n    if len(numerical_cols) > 10:\n        pca = PCA(n_components=10)\n        pca_result = pca.fit_transform(df[numerical_cols])\n        for i in range(10):\n            df[f'pca_{i+1}'] = pca_result[:, i]\n    \n    return df\n\ndef select_features(X, y):\n    # Method 1: Feature importance from Random Forest\n    rf = RandomForestClassifier(n_estimators=100)\n    rf.fit(X, y)\n    \n    # Select top features based on importance\n    sfm = SelectFromModel(rf, threshold='median')\n    X_selected = sfm.fit_transform(X, y)\n    \n    # Method 2: Recursive Feature Elimination\n    rfe = RFE(estimator=RandomForestClassifier(n_estimators=50), n_features_to_select=20)\n    X_rfe = rfe.fit_transform(X, y)\n    \n    # Get feature names\n    selected_features = X.columns[sfm.get_support()]\n    rfe_features = X.columns[rfe.get_support()]\n    \n    # Combine both methods\n    final_features = list(set(selected_features) | set(rfe_features))\n    \n    return X[final_features], final_features\n\ndef train_ensemble(X, y):\n    # Split data\n    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n    \n    # Create base models\n    rf = RandomForestClassifier(n_estimators=200, max_depth=20)\n    gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1)\n    lr = LogisticRegression(C=0.1)\n    \n    # Create and train ensemble\n    ensemble = VotingClassifier(\n        estimators=[('rf', rf), ('gb', gb), ('lr', lr)],\n        voting='soft'\n    )\n    ensemble.fit(X_train, y_train)\n    \n    # Evaluate\n    accuracy = ensemble.score(X_test, y_test)\n    \n    return ensemble, accuracy\n\n# Main pipeline execution\ndef run_pipeline(data_path, target_col):\n    # Load data\n    df = load_data(data_path)\n    \n    # Preprocess\n    df_processed = preprocess(df)\n    \n    # Advanced feature engineering\n    df_engineered = engineer_features_advanced(df_processed)\n    \n    # Split features and target\n    X = df_engineered.drop(target_col, axis=1)\n    y = df_engineered[target_col]\n    \n    # Feature selection\n    X_selected, selected_features = select_features(X, y)\n    print(f\"Selected {len(selected_features)} features\")\n    \n    # Train ensemble model\n    model, accuracy = train_ensemble(X_selected, y)\n    \n    print(f\"Ensemble model trained with accuracy: {accuracy:.4f}\")\n    return model, accuracy",
      },
    ],
    models: [
      { id: "model-1", name: "random_forest_v1.pkl", version: "v1", size: "2.4 MB", accuracy: 0.87 },
      { id: "model-2", name: "random_forest_v2.pkl", version: "v2", size: "3.1 MB", accuracy: 0.92 },
      { id: "model-3", name: "gradient_boost_v3.pkl", version: "v3", size: "4.2 MB", accuracy: 0.94 },
      { id: "model-4", name: "ensemble_model_v4.pkl", version: "v4", size: "8.5 MB", accuracy: 0.96 },
    ],
    metrics: [
      { id: "metric-1", name: "model_metrics", version: "v1", type: "line_chart", accuracies: [0.87] },
      { id: "metric-2", name: "model_metrics", version: "v2", type: "line_chart", accuracies: [0.87, 0.92] },
      { id: "metric-3", name: "model_metrics", version: "v3", type: "line_chart", accuracies: [0.87, 0.92, 0.94] },
      {
        id: "metric-4",
        name: "model_metrics",
        version: "v4",
        type: "line_chart",
        accuracies: [0.87, 0.92, 0.94, 0.96],
      },
    ],
  })

  const [agentActivities, setAgentActivities] = useState<any[]>([])
  const [currentMetricData, setCurrentMetricData] = useState<number[]>([])

  // Auto-scroll to the bottom of messages when new messages are added
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  // Scroll to highlighted artifact
  useEffect(() => {
    if (highlightedArtifact && artifactRefs.current[highlightedArtifact]) {
      artifactRefs.current[highlightedArtifact]?.scrollIntoView({ behavior: "smooth", block: "center" })

      // Flash effect
      const element = artifactRefs.current[highlightedArtifact]
      element?.classList.add("bg-azure-blue-10")
      setTimeout(() => {
        element?.classList.remove("bg-azure-blue-10")
      }, 2000)
    }
  }, [highlightedArtifact])

  // Auto-scroll to the bottom of the activity stream when new activities are added
  useEffect(() => {
    if (agentActivities.length > 0 && isStreaming) {
      activityStreamEndRef.current?.scrollIntoView({ behavior: "smooth" })
    }
  }, [agentActivities, isStreaming])

  // Set initial code file when artifacts change or version changes
  useEffect(() => {
    const versionCodeFiles = artifacts.code.filter(
      (file) => file.version === selectedVersion && readyArtifacts.code.includes(file.id),
    )
    if (versionCodeFiles.length > 0) {
      setSelectedCodeFile(versionCodeFiles[0].id)
    }
  }, [artifacts.code, selectedVersion, readyArtifacts.code])

  // Update available versions when ready artifacts change
  useEffect(() => {
    const versions = new Set<string>()

    // Add versions from ready artifacts
    readyArtifacts.code.forEach((id) => {
      const artifact = artifacts.code.find((a) => a.id === id)
      if (artifact) versions.add(artifact.version)
    })

    readyArtifacts.models.forEach((id) => {
      const artifact = artifacts.models.find((a) => a.id === id)
      if (artifact) versions.add(artifact.version)
    })

    readyArtifacts.metrics.forEach((id) => {
      const artifact = artifacts.metrics.find((a) => a.id === id)
      if (artifact) versions.add(artifact.version)
    })

    const sortedVersions = Array.from(versions).sort((a, b) => {
      // Sort by version number (v1, v2, v3, v4)
      return a.localeCompare(b, undefined, { numeric: true })
    })

    setAvailableVersions(sortedVersions)

    // Always set selected version to the latest available
    if (sortedVersions.length > 0) {
      setSelectedVersion(sortedVersions[sortedVersions.length - 1])
    }
  }, [readyArtifacts, artifacts])

  // Update metrics data when a new metric is added
  useEffect(() => {
    const latestMetric = getLatestMetric()
    if (latestMetric) {
      setCurrentMetricData(latestMetric.accuracies)
    }
  }, [readyArtifacts.metrics])

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFiles(Array.from(e.target.files))
    }
  }

  // Add a state to store the interval ID
  const [agentIntervalId, setAgentIntervalId] = useState<number | null>(null)
  const [currentActivityIndex, setCurrentActivityIndex] = useState<number>(-1)

  const handleSendMessage = () => {
    if (!userMessage.trim() && !readyToStart) return

    // If we're ready to start and the user clicks the button, add a confirmation message
    if (readyToStart && !userMessage) {
      const confirmMessage = {
        role: "user" as const,
        content: "Yes, let's start building the model!",
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, confirmMessage])

      // Immediately hide the buttons by setting readyToStart to false
      setReadyToStart(false)

      // Start the agent after a short delay
      setTimeout(() => {
        startAgent()
      }, 1000)

      return
    }

    // Add user message to chat
    const newUserMessage = {
      role: "user" as const,
      content: userMessage,
      timestamp: new Date(),
    }

    setMessages((prev) => [...prev, newUserMessage])
    setUserMessage("")

    // If this is about starting an ML task, capture it as the task description
    if (messages.length === 1) {
      setTaskDescription(userMessage)
    }
  }

  // Function to simulate a conversation between agent and user
  const simulateConversation = () => {
    if (isSimulating) return

    setIsSimulating(true)

    const conversation = [
      {
        role: "user",
        content: "I need to build a model to predict customer churn for my telecom company.",
        timestamp: new Date(),
      },
      {
        role: "agent",
        content:
          "Thanks for sharing your ML task! I can definitely help with customer churn prediction. Could you tell me more about your specific requirements and the data you have?",
        timestamp: new Date(),
      },
      {
        role: "user",
        content:
          "We have customer data including usage patterns, billing information, customer service interactions, and whether they churned or not. We want to identify customers at risk of leaving.",
        timestamp: new Date(),
      },
      {
        role: "agent",
        content:
          "That's great information. For churn prediction, we typically use these evaluation metrics:\n\n- Accuracy: Overall correctness of predictions\n- Precision: Proportion of positive identifications that were actually correct\n- Recall: Proportion of actual positives that were identified correctly\n- F1 Score: Harmonic mean of precision and recall\n- ROC-AUC: Area under the ROC curve\n\nWhich metrics would be most important for your business case?",
        timestamp: new Date(),
      },
      {
        role: "user",
        content:
          "Recall is most important since we don't want to miss customers who might churn. But we also care about precision because our retention program has limited resources.",
        timestamp: new Date(),
      },
      {
        role: "agent",
        content:
          "Perfect! Optimizing for both recall and precision makes sense for your use case. Now I'll need your dataset to get started.\n\nPlease upload your customer data file. It should include the features you mentioned (usage patterns, billing info, customer service interactions) and the churn label.",
        timestamp: new Date(),
      },
      {
        role: "user",
        content: "Here's our customer dataset with 3 years of historical data.",
        timestamp: new Date(),
        showFiles: true,
      },
      {
        role: "agent",
        content:
          "Thanks for uploading the dataset! I can see it contains customer data with various features.\n\nWhich column in your dataset indicates whether a customer churned or not? This will be our target variable for prediction.",
        timestamp: new Date(),
      },
      {
        role: "user",
        content:
          "The target column is called 'Churn' - it's a binary variable where 1 means the customer churned and 0 means they stayed.",
        timestamp: new Date(),
      },
      {
        role: "agent",
        content:
          "Perfect! I now have all the information I need to start building your churn prediction model:\n\n- Task: Customer churn prediction for telecom company\n- Dataset: Customer data with usage patterns, billing info, and service interactions\n- Target column: 'Churn' (binary: 1=churned, 0=stayed)\n- Key metrics: Recall and precision\n\nI'll build multiple models, optimize them for your specific needs, and provide you with the best solution. Are you ready to start?",
        timestamp: new Date(),
      },
    ]

    // Create a mock file for the dataset
    const mockFile = new File([""], "telecom_customer_data.csv", { type: "text/csv" })

    // Stream messages with delay
    let step = 0
    const streamMessages = () => {
      if (step < conversation.length) {
        const message = conversation[step]

        // Add message to chat
        setMessages((prev) => [...prev, message])

        // If this is the file upload message, add the file
        if (message.showFiles) {
          setFiles([mockFile])
        }

        // If this is the task description, save it
        if (step === 0) {
          setTaskDescription(message.content)
        }

        // If this is the last message (summary), set ready to start
        if (step === conversation.length - 1) {
          setTimeout(() => {
            setReadyToStart(true)
            setIsSimulating(false)
          }, 500)
        }

        // Update simulation step
        setSimulationStep(step)

        // Schedule next message
        step++
        setTimeout(streamMessages, 1500)
      }
    }

    // Start streaming messages
    streamMessages()
  }

  // Start simulation when component mounts
  useEffect(() => {
    // Start with a slight delay to allow the UI to render
    const timer = setTimeout(() => {
      simulateConversation()
    }, 1000)

    return () => clearTimeout(timer)
  }, [])

  // Update the startAgent function to store the interval ID
  const startAgent = () => {
    if (!readyToStart) return

    setIsAgentRunning(true)
    setIsStreaming(true)
    setAgentActivities([])
    setCurrentActivityIndex(0)
    setCurrentMetricData([])

    // Reset ready artifacts
    setReadyArtifacts({
      code: [],
      models: [],
      metrics: [],
    })

    // Reset available versions
    setAvailableVersions([])
    setSelectedVersion("")

    // Simulate agent activities and store the interval ID
    const intervalId = simulateAgentActivities()
    setAgentIntervalId(intervalId)
  }

  const simulateAgentActivities = () => {
    const activities = [
      {
        message: "Starting ML agent for task",
        shortDescription: "Initializing environment and loading dependencies",
        details: `Starting ML agent for the following task: "${taskDescription}". Initializing environment and loading dependencies.`,
        type: "info",
        version: "v1",
      },
      {
        message: "Loading and analyzing datasets",
        shortDescription: "Found 3 numerical features and 2 categorical features with 5% missing values",
        details:
          "Loading datasets into memory. Performing initial analysis: checking for missing values, data types, and basic statistics. Found 3 numerical features and 2 categorical features with 5% missing values.",
        type: "info",
        version: "v1",
      },
      {
        message: "Generating new hypothesis to improve model",
        shortDescription: "Basic preprocessing with Random Forest classifier",
        details:
          "Generating hypothesis: Basic preprocessing with simple imputation and normalization, followed by Random Forest classifier with default parameters should provide a good baseline.",
        type: "info",
        version: "v1",
      },
      {
        message: "Generating main.py pipeline code",
        shortDescription: "Created pipeline with basic preprocessing and Random Forest model",
        details:
          "Generated main.py with complete pipeline code including data loading, preprocessing, and model training with Random Forest classifier.",
        type: "code",
        artifactId: "code-1",
        artifactName: "main.py",
        version: "v1",
      },
      {
        message: "Running pipeline",
        shortDescription: "Executing pipeline with basic preprocessing and Random Forest model",
        details:
          "Running pipeline with basic preprocessing and Random Forest model training. Using default parameters and 80/20 train/test split.",
        type: "info",
        version: "v1",
      },
      {
        message: "Created Random Forest model (v1)",
        shortDescription: "Model has 100 trees with 87% accuracy on validation data",
        details:
          "Successfully trained initial Random Forest model. Model has 100 trees and achieved 87% accuracy on validation data.",
        type: "model",
        artifactId: "model-1",
        artifactName: "random_forest_v1.pkl",
        version: "v1",
      },
      {
        message: "Generated model metrics",
        shortDescription: "Created visualization comparing model performance metrics",
        details:
          "Created visualization comparing model performance metrics including accuracy, precision, recall, and F1 score.",
        type: "metrics",
        artifactId: "metric-1",
        artifactName: "model_metrics",
        version: "v1",
      },
      {
        message: "Generating new hypothesis to improve model",
        shortDescription: "Enhanced preprocessing and feature engineering",
        details:
          "Generating hypothesis: KNN imputation for missing values and feature engineering with interaction terms should improve model performance. Optimizing Random Forest hyperparameters will further enhance results.",
        type: "info",
        version: "v2",
      },
      {
        message: "Generating main.py pipeline code v2",
        shortDescription: "Updated pipeline with KNN imputation and feature engineering",
        details:
          "Generated main.py v2 with improved pipeline including KNN imputation for missing values, feature engineering, and optimized Random Forest model.",
        type: "code",
        artifactId: "code-2",
        artifactName: "main.py",
        version: "v2",
      },
      {
        message: "Running pipeline",
        shortDescription: "Executing pipeline with enhanced preprocessing and feature engineering",
        details:
          "Running pipeline with KNN imputation for missing values, feature engineering, and optimized Random Forest model.",
        type: "info",
        version: "v2",
      },
      {
        message: "Created improved Random Forest model (v2)",
        shortDescription: "Optimized model with 200 trees, max_depth=15, 92% accuracy",
        details:
          "Successfully trained optimized Random Forest model with 200 trees, max_depth=15. Model achieved 92% accuracy on validation data.",
        type: "model",
        artifactId: "model-2",
        artifactName: "random_forest_v2.pkl",
        version: "v2",
      },
      {
        message: "Updated model metrics",
        shortDescription: "Added v2 model performance to metrics visualization",
        details: "Updated visualization with v2 model performance metrics showing improvement over baseline.",
        type: "metrics",
        artifactId: "metric-2",
        artifactName: "model_metrics",
        version: "v2",
      },
      {
        message: "Generating new hypothesis to improve model",
        shortDescription: "Testing gradient boosting algorithms for better performance",
        details:
          "Generating hypothesis: Gradient Boosting classifier with advanced feature engineering including polynomial features and PCA should capture more complex patterns in the data.",
        type: "info",
        version: "v3",
      },
      {
        message: "Generating main.py pipeline code v3",
        shortDescription: "Updated pipeline with advanced features and Gradient Boosting",
        details:
          "Generated main.py v3 with advanced feature engineering including polynomial features, PCA, and Gradient Boosting classifier.",
        type: "code",
        artifactId: "code-3",
        artifactName: "main.py",
        version: "v3",
      },
      {
        message: "Running pipeline",
        shortDescription: "Executing pipeline with advanced features and Gradient Boosting",
        details:
          "Running pipeline with advanced feature engineering and Gradient Boosting classifier to improve model performance.",
        type: "info",
        version: "v3",
      },
      {
        message: "Created Gradient Boosting model (v3)",
        shortDescription: "Model achieved 94% accuracy, 2% improvement over v2",
        details:
          "Successfully trained Gradient Boosting model with 200 trees and learning rate of 0.1. Model achieved 94% accuracy on validation data.",
        type: "model",
        artifactId: "model-3",
        artifactName: "gradient_boost_v3.pkl",
        version: "v3",
      },
      {
        message: "Updated model metrics",
        shortDescription: "Added v3 model performance to metrics visualization",
        details: "Updated visualization with v3 model performance metrics showing continued improvement.",
        type: "metrics",
        artifactId: "metric-3",
        artifactName: "model_metrics",
        version: "v3",
      },
      {
        message: "Generating new hypothesis to improve model",
        shortDescription: "Feature selection and ensemble modeling",
        details:
          "Generating hypothesis: Feature selection to reduce dimensionality combined with ensemble modeling that leverages multiple algorithms should provide the best performance.",
        type: "info",
        version: "v4",
      },
      {
        message: "Generating main.py pipeline code v4",
        shortDescription: "Created final pipeline with feature selection and ensemble model",
        details:
          "Generated main.py v4 with complete pipeline including feature selection and ensemble model combining Random Forest, Gradient Boosting, and Logistic Regression.",
        type: "code",
        artifactId: "code-4",
        artifactName: "main.py",
        version: "v4",
      },
      {
        message: "Running pipeline",
        shortDescription: "Executing pipeline with feature selection and ensemble modeling",
        details:
          "Running pipeline with feature selection techniques and ensemble modeling combining multiple algorithms.",
        type: "info",
        version: "v4",
      },
      {
        message: "Created Ensemble model (v4)",
        shortDescription: "Final model achieved 96% accuracy, best performance overall",
        details:
          "Successfully trained ensemble model combining Random Forest, Gradient Boosting, and Logistic Regression. Model achieved 96% accuracy on validation data.",
        type: "model",
        artifactId: "model-4",
        artifactName: "ensemble_model_v4.pkl",
        version: "v4",
      },
      {
        message: "Updated model metrics",
        shortDescription: "Added v4 model performance to metrics visualization",
        details:
          "Updated visualization with v4 model performance metrics showing the progression of improvements across all versions.",
        type: "metrics",
        artifactId: "metric-4",
        artifactName: "model_metrics",
        version: "v4",
      },
      {
        message: "ML pipeline complete",
        shortDescription: "Final ensemble model (v4) achieved 96% accuracy, 9% improvement over baseline",
        details:
          "Machine learning pipeline completed successfully. Final ensemble model (v4) achieved 96% accuracy, a 9% improvement over the baseline model.",
        type: "complete",
        version: "v4",
      },
    ]

    let i = 0
    const interval = setInterval(() => {
      if (i < activities.length) {
        const activity = {
          id: `activity-${Date.now()}`,
          timestamp: new Date(),
          message: activities[i].message,
          shortDescription: activities[i].shortDescription,
          details: activities[i].details,
          type: activities[i].type as any,
          artifactId: activities[i].artifactId,
          artifactName: activities[i].artifactName,
          version: activities[i].version,
          status: "done", // All previous activities are done
        }

        // Update activities with status
        setAgentActivities((prev) => {
          // Mark all previous activities as done
          const updatedActivities = prev.map((act) => ({
            ...act,
            status: "done",
          }))

          // Add the new activity
          return [...updatedActivities, activity]
        })

        setCurrentActivityIndex(i)
        setProgress(Math.min(100, Math.round(((i + 1) / activities.length) * 100)))

        // Update ready artifacts when an artifact is created
        if (activity.artifactId) {
          if (activity.artifactId.startsWith("code")) {
            setReadyArtifacts((prev) => ({
              ...prev,
              code: [...prev.code, activity.artifactId!],
            }))
          } else if (activity.artifactId.startsWith("model")) {
            setReadyArtifacts((prev) => ({
              ...prev,
              models: [...prev.models, activity.artifactId!],
            }))
          } else if (activity.artifactId.startsWith("metric")) {
            setReadyArtifacts((prev) => ({
              ...prev,
              metrics: [...prev.metrics, activity.artifactId!],
            }))
          }
        }

        i++
      } else {
        clearInterval(interval)
        setIsStreaming(false)
        setCurrentActivityIndex(-1) // No current activity
      }
    }, 1500) // Simulating streaming updates every 1.5 seconds

    // Store interval ID so we can clear it if the user stops the agent
    return interval
  }

  // Update the stopAgent function to clear the interval
  const stopAgent = () => {
    if (agentIntervalId) {
      clearInterval(agentIntervalId)
      setAgentIntervalId(null)
    }
    setIsStreaming(false)
    setCurrentActivityIndex(-1)
  }

  // Modify the getFilteredArtifacts function to only return ready artifacts
  const getFilteredArtifacts = (type: "code" | "models" | "metrics") => {
    return artifacts[type]
      .filter((item) => item.version === selectedVersion)
      .filter((item) => readyArtifacts[type].includes(item.id))
  }

  // Get the latest metric for the current version
  const getLatestMetric = () => {
    const versionMetrics = artifacts.metrics
      .filter((metric) => readyArtifacts.metrics.includes(metric.id))
      .sort((a, b) => {
        // Sort by version (v1, v2, v3, v4)
        return a.version.localeCompare(b.version, undefined, { numeric: true })
      })

    return versionMetrics.length > 0 ? versionMetrics[versionMetrics.length - 1] : null
  }

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

  const toggleSection = (section: keyof typeof openSections) => {
    setOpenSections((prev) => ({
      ...prev,
      [section]: !prev[section],
    }))
  }

  const toggleActivityExpand = (activityId: string) => {
    setExpandedActivities((prev) => ({
      ...prev,
      [activityId]: !prev[activityId],
    }))
  }

  const handleArtifactLink = (artifactId: string | undefined, version: string) => {
    if (!artifactId) return

    setSelectedVersion(version)
    setHighlightedArtifact(artifactId)

    // Find the section this artifact belongs to and open it
    if (artifactId.startsWith("code")) {
      setOpenSections((prev) => ({ ...prev, code: true }))
      setSelectedCodeFile(artifactId)
    } else if (artifactId.startsWith("model")) {
      setOpenSections((prev) => ({ ...prev, models: true }))
    } else if (artifactId.startsWith("metric")) {
      setOpenSections((prev) => ({ ...prev, metrics: true }))
    }
  }

  // Function to render the metrics chart
  const renderMetricsChart = (accuracies: number[]) => {
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
  }

  // Setup form when agent is not running
  if (!isAgentRunning) {
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
                    onClick={() => handleSendMessage()}
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
                  onClick={handleSendMessage}
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

  // Agent running UI with header and split panes
  return (
    <div className="flex flex-col h-screen bg-[#f9f9f9]">
      {/* Header with task summary */}
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
                        {expandedTaskDescription ? (
                          <ChevronUp className="h-3 w-3" />
                        ) : (
                          <ChevronDown className="h-3 w-3" />
                        )}
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
                <Badge className="bg-azure-gray text-gray-700 font-normal">{files.length} files</Badge>
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

      {/* Main content area with split panes */}
      <div className="flex-1 overflow-hidden">
        <div className="container mx-auto h-full py-4">
          <div className="grid grid-cols-1 md:grid-cols-12 gap-6 h-full">
            {/* Left pane: Agent progress */}
            <div className="flex flex-col h-full md:col-span-5">
              <Card className="flex-1 flex flex-col border-azure-border shadow-sm">
                <CardHeader className="border-b border-azure-border bg-white pb-3">
                  <CardTitle className="text-lg font-semibold text-azure-dark-blue">Agent Activity</CardTitle>
                  <CardDescription className="text-gray-600">
                    Live stream of the agent's actions and progress
                  </CardDescription>
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
                                    <Badge className="bg-azure-gray text-gray-700 text-xs font-normal">
                                      {activity.version}
                                    </Badge>

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
            </div>

            {/* Right pane: Artifacts */}
            <div className="flex flex-col h-full md:col-span-7">
              <div>WORK IN PROGRESS</div>
              <Card className="flex-1 flex flex-col border-azure-border shadow-sm hidden-view">
                <CardHeader className="border-b border-azure-border bg-white pb-3">
                  <div className="flex items-center justify-between">
                    <div>
                      <CardTitle className="text-lg font-semibold text-azure-dark-blue">ML Artifacts</CardTitle>
                      <CardDescription className="text-gray-600">
                        Code, models, and metrics generated by the agent
                      </CardDescription>
                    </div>

                    {/* Version selector - only show when versions are available */}
                    {availableVersions.length > 0 && (
                      <div className="flex items-center gap-2">
                        <Label htmlFor="version-select" className="text-sm text-gray-600">
                          Version:
                        </Label>
                        <Select value={selectedVersion} onValueChange={setSelectedVersion}>
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
                    <div className="space-y-6 pr-2">
                      {/* Models */}
                      <Collapsible
                        open={openSections.models}
                        onOpenChange={() => toggleSection("models")}
                        className="transition-all"
                      >
                        <CollapsibleTrigger className="flex items-center justify-between w-full p-2 rounded-md hover:bg-azure-gray/50">
                          <div className="flex items-center gap-2">
                            <Database className="h-5 w-5 text-azure-blue" />
                            <h3 className="text-sm font-medium text-gray-700">Models</h3>
                          </div>
                          {openSections.models ? (
                            <ChevronDown className="h-4 w-4 text-gray-500" />
                          ) : (
                            <ChevronRight className="h-4 w-4 text-gray-500" />
                          )}
                        </CollapsibleTrigger>
                        <CollapsibleContent className="pt-2">
                          <div className="space-y-4">
                            {getFilteredArtifacts("models").length > 0 ? (
                              getFilteredArtifacts("models").map((model) => (
                                <Card
                                  key={model.id}
                                  ref={(el) => (artifactRefs.current[model.id] = el)}
                                  className="transition-all duration-300 border-azure-border shadow-sm"
                                >
                                  <CardHeader className="py-3 bg-white">
                                    <div className="flex items-center justify-between">
                                      <div className="flex items-center gap-2">
                                        <CardTitle className="text-base font-medium text-gray-800">
                                          {model.name}
                                        </CardTitle>
                                        <Badge className="bg-azure-gray text-gray-700 text-xs font-normal">
                                          {model.version}
                                        </Badge>
                                      </div>
                                      <Button
                                        variant="outline"
                                        size="sm"
                                        className="border-azure-blue text-azure-blue hover:bg-azure-blue-5"
                                      >
                                        <Rocket className="h-3 w-3 mr-1" />
                                        Deploy
                                      </Button>
                                    </div>
                                  </CardHeader>
                                  <CardContent className="py-2 bg-azure-gray/10">
                                    <div className="grid grid-cols-2 gap-2 text-sm">
                                      <div>
                                        <span className="text-gray-600">Size:</span>{" "}
                                        <span className="text-gray-800">{model.size}</span>
                                      </div>
                                      <div>
                                        <span className="text-gray-600">Accuracy:</span>{" "}
                                        <span className="text-gray-800">{model.accuracy.toFixed(2)}</span>
                                      </div>
                                    </div>
                                  </CardContent>
                                </Card>
                              ))
                            ) : (
                              <div className="text-center p-6 text-gray-500 bg-azure-gray/20 rounded-md border border-azure-border flex flex-col items-center">
                                <Shield className="h-12 w-12 text-azure-blue/30 mb-2" />
                                <p>No models available yet</p>
                              </div>
                            )}
                          </div>
                        </CollapsibleContent>
                      </Collapsible>

                      {/* Metrics */}
                      <Collapsible
                        open={openSections.metrics}
                        onOpenChange={() => toggleSection("metrics")}
                        className="transition-all"
                      >
                        <CollapsibleTrigger className="flex items-center justify-between w-full p-2 rounded-md hover:bg-azure-gray/50">
                          <div className="flex items-center gap-2">
                            <BarChart3 className="h-5 w-5 text-azure-blue" />
                            <h3 className="text-sm font-medium text-gray-700">Metrics</h3>
                          </div>
                          {openSections.metrics ? (
                            <ChevronDown className="h-4 w-4 text-gray-500" />
                          ) : (
                            <ChevronRight className="h-4 w-4 text-gray-500" />
                          )}
                        </CollapsibleTrigger>
                        <CollapsibleContent className="pt-2">
                          <div className="space-y-4">
                            {currentMetricData.length > 0 ? (
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
                                    <Badge className="bg-azure-gray text-gray-700 text-xs font-normal">
                                      {selectedVersion}
                                    </Badge>
                                  </div>
                                </div>

                                {/* Render the metrics chart with the current data */}
                                {renderMetricsChart(currentMetricData)}

                                <div className="mt-2 text-xs text-center text-gray-500">
                                  Model accuracy comparison across versions
                                </div>
                              </div>
                            ) : (
                              <div className="text-center p-6 text-gray-500 bg-azure-gray/20 rounded-md border border-azure-border flex flex-col items-center">
                                <BarChart3 className="h-12 w-12 text-azure-blue/30 mb-2" />
                                <p>No metrics available yet</p>
                              </div>
                            )}
                          </div>
                        </CollapsibleContent>
                      </Collapsible>

                      {/* Code Files */}
                      <Collapsible
                        open={openSections.code}
                        onOpenChange={() => toggleSection("code")}
                        className="transition-all"
                      >
                        <CollapsibleTrigger className="flex items-center justify-between w-full p-2 rounded-md hover:bg-azure-gray/50">
                          <div className="flex items-center gap-2">
                            <FileCode className="h-5 w-5 text-azure-blue" />
                            <h3 className="text-sm font-medium text-gray-700">Code Files</h3>
                          </div>
                          {openSections.code ? (
                            <ChevronDown className="h-4 w-4 text-gray-500" />
                          ) : (
                            <ChevronRight className="h-4 w-4 text-gray-500" />
                          )}
                        </CollapsibleTrigger>
                        <CollapsibleContent className="pt-2">
                          <div
                            className="transition-all duration-300"
                            ref={(el) => (artifactRefs.current["code-section"] = el)}
                          >
                            {getFilteredArtifacts("code").length > 0 ? (
                              <div className="mb-4 border border-azure-border rounded-md shadow-sm">
                                <Tabs value={selectedCodeFile} onValueChange={setSelectedCodeFile}>
                                  <div className="border-b border-azure-border bg-azure-gray/30 p-1">
                                    <TabsList className="w-full justify-start h-auto bg-transparent p-0">
                                      {getFilteredArtifacts("code").map((file) => (
                                        <TabsTrigger
                                          key={file.id}
                                          value={file.id}
                                          className={`text-xs py-1 px-3 data-[state=active]:bg-white data-[state=active]:text-azure-blue ${
                                            selectedCodeFile === file.id ? "data-[state=active]:shadow-sm" : ""
                                          }`}
                                        >
                                          {file.name}
                                        </TabsTrigger>
                                      ))}
                                    </TabsList>
                                  </div>

                                  <div className="p-4 bg-white">
                                    {getFilteredArtifacts("code").map((file) => (
                                      <TabsContent
                                        key={file.id}
                                        value={file.id}
                                        className="m-0 transition-all rounded-md"
                                      >
                                        <div ref={(el) => (artifactRefs.current[file.id] = el)}>
                                          <div className="flex items-center justify-between mb-2">
                                            <div className="flex items-center gap-2">
                                              <h3 className="text-sm font-medium text-gray-800">{file.name}</h3>
                                              <Badge className="bg-azure-gray text-gray-700 text-xs font-normal">
                                                {file.version}
                                              </Badge>
                                            </div>
                                            <Button
                                              variant="outline"
                                              size="sm"
                                              className="border-azure-blue text-azure-blue hover:bg-azure-blue-5"
                                            >
                                              <FileText className="h-3 w-3 mr-1" />
                                              Download
                                            </Button>
                                          </div>
                                          <div className="bg-azure-gray/20 p-4 rounded-md border border-azure-border">
                                            <pre className="text-xs overflow-x-auto text-gray-800">
                                              <code>{file.content}</code>
                                            </pre>
                                          </div>
                                        </div>
                                      </TabsContent>
                                    ))}
                                  </div>
                                </Tabs>
                              </div>
                            ) : (
                              <div className="text-center p-6 text-gray-500 bg-azure-gray/20 rounded-md border border-azure-border flex flex-col items-center">
                                <FileCode className="h-12 w-12 text-azure-blue/30 mb-2" />
                                <p>No code files available yet</p>
                              </div>
                            )}
                          </div>
                        </CollapsibleContent>
                      </Collapsible>
                    </div>
                  </ScrollArea>
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
