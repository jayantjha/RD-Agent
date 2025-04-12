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
import { Avatar, AvatarFallback } from "@/components/ui/avatar"

// Import ML Agent components
import { AgentProgress } from "@/components/ml-agent/AgentProgress"
import { AgentHeader } from "@/components/ml-agent/AgentHeader"
import { AgentArtifacts } from "@/components/ml-agent/AgentArtifacts"
import { ChatUI } from "@/components/ml-agent/ChatUI"
import { MetricsChart } from "@/components/ml-agent/metrics-chart"

export default function MLAgentPage() {
  const [userMessage, setUserMessage] = useState("")
  const [files, setFiles] = useState<File[]>([])
  const [isAgentRunning, setIsAgentRunning] = useState(false)
  const [isStreaming, setIsStreaming] = useState(false)
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
          "# ML Pipeline v2\nimport pandas as pd\nimport numpy as np\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.impute import KNNImputer\n\ndef load_data(file_path):\n    # Load the dataset\n    df = pd.read_csv(file_path)\n    return df\n\ndef preprocess(df):\n    # Handle missing values with KNN imputation\n    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns\n    categorical_cols = df.select_dtypes(include(['object']).columns\n    \n    # Handle categorical missing values\n    for col in categorical_cols:\n        df[col] = df[col].fillna(df[col].mode()[0])\n    \n    # Handle numerical missing values with KNN\n    if len(numerical_cols) > 0:\n        imputer = KNNImputer(n_neighbors=5)\n        df[numerical_cols] = imputer.fit.transform(df[numerical_cols])\n    \n    # Normalize numerical features\n    df[numerical_cols] = (df[numerical_cols] - df[numerical_cols].mean()) / df[numerical_cols].std()\n    \n    return df\n\ndef engineer_features(df):\n    # Create interaction features\n    for col1 in df.columns[:3]:\n        for col2 in df.columns[3:6]:\n            df[f'{col1}_{col2}_interaction'] = df[col1] * df[col2]\n    \n    # Create polynomial features\n    for col in df.select_dtypes(include(['float64', 'int64']).columns:\n        df[f'{col}_squared'] = df[col] ** 2\n    \n    return df\n\ndef train_model(X, y):\n    # Split data\n    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n    \n    # Train model with optimized hyperparameters\n    model = RandomForestClassifier(n_estimators=200, max_depth=15)\n    model.fit(X_train, y_train)\n    \n    # Evaluate\n    accuracy = model.score(X_test, y_test)\n    \n    return model, accuracy\n\n# Main pipeline execution\ndef run_pipeline(data_path, target_col):\n    # Load data\n    df = load_data(data_path)\n    \n    # Preprocess\n    df_processed = preprocess(df)\n    \n    # Feature engineering\n    df_engineered = engineer_features(df_processed)\n    \n    # Split features and target\n    X = df_engineered.drop(target_col, axis=1)\n    y = df_engineered[target_col]\n    \n    # Train model\n    model, accuracy = train_model(X, y)\n    \n    print(f\"Model trained with accuracy: {accuracy:.4f}\")\n    return model, accuracy",
      },
      {
        id: "code-3",
        name: "main.py",
        version: "v3",
        content:
          "# ML Pipeline v3\nimport pandas as pd\nimport numpy as np\nfrom sklearn.ensemble import GradientBoostingClassifier\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.impute import KNNImputer\nfrom sklearn.decomposition import PCA\n\ndef load_data(file_path):\n    # Load the dataset\n    df = pd.read_csv(file_path)\n    return df\n\ndef preprocess(df):\n    # Handle missing values with KNN imputation\n    numerical_cols = df.select_dtypes(include(['float64', 'int64']).columns\n    categorical_cols = df.select_dtypes(include(['object']).columns\n    \n    # Handle categorical missing values\n    for col in categorical_cols:\n        df[col] = df[col].fillna(df[col].mode()[0])\n    \n    # Handle numerical missing values with KNN\n    if len(numerical_cols) > 0:\n        imputer = KNNImputer(n_neighbors=5)\n        df[numerical_cols] = imputer.fit.transform(df[numerical_cols])\n    \n    # Normalize numerical features\n    df[numerical_cols] = (df[numerical_cols] - df[numerical_cols].mean()) / df[numerical_cols].std()\n    \n    return df\n\ndef engineer_features_advanced(df):\n    # Create interaction features\n    for col1 in df.columns[:3]:\n        for col2 in df.columns[3:6]:\n            df[f'{col1}_{col2}_interaction'] = df[col1] * df[col2]\n    \n    # Create polynomial features\n    for col in df.select_dtypes(include(['float64', 'int64']).columns[:5]:\n        df[f'{col}_squared'] = df[col] ** 2\n        df[f'{col}_cubed'] = df[col] ** 3\n    \n    # Apply PCA for dimensionality reduction\n    numerical_cols = df.select_dtypes(include(['float64', 'int64']).columns\n    if len(numerical_cols) > 10:\n        pca = PCA(n_components=10)\n        pca_result = pca.fit.transform(df[numerical_cols])\n        for i in range(10):\n            df[f'pca_{i+1}'] = pca_result[:, i]\n    \n    return df\n\ndef train_model(X, y):\n    # Split data\n    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n    \n    # Train model\n    model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1)\n    model.fit(X_train, y_train)\n    \n    # Evaluate\n    accuracy = model.score(X_test, y_test)\n    \n    return model, accuracy\n\n# Main pipeline execution\ndef run_pipeline(data_path, target_col):\n    # Load data\n    df = load_data(data_path)\n    \n    # Preprocess\n    df_processed = preprocess(df)\n    \n    # Advanced feature engineering\n    df_engineered = engineer_features_advanced(df_processed)\n    \n    # Split features and target\n    X = df_engineered.drop(target_col, axis=1)\n    y = df_engineered[target_col]\n    \n    # Train model\n    model, accuracy = train_model(X, y)\n    \n    print(f\"Model trained with accuracy: {accuracy:.4f}\")\n    return model, accuracy",
      },
      {
        id: "code-4",
        name: "main.py",
        version: "v4",
        content:
          "# ML Pipeline v4\nimport pandas as pd\nimport numpy as np\nfrom sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.impute import KNNImputer\nfrom sklearn.decomposition import PCA\nfrom sklearn.feature_selection import SelectFromModel, RFE\n\ndef load_data(file_path):\n    # Load the dataset\n    df = pd.read_csv(file_path)\n    return df\n\ndef preprocess(df):\n    # Handle missing values with KNN imputation\n    numerical_cols = df.select_dtypes(include(['float64', 'int64']).columns\n    categorical_cols = df.select_dtypes(include(['object']).columns\n    \n    # Handle categorical missing values\n    for col in categorical_cols:\n        df[col] = df[col].fillna(df[col].mode()[0])\n    \n    # Handle numerical missing values with KNN\n    if len(numerical_cols) > 0:\n        imputer = KNNImputer(n_neighbors=5)\n        df[numerical_cols] = imputer.fit.transform(df[numerical_cols])\n    \n    # Normalize numerical features\n    df[numerical_cols] = (df[numerical_cols] - df[numerical_cols].mean()) / df[numerical_cols].std()\n    \n    return df\n\ndef engineer_features_advanced(df):\n    # Create interaction features\n    for col1 in df.columns[:3]:\n        for col2 in df.columns[3:6]:\n            df[f'{col1}_{col2}_interaction'] = df[col1] * df[col2]\n    \n    # Create polynomial features\n    for col in df.select_dtypes(include(['float64', 'int64']).columns[:5]:\n        df[f'{col}_squared'] = df[col] ** 2\n        df[f'{col}_cubed'] = df[col] ** 3\n    \n    # Apply PCA for dimensionality reduction\n    numerical_cols = df.select_dtypes(include(['float64', 'int64']).columns\n    if len(numerical_cols) > 10:\n        pca = PCA(n_components=10)\n        pca_result = pca.fit.transform(df[numerical_cols])\n        for i in range(10):\n            df[f'pca_{i+1}'] = pca_result[:, i]\n    \n    return df\n\ndef select_features(X, y):\n    # Method 1: Feature importance from Random Forest\n    rf = RandomForestClassifier(n_estimators=100)\n    rf.fit(X, y)\n    \n    # Select top features based on importance\n    sfm = SelectFromModel(rf, threshold='median')\n    X_selected = sfm.fit.transform(X, y)\n    \n    # Method 2: Recursive Feature Elimination\n    rfe = RFE(estimator=RandomForestClassifier(n_estimators=50), n_features_to_select=20)\n    X_rfe = rfe.fit.transform(X, y)\n    \n    # Get feature names\n    selected_features = X.columns[sfm.get_support()]\n    rfe_features = X.columns[rfe.get_support()]\n    \n    # Combine both methods\n    final_features = list(set(selected_features) | set(rfe_features))\n    \n    return X[final_features], final_features\n\ndef train_ensemble(X, y):\n    # Split data\n    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n    \n    # Create base models\n    rf = RandomForestClassifier(n_estimators=200, max_depth=20)\n    gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1)\n    lr = LogisticRegression(C=0.1)\n    \n    # Create and train ensemble\n    ensemble = VotingClassifier(\n        estimators=[('rf', rf), ('gb', gb), ('lr', lr)],\n        voting='soft'\n    )\n    ensemble.fit(X_train, y_train)\n    \n    # Evaluate\n    accuracy = ensemble.score(X_test, y_test)\n    \n    return ensemble, accuracy\n\n# Main pipeline execution\ndef run_pipeline(data_path, target_col):\n    # Load data\n    df = load_data(data_path)\n    \n    # Preprocess\n    df_processed = preprocess(df)\n    \n    # Advanced feature engineering\n    df_engineered = engineer_features_advanced(df_processed)\n    \n    # Split features and target\n    X = df_engineered.drop(target_col, axis=1)\n    y = df.engineered[target_col]\n    \n    # Feature selection\n    X_selected, selected_features = select_features(X, y)\n    print(f\"Selected {len(selected_features)} features\")\n    \n    # Train ensemble model\n    model, accuracy = train_ensemble(X_selected, y)\n    \n    print(f\"Ensemble model trained with accuracy: {accuracy:.4f}\")\n    return model, accuracy",
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
  // useEffect(() => {
  //   if (highlightedArtifact && artifactRefs.current[highlightedArtifact]) {
  //     artifactRefs.current[highlightedArtifact]?.scrollIntoView({ behavior: "smooth", block: "center" })

  //     // Flash effect - using bg-blue-100 instead of bg-azure-blue-10 which may not exist in default Tailwind
  //     const element = artifactRefs.current[highlightedArtifact]
  //     element?.classList.add("bg-blue-100")
  //     setTimeout(() => {
  //       element?.classList.remove("bg-blue-100")
  //     }, 2000)
  //   }
  // }, [highlightedArtifact])

  // Auto-scroll to the bottom of the activity stream when new activities are added
  // useEffect(() => {
  //   if (agentActivities.length > 0 && isStreaming) {
  //     activityStreamEndRef.current?.scrollIntoView({ behavior: "smooth" })
  //   }
  // }, [agentActivities, isStreaming])

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
    // test code
    versions.add ("0");
    versions.add ("1");
    console.log(Array.from(versions));
    setAvailableVersions(Array.from(versions))

    // Add versions from ready artifacts
    // readyArtifacts.code.forEach((id) => {
    //   const artifact = artifacts.code.find((a) => a.id === id)
    //   if (artifact) versions.add(artifact.version)
    // })

    // readyArtifacts.models.forEach((id) => {
    //   const artifact = artifacts.models.find((a) => a.id === id)
    //   if (artifact) versions.add(artifact.version)
    // })

    // readyArtifacts.metrics.forEach((id) => {
    //   const artifact = artifacts.metrics.find((a) => a.id === id)
    //   if (artifact) versions.add(artifact.version)
    // })

    // const sortedVersions = Array.from(versions).sort((a, b) => {
    //   // Sort by version number (v1, v2, v3, v4)
    //   return a.localeCompare(b, undefined, { numeric: true })
    // })

    // setAvailableVersions(sortedVersions)

    // // Always set selected version to the latest available
    // if (sortedVersions.length > 0) {
    //   setSelectedVersion(sortedVersions[sortedVersions.length - 1])
    // }

    // readyArtifacts, artifacts
  }, [isAgentRunning])

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
  const [eventSource, setEventSource] = useState<EventSource | null>(null)

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
        setMessages((prev) => [
          ...prev,
          {
            ...message,
            role: message.role as "user" | "agent", // Ensure role is explicitly typed
          },
        ])

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
    }, 100)
    // 1000

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

    // Connect to the streaming updates endpoint
    connectToEventStream()
  }

  // event mappings
  const mappings: Record<string, string> = {
    "DS_LOOP": "ML Agent",
    "DS_SCENARIO": "Understanding data and requirements",
    "RDLOOP": "Main R & D loop",
    "CODING": "Coder agent",
    "EXPERIMENT_GENERATION": "Generating experiment for the loop",
    "DATA_LOADING": "Code for loading data",
    "FEATURE_TASK": "Code for feature engineering",
    "MODEL_TASK": "Code for hypothesized model",
    "ENSEMBLE_TASK": "Generating ensemble model",
    "WORKFLOW_TASK": "Developing workflow",
    "FEEDBACK": "Gathering feedback for the loop",
    "RECORD": "Recording results"
  }
  
  const connectToEventStream = () => {
    // Close any existing connection
    if (eventSource) {
      eventSource.close()
    }
    
    // Create a new EventSource connection
    const newEventSource = new EventSource('http://localhost:5000/updates/saved/thread_r4EZ1fbjwQiUmrtZUULEjh8M')
    setEventSource(newEventSource)

    // Set up event handlers
    newEventSource.onmessage = (event) => {
      try {
        console.log(event)
        const data = JSON.parse(event.data)
        processAgentActivity(data)
      } catch (error) {
        console.error('Error parsing event data:', error)
      }
    }

    newEventSource.onerror = (error) => {
      console.error('EventSource error:', error)
      
      // Attempt to reconnect after a delay if streaming is still active
      if (isStreaming) {
        setTimeout(() => {
          if (isStreaming) {
            console.log('Attempting to reconnect to event stream...')
            connectToEventStream()
          }
        }, 3000)
      } else {
        newEventSource.close()
        setEventSource(null)
      }
    }
  }

  const processAgentActivity = (data: any) => {
    // Convert streaming data to activity format
    console.log("activity", data)

    // activity mappings
    
    const activity = {
      id: data.id || `activity-${Date.now()}`,
      timestamp: new Date(data.createdAt * 1000),
      message: `${mappings[data.task] || data.task} : ${data.status.toLowerCase()}`,
      shortDescription: data.shortDescription || "",
      details: data.message || "No details provided",
      type: data.type || "info",
      artifactId: data.artifactId,
      artifactName: data.artifactName,
      version: data.version || "v1",
      status: data.status || "done",
    }

    // Update activities state
    setAgentActivities((prev) => {
      // Mark all previous activities as done
      const updatedActivities = prev.map((act) => ({
        ...act,
        status: "done",
      }))

      // Add the new activity
      return [...updatedActivities, activity]
    })

    // Update current activity index
    setCurrentActivityIndex((prev) => prev + 1)
    
    // Calculate progress (approximation since we don't know total number)
    // You may need to adjust this based on your specific use case
    setProgress((prevProgress) => Math.min(100, prevProgress + 5))

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

    // If the message indicates completion, end the streaming
    if (activity.type === "complete") {
      setIsStreaming(false)
      setCurrentActivityIndex(-1)
      if (eventSource) {
        eventSource.close()
        setEventSource(null)
      }
    }
  }

  const stopAgent = () => {
    // Close the EventSource connection
    if (eventSource) {
      eventSource.close()
      setEventSource(null)
    }
    
    // Clean up any existing interval
    if (agentIntervalId) {
      clearInterval(agentIntervalId)
      setAgentIntervalId(null)
    }
    
    setIsStreaming(false)
    setCurrentActivityIndex(-1)
  }

  // Clean up EventSource on component unmount
  useEffect(() => {
    return () => {
      if (eventSource) {
        eventSource.close()
      }
    }
  }, [eventSource])

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
    // Use the MetricsChart component instead of inline SVG
    return <MetricsChart accuracies={accuracies} chartRef={chartRef} />
  }

  // Setup form when agent is not running
  // !isAgentRunning
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

          <ChatUI 
            messages={messages}
            userMessage={userMessage}
            setUserMessage={setUserMessage}
            handleSendMessage={handleSendMessage}
            handleFileChange={handleFileChange}
            files={files}
            readyToStart={readyToStart}
            startAgent={startAgent} // Add missing prop
            messagesEndRef={messagesEndRef as React.RefObject<HTMLDivElement>}
            setReadyToStart={function (ready: boolean): void {
              throw new Error("Function not implemented.")
            } }          />
        </div>
      </div>
    )
  }

  // Agent running UI with header and split panes
  return (
    <div className="flex flex-col h-screen bg-[#f9f9f9]">
      {/* Header with task summary */}
      <AgentHeader 
        taskDescription={taskDescription}
        filesCount={files.length}
        progress={progress}
        isStreaming={isStreaming}
        stopAgent={stopAgent}
      />

      {/* Main content area with split panes */}
      <div className="flex-1 overflow-hidden">
        <div className="container mx-auto h-full py-4">
          <div className="grid grid-cols-1 md:grid-cols-12 gap-6 h-full">
            {/* Left pane: Agent progress */}
            <div className="flex flex-col h-full md:col-span-5">
              <AgentProgress 
                startAgent={startAgent}
                agentActivities={agentActivities}
                isStreaming={isStreaming}
                currentActivityIndex={currentActivityIndex}
                expandedActivities={expandedActivities}
                toggleActivityExpand={toggleActivityExpand}
                // getActivityIcon={getActivityIcon}
                handleArtifactLink={handleArtifactLink}
                activityStreamEndRef={activityStreamEndRef as React.RefObject<HTMLDivElement>}
              />   
            </div>
            {/* Right pane: Artifacts */}
            <div className="flex flex-col h-full md:col-span-7">
              <AgentArtifacts 
                availableVersions={availableVersions}
                selectedVersion={selectedVersion}
                setSelectedVersion={setSelectedVersion}
                openSections={openSections}
                toggleSection={toggleSection}
                getFilteredArtifacts={getFilteredArtifacts}
                selectedCodeFile={selectedCodeFile}
                setSelectedCodeFile={setSelectedCodeFile}
                readyArtifacts={readyArtifacts}
                artifactRefs={artifactRefs}
                artifacts={artifacts}
                currentMetricData={currentMetricData}
                getLatestMetric={getLatestMetric}
                renderMetricsChart={renderMetricsChart}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
