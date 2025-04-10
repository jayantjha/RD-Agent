# Azure ML Application - Client

This is the client-side application for the Azure Machine Learning Research and Development Agent project. It provides a user interface for interacting with Azure Machine Learning services.

## Overview

This client application allows users to:
- Interact with Azure Machine Learning workspaces
- Submit and monitor machine learning jobs
- View experiment results and metrics
- Manage machine learning models and deployments

## Prerequisites

- Node.js (v14.x or higher)
- npm (v6.x or higher) or yarn (v1.22.x or higher)
- Access to Azure Machine Learning resources

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-organization/aml-rd-agent.git
   cd aml-rd-agent/azureml-app/client
   ```

2. Install dependencies:
   ```bash
   npm install
   # or using yarn
   yarn install
   ```

3. Create a `.env` file in the client directory with the following variables:
   ```
   REACT_APP_API_URL=http://localhost:5000/api
   ```

## Running the Application

### Development Mode

To start the development server:

```bash
npm start
# or using yarn
yarn start
```

The application will be available at `http://localhost:3000`.

### Production Build

To create a production build:

```bash
npm run build
# or using yarn
yarn build
```

The build output will be in the `build` directory.

## Project Structure

```
client/
├── public/          # Static files
├── src/             # Source files
│   ├── components/  # React components
│   ├── pages/       # Page components
│   ├── services/    # API services
│   ├── utils/       # Utility functions
│   ├── App.js       # Main App component
│   └── index.js     # Entry point
├── .env             # Environment variables
└── package.json     # Project dependencies
```

## Available Scripts

- `npm start` - Starts the development server
- `npm run build` - Builds the app for production
- `npm test` - Runs the test suite
- `npm run eject` - Ejects from Create React App

## Connecting to the Backend

The client application connects to the backend API defined in the `azureml-app/server` directory. Make sure the server is running before using the client application.

## Contributing

1. Follow the established coding style and patterns
2. Write tests for new features
3. Make sure all tests pass before submitting a pull request

## Troubleshooting

If you encounter issues:
1. Verify your environment variables are correctly set
2. Check that the backend server is running
3. Clear browser cache and node_modules if facing persistent issues
4. Check browser console for specific error messages

For additional help, refer to the project documentation or contact the development team.
