import React, { useEffect, useState } from "react";
import { Shield, Rocket, Database } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { useFileContent } from "@/lib/queries/useFileContent";
import ReactMarkdown from "react-markdown";
import 'github-markdown-css/github-markdown.css';

interface ModelsArtifactsTabProps {
  getFilteredArtifacts: (type: "code" | "models" | "metrics") => any[];
  artifactRefs: React.MutableRefObject<{ [key: string]: HTMLDivElement | null }>;
  manifestData: any | null;
  formatSizeInMB: (sizeInBytes: number) => string;
  sessionId: string | undefined;
}


export function ModelsArtifactsTab({
  getFilteredArtifacts,
  artifactRefs,
  manifestData,
  formatSizeInMB,
  sessionId,
}: ModelsArtifactsTabProps) {
  // Filter models artifacts as before
  const modelArtifacts = getFilteredArtifacts("models");
  
  // Extract model information from manifest data
  const modelInfo = manifestData?.model;
  const readmeFile = manifestData?.files.find((file: { name: string; }) => file.name === "README.md");
  let readmeFilePath;
  if (readmeFile) {
    readmeFilePath = `${manifestData.workspace_path}/${readmeFile.name}`;
  }
  
  const { data: fileContent, isLoading, error } = useFileContent(sessionId, readmeFilePath);

  console.log(modelInfo);
  return (
    <div className="space-y-4">
      {/* First row: Display model information from manifest if available */}
      {modelInfo && (
        <div className="flex justify-center">
          <Card className="w-96 border-azure-border shadow-sm">
            <CardHeader className="pb-2">
              <CardTitle className="text-lg font-semibold flex items-center">
                <Database className="h-5 w-5 mr-2 text-azure-blue" />
                Model Information
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="font-medium text-gray-700">Name:</span>
                  <Badge variant="outline" className="bg-azure-lightest text-azure-dark-blue">
                    {modelInfo.name}
                  </Badge>
                </div>
                <div className="flex justify-between items-center">
                  <span className="font-medium text-gray-700">Size:</span>
                  <Badge variant="outline" className="bg-azure-lightest text-azure-dark-blue">
                    {formatSizeInMB(modelInfo.size)}
                  </Badge>
                </div>
                {modelInfo.files && modelInfo.files.length > 0 && (
                  <div className="border-t pt-2 mt-2">
                    <p className="font-medium text-gray-700 mb-2">Files:</p>
                    {modelInfo.files.map((file: any, index: number) => (
                      <div key={index} className="pl-2 text-sm border-l-2 border-azure-light mb-1">
                        <p>{file.name}</p>
                        <p className="text-gray-500">{formatSizeInMB(file.size)}</p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Second row: Display README content if available */}
      {fileContent && (
         <div className="markdown-body">
         <ReactMarkdown>{fileContent}</ReactMarkdown>
         {/* <pre className="bg-gray-100 p-4 rounded-md overflow-auto">
           {fileContent || ''}
         </pre> */}
       </div>
        // <div className="flex justify-center">
        //   <Card className="w-full max-w-4xl border-azure-border shadow-sm">
        //     <CardContent>
        //       <div className="markdown-body">
        //         <ReactMarkdown>{fileContent}</ReactMarkdown>
        //         {/* <pre className="bg-gray-100 p-4 rounded-md overflow-auto">
        //           {fileContent || ''}
        //         </pre> */}
        //       </div>
        //     </CardContent>
        //   </Card>
        // </div>
      )}

      {/* Display message if no model information is available */}
      {!modelInfo && !fileContent && (
        <div className="text-center p-6 text-gray-500 bg-azure-gray/20 rounded-md border border-azure-border flex flex-col items-center">
          <Shield className="h-12 w-12 text-azure-blue/30 mb-2" />
          <p>No models available yet</p>
        </div>
      )}
    </div>
  );
}
