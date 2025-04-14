import React, { useEffect } from "react"
import { FileCode, FileText, Download } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { TabsList, TabsTrigger, Tabs, TabsContent } from "@/components/ui/tabs"
import { useFileContent } from "@/lib/queries/useFileContent"
import SyntaxHighlighter from 'react-syntax-highlighter'
import { vs2015 } from 'react-syntax-highlighter/dist/esm/styles/hljs'
import { Skeleton } from "@/components/ui/skeleton"

interface CodeArtifactsTabProps {
  getFilteredArtifacts: (type: "code" | "models" | "metrics") => any[]
  selectedCodeFile: string
  setSelectedCodeFile: (fileId: string) => void
  artifactRefs: React.MutableRefObject<{ [key: string]: HTMLDivElement | null }>
  manifestData: any
  runId: string | undefined
  sessionId: string | undefined
}

export function CodeArtifactsTab({
  getFilteredArtifacts,
  selectedCodeFile,
  setSelectedCodeFile,
  artifactRefs,
  manifestData,
  runId,
  sessionId,
}: CodeArtifactsTabProps) {
  // const codeFiles = getFilteredArtifacts("code");
  // const selectedFile = codeFiles.find(file => file.id === selectedCodeFile);
  // hardcoding
  const selectedFile = {
    name: "main.py"
  };
  // Constants
  const codeFiles = [{
    id: "",
    name: "main.py"
  }];
  
  // Determine the file path from manifest data and selected file
  const filePath = selectedFile && manifestData?.workspace_path 
    ? `${manifestData.workspace_path}/${selectedFile.name}`
    : undefined;
  // Fetch file content using our new hook with the sessionId from props
  const { data: fileContent, isLoading, error } = useFileContent(sessionId, filePath);
  
  // If no file is selected and we have files, select the first one
  useEffect(() => {
    if (codeFiles.length > 0 && !selectedCodeFile) {
      setSelectedCodeFile(codeFiles[0].id);
    }
  }, [codeFiles, selectedCodeFile, setSelectedCodeFile]);

  // Determine file language for syntax highlighting
  const getLanguage = (fileName: string): string => {
    const extension = fileName.split('.').pop()?.toLowerCase() || '';
    switch (extension) {
      case 'py': return 'python';
      case 'js': return 'javascript';
      case 'ts': return 'typescript';
      case 'tsx': return 'typescript';
      case 'jsx': return 'javascript';
      case 'json': return 'json';
      case 'html': return 'html';
      case 'css': return 'css';
      case 'cpp': case 'c': case 'h': return 'cpp';
      default: return 'text';
    }
  };

  // Handle file download
  const handleDownload = (fileName: string, content: string) => {
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = fileName;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div
      className="transition-all duration-300"
      ref={(el) => {
        artifactRefs.current["code-section"] = el;
      }}
    >
      {manifestData && codeFiles.length > 0 ? (
        <div className="border border-azure-border rounded-md shadow-sm">
          <Tabs value={selectedCodeFile} onValueChange={setSelectedCodeFile}>
            <div className="border-b border-azure-border bg-azure-gray/30 p-1">
              <TabsList className="w-full justify-start h-auto bg-transparent p-0">
                {codeFiles.map((file) => (
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
              {codeFiles.map((file) => (
                <TabsContent key={file.id} value={file.id} className="m-0 transition-all rounded-md">
                  <div ref={(el) => { artifactRefs.current[file.id] = el; }}>
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <h3 className="text-sm font-medium text-gray-800">{file.name}</h3>
                        {/* <Badge className="bg-azure-gray text-gray-700 text-xs font-normal">
                          {file.version}
                        </Badge> */}
                      </div>
                      {fileContent && (
                        <Button
                          variant="outline"
                          size="sm"
                          className="border-azure-blue text-azure-blue hover:bg-azure-blue-5"
                          onClick={() => handleDownload(file.name, fileContent)}
                        >
                          <Download className="h-3 w-3 mr-1" />
                          Download
                        </Button>
                      )}
                    </div>
                    <div className="bg-azure-gray/20 rounded-md border border-azure-border overflow-hidden">
                      {isLoading ? (
                        <div className="p-4">
                          <Skeleton className="h-4 w-full mb-2" />
                          <Skeleton className="h-4 w-3/4 mb-2" />
                          <Skeleton className="h-4 w-5/6 mb-2" />
                          <Skeleton className="h-4 w-2/3" />
                        </div>
                      ) : error ? (
                        <div className="text-red-500 p-4 text-sm">
                          Error loading file content: {error.message}
                        </div>
                      ) : (
                        <SyntaxHighlighter
                          language={getLanguage(file.name)}
                          style={vs2015}
                          customStyle={{
                            margin: 0,
                            padding: '1rem',
                            fontSize: '0.75rem',
                            lineHeight: '1.4',
                            borderRadius: '0.25rem',
                          }}
                        >
                          {fileContent || ''}
                        </SyntaxHighlighter>
                      )}
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
  )
}
