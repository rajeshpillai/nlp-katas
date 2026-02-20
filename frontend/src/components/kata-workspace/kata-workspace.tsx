import { createSignal } from "solid-js";
import Resizable from "@corvu/resizable";
import { CodePanel } from "./code-panel";
import { OutputPanel } from "./output-panel";
import { executeCode, type ExecutionResult } from "../../lib/api-client";
import "./kata-workspace.css";

interface Props {
  markdown: string;
  kataId: string;
}

function extractCode(markdown: string): string {
  const match = markdown.match(/```python\n([\s\S]*?)```/);
  return match ? match[1].trim() : "# No code found in this kata";
}

export function KataWorkspace(props: Props) {
  const starterCode = () => extractCode(props.markdown);
  const [code, setCode] = createSignal(starterCode());
  const [output, setOutput] = createSignal<ExecutionResult | null>(null);
  const [running, setRunning] = createSignal(false);

  const handleRun = async () => {
    setRunning(true);
    try {
      const result = await executeCode(code(), props.kataId);
      setOutput(result);
    } catch {
      setOutput({ stdout: "", stderr: "", error: "Failed to connect to server.", execution_time_ms: 0 });
    } finally {
      setRunning(false);
    }
  };

  const handleReset = () => {
    setCode(starterCode());
    setOutput(null);
  };

  return (
    <Resizable class="kata-workspace">
      <Resizable.Panel class="kata-workspace-panel" initialSize={0.5} minSize={0.2}>
        <CodePanel
          code={code()}
          onCodeChange={setCode}
          onRun={handleRun}
          onReset={handleReset}
          running={running()}
        />
      </Resizable.Panel>
      <Resizable.Handle class="kata-workspace-handle" />
      <Resizable.Panel class="kata-workspace-panel" initialSize={0.5} minSize={0.2}>
        <OutputPanel result={output()} running={running()} />
      </Resizable.Panel>
    </Resizable>
  );
}
