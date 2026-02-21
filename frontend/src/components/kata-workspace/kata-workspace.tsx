import { createSignal, createEffect, Show } from "solid-js";
import Resizable from "@corvu/resizable";
import { CodePanel } from "./code-panel";
import { OutputPanel } from "./output-panel";
import { executeCode, type ExecutionResult } from "../../lib/api-client";
import "./kata-workspace.css";

interface Props {
  markdown: string;
  kataId: string;
  trackId: string;
}

function extractCode(markdown: string, trackId: string): string {
  if (trackId.includes("rust")) {
    const rustMatch = markdown.match(/```rust\n([\s\S]*?)```/);
    if (rustMatch) return rustMatch[1].trim();
    return "fn main() {\n    // No code found in this kata\n}";
  }
  const match = markdown.match(/```python\n([\s\S]*?)```/);
  return match ? match[1].trim() : "# No code found in this kata";
}

export type MaximizedPanel = "code" | "output" | null;

export function KataWorkspace(props: Props) {
  const [code, setCode] = createSignal(extractCode(props.markdown, props.trackId));
  const [output, setOutput] = createSignal<ExecutionResult | null>(null);
  const [running, setRunning] = createSignal(false);

  // Reset code and output when navigating to a different kata
  createEffect(() => {
    setCode(extractCode(props.markdown, props.trackId));
    setOutput(null);
  });
  const [maximized, setMaximized] = createSignal<MaximizedPanel>(null);
  const [sizes, setSizes] = createSignal([0.5, 0.5]);

  const toggleMaximize = (panel: "code" | "output") => {
    if (maximized() === panel) {
      setMaximized(null);
      setSizes([0.5, 0.5]);
    } else {
      setMaximized(panel);
      setSizes(panel === "code" ? [1, 0] : [0, 1]);
    }
  };

  const handleRun = async () => {
    setRunning(true);
    try {
      const result = await executeCode(code(), props.kataId, props.trackId);
      setOutput(result);
    } catch {
      setOutput({ stdout: "", stderr: "", error: "Failed to connect to server.", execution_time_ms: 0 });
    } finally {
      setRunning(false);
    }
  };

  const handleReset = () => {
    setCode(extractCode(props.markdown, props.trackId));
    setOutput(null);
  };

  const language = () => props.trackId.includes("rust") ? "rust" as const : "python" as const;

  return (
    <Resizable
      class="kata-workspace"
      sizes={sizes()}
      onSizesChange={setSizes}
    >
      <Resizable.Panel
        class="kata-workspace-panel"
        minSize={maximized() === "output" ? 0 : 0.2}
      >
        <Show when={maximized() !== "output"}>
          <CodePanel
            code={code()}
            onCodeChange={setCode}
            onRun={handleRun}
            onReset={handleReset}
            running={running()}
            maximized={maximized() === "code"}
            onToggleMaximize={() => toggleMaximize("code")}
            language={language()}
          />
        </Show>
      </Resizable.Panel>
      <Show when={!maximized()}>
        <Resizable.Handle class="kata-workspace-handle" />
      </Show>
      <Resizable.Panel
        class="kata-workspace-panel"
        minSize={maximized() === "code" ? 0 : 0.2}
      >
        <Show when={maximized() !== "code"}>
          <OutputPanel
            result={output()}
            running={running()}
            maximized={maximized() === "output"}
            onToggleMaximize={() => toggleMaximize("output")}
          />
        </Show>
      </Resizable.Panel>
    </Resizable>
  );
}
