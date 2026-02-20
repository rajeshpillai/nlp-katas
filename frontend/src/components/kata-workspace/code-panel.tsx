import { Show } from "solid-js";
import "./code-panel.css";

interface Props {
  code: string;
  onCodeChange: (code: string) => void;
  onRun: () => void;
  onReset: () => void;
  running: boolean;
}

export function CodePanel(props: Props) {
  const handleKeyDown = (e: KeyboardEvent) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      props.onRun();
    }
    if (e.key === "Tab") {
      e.preventDefault();
      const target = e.target as HTMLTextAreaElement;
      const start = target.selectionStart;
      const end = target.selectionEnd;
      const value = target.value;
      const newValue = value.substring(0, start) + "    " + value.substring(end);
      props.onCodeChange(newValue);
      requestAnimationFrame(() => {
        target.selectionStart = target.selectionEnd = start + 4;
      });
    }
  };

  return (
    <div class="code-panel">
      <div class="code-panel-toolbar">
        <button
          class="code-panel-run"
          onClick={props.onRun}
          disabled={props.running}
        >
          <Show when={props.running} fallback="Run">
            Running...
          </Show>
        </button>
        <button class="code-panel-reset" onClick={props.onReset}>
          Reset
        </button>
        <span class="code-panel-hint">Ctrl+Enter to run</span>
      </div>
      <textarea
        class="code-panel-editor"
        value={props.code}
        onInput={(e) => props.onCodeChange(e.currentTarget.value)}
        onKeyDown={handleKeyDown}
        spellcheck={false}
      />
    </div>
  );
}
