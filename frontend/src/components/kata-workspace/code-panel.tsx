import { createCodeMirror, createEditorControlledValue } from "solid-codemirror";
import { python } from "@codemirror/lang-python";
import { indentUnit } from "@codemirror/language";
import {
  defaultKeymap,
  historyKeymap,
  history,
  indentWithTab,
} from "@codemirror/commands";
import { keymap, lineNumbers, highlightActiveLine } from "@codemirror/view";
import { useTheme } from "../../context/theme-context";
import {
  editorTheme,
  lightSyntaxHighlighting,
  darkSyntaxHighlighting,
} from "./editor-theme";
import "./code-panel.css";

interface Props {
  code: string;
  onCodeChange: (code: string) => void;
  onRun: () => void;
  onReset: () => void;
  running: boolean;
  maximized: boolean;
  onToggleMaximize: () => void;
}

export function CodePanel(props: Props) {
  const { theme } = useTheme();

  const { ref, editorView, createExtension } = createCodeMirror({
    value: props.code,
    onValueChange: props.onCodeChange,
  });

  createEditorControlledValue(editorView, () => props.code);

  createExtension(python());
  createExtension(history());
  createExtension(lineNumbers());
  createExtension(highlightActiveLine());
  createExtension(indentUnit.of("    "));

  createExtension(
    keymap.of([
      {
        key: "Ctrl-Enter",
        mac: "Cmd-Enter",
        run: () => {
          props.onRun();
          return true;
        },
      },
      indentWithTab,
      ...defaultKeymap,
      ...historyKeymap,
    ])
  );

  createExtension(editorTheme);
  createExtension(() =>
    theme() === "dark" ? darkSyntaxHighlighting : lightSyntaxHighlighting
  );

  return (
    <div class="code-panel">
      <div class="code-panel-toolbar">
        <button
          class="code-panel-run"
          onClick={props.onRun}
          disabled={props.running}
        >
          {props.running ? "Running..." : "Run"}
        </button>
        <button class="code-panel-reset" onClick={props.onReset}>
          Reset
        </button>
        <span class="code-panel-hint">Ctrl+Enter to run</span>
        <button
          class="panel-maximize-btn"
          onClick={props.onToggleMaximize}
          title={props.maximized ? "Restore" : "Maximize"}
        >
          {props.maximized ? "\u29C9" : "\u2922"}
        </button>
      </div>
      <div ref={ref} class="code-panel-editor" />
    </div>
  );
}
