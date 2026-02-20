import { onMount, onCleanup, createEffect } from "solid-js";
import { EditorView, keymap, lineNumbers, highlightActiveLine } from "@codemirror/view";
import { EditorState, Compartment } from "@codemirror/state";
import { python } from "@codemirror/lang-python";
import { indentUnit } from "@codemirror/language";
import {
  defaultKeymap,
  historyKeymap,
  history,
  indentWithTab,
} from "@codemirror/commands";
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
  let editorEl!: HTMLDivElement;
  let view: EditorView;
  const highlightCompartment = new Compartment();
  const { theme } = useTheme();

  onMount(() => {
    const state = EditorState.create({
      doc: props.code,
      extensions: [
        python(),
        history(),
        lineNumbers(),
        highlightActiveLine(),
        indentUnit.of("    "),
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
        ]),
        editorTheme,
        highlightCompartment.of(
          theme() === "dark" ? darkSyntaxHighlighting : lightSyntaxHighlighting
        ),
        EditorView.updateListener.of((update) => {
          if (update.docChanged) {
            props.onCodeChange(update.state.doc.toString());
          }
        }),
      ],
    });
    view = new EditorView({ state, parent: editorEl });
  });

  // Sync external code changes (Reset button)
  createEffect(() => {
    if (!view) return;
    const current = view.state.doc.toString();
    if (current !== props.code) {
      view.dispatch({
        changes: { from: 0, to: current.length, insert: props.code },
      });
    }
  });

  // Reactive theme switching
  createEffect(() => {
    if (!view) return;
    const t = theme();
    view.dispatch({
      effects: highlightCompartment.reconfigure(
        t === "dark" ? darkSyntaxHighlighting : lightSyntaxHighlighting
      ),
    });
  });

  onCleanup(() => view?.destroy());

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
      <div ref={editorEl!} class="code-panel-editor" />
    </div>
  );
}
