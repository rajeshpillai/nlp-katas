import { Show } from "solid-js";
import type { ExecutionResult } from "../../lib/api-client";
import "./output-panel.css";

interface Props {
  result: ExecutionResult | null;
  running: boolean;
}

export function OutputPanel(props: Props) {
  return (
    <div class="output-panel">
      <div class="output-panel-header">
        Output
        <Show when={props.result}>
          <span class="output-panel-time">
            {props.result!.execution_time_ms.toFixed(1)}ms
          </span>
        </Show>
      </div>
      <div class="output-panel-body">
        <Show when={props.running}>
          <p class="output-panel-running">Running...</p>
        </Show>
        <Show when={!props.running && !props.result}>
          <p class="output-panel-empty">Run your code to see output here.</p>
        </Show>
        <Show when={!props.running && props.result}>
          <Show when={props.result!.stdout}>
            <pre class="output-panel-stdout">{props.result!.stdout}</pre>
          </Show>
          <Show when={props.result!.error}>
            <pre class="output-panel-error">{props.result!.error}</pre>
          </Show>
          <Show when={props.result!.stderr && !props.result!.error}>
            <pre class="output-panel-stderr">{props.result!.stderr}</pre>
          </Show>
        </Show>
      </div>
    </div>
  );
}
