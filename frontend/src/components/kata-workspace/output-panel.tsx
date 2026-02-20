import { Show, For, Switch, Match } from "solid-js";
import type { ExecutionResult } from "../../lib/api-client";
import { parseOutput, type OutputSegment } from "../../lib/output-parser";
import { RichTextSegment } from "./rich-output/rich-text-segment";
import { RichHtmlSegment } from "./rich-output/rich-html-segment";
import { RichSvgSegment } from "./rich-output/rich-svg-segment";
import { RichImageSegment } from "./rich-output/rich-image-segment";
import { RichChartSegment } from "./rich-output/rich-chart-segment";
import "./output-panel.css";
import "./rich-output/rich-output.css";

interface Props {
  result: ExecutionResult | null;
  running: boolean;
  maximized: boolean;
  onToggleMaximize: () => void;
}

function SegmentRenderer(props: { segment: OutputSegment }) {
  return (
    <Switch fallback={<RichTextSegment content={props.segment.content} />}>
      <Match when={props.segment.type === "text"}>
        <RichTextSegment content={props.segment.content} />
      </Match>
      <Match when={props.segment.type === "html"}>
        <RichHtmlSegment content={props.segment.content} />
      </Match>
      <Match when={props.segment.type === "svg"}>
        <RichSvgSegment content={props.segment.content} />
      </Match>
      <Match when={props.segment.type === "image"}>
        <RichImageSegment content={props.segment.content} />
      </Match>
      <Match when={props.segment.type === "chart"}>
        <RichChartSegment content={props.segment.content} />
      </Match>
    </Switch>
  );
}

export function OutputPanel(props: Props) {
  const segments = () => {
    if (!props.result?.stdout) return [];
    return parseOutput(props.result.stdout);
  };

  return (
    <div class="output-panel">
      <div class="output-panel-header">
        <span>Output</span>
        <div class="output-panel-header-actions">
          <Show when={props.result}>
            <span class="output-panel-time">
              {props.result!.execution_time_ms.toFixed(1)}ms
            </span>
          </Show>
          <button
            class="panel-maximize-btn"
            onClick={props.onToggleMaximize}
            title={props.maximized ? "Restore" : "Maximize"}
          >
            {props.maximized ? "\u29C9" : "\u2922"}
          </button>
        </div>
      </div>
      <div class="output-panel-body">
        <Show when={props.running}>
          <p class="output-panel-running">Running...</p>
        </Show>
        <Show when={!props.running && !props.result}>
          <p class="output-panel-empty">Run your code to see output here.</p>
        </Show>
        <Show when={!props.running && props.result}>
          <Show when={segments().length > 0}>
            <div class="output-panel-segments">
              <For each={segments()}>
                {(segment) => <SegmentRenderer segment={segment} />}
              </For>
            </div>
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
