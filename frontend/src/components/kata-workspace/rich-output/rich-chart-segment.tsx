import { createMemo, createEffect, onCleanup, Show } from "solid-js";
import { useTheme } from "../../../context/theme-context";
import { renderChart, type AnyChartSpec } from "../../../lib/chart-renderer";

export function RichChartSegment(props: { content: string }) {
  let canvasEl!: HTMLCanvasElement;
  const { theme } = useTheme();

  const spec = createMemo((): AnyChartSpec | null => {
    try {
      return JSON.parse(props.content);
    } catch {
      return null;
    }
  });

  createEffect(() => {
    const s = spec();
    const t = theme();
    if (s && canvasEl) {
      renderChart(canvasEl, s, t);
    }
  });

  onCleanup(() => {
    // Chart.js cleanup is handled by renderChart destroying previous instance
  });

  return (
    <Show
      when={spec()}
      fallback={
        <pre class="output-panel-error">
          Invalid chart JSON: {props.content.slice(0, 200)}
        </pre>
      }
    >
      <div class="rich-output-chart">
        <canvas ref={canvasEl!} width={600} height={350} />
      </div>
    </Show>
  );
}
