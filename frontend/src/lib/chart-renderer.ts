import {
  Chart,
  BarController,
  LineController,
  ScatterController,
  PieController,
  DoughnutController,
  BarElement,
  LineElement,
  PointElement,
  ArcElement,
  CategoryScale,
  LinearScale,
  Title,
  Tooltip,
  Legend,
} from "chart.js";

Chart.register(
  BarController, LineController, ScatterController,
  PieController, DoughnutController,
  BarElement, LineElement, PointElement, ArcElement,
  CategoryScale, LinearScale,
  Title, Tooltip, Legend,
);

const PALETTE = [
  "#3b82f6", "#ef4444", "#10b981", "#f59e0b",
  "#8b5cf6", "#ec4899", "#06b6d4", "#84cc16",
];

export interface ChartDataset {
  label: string;
  data: number[] | Array<{ x: number; y: number }>;
  color?: string;
  borderColor?: string;
}

export interface ChartSpec {
  type: "bar" | "line" | "scatter" | "pie" | "doughnut";
  title?: string;
  labels?: string[];
  datasets: ChartDataset[];
  options?: {
    x_label?: string;
    y_label?: string;
    stacked?: boolean;
  };
}

export interface HeatmapSpec {
  type: "heatmap";
  title?: string;
  x_labels: string[];
  y_labels: string[];
  data: number[][];
  color_scale?: string;
}

export type AnyChartSpec = ChartSpec | HeatmapSpec;

export function renderChart(
  canvas: HTMLCanvasElement,
  spec: AnyChartSpec,
  theme: "light" | "dark",
): void {
  const existing = Chart.getChart(canvas);
  if (existing) existing.destroy();

  if (spec.type === "heatmap") {
    renderHeatmap(canvas, spec as HeatmapSpec, theme);
    return;
  }

  const textColor = theme === "dark" ? "#cdd6f4" : "#1a1a2e";
  const gridColor = theme === "dark" ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)";

  const chartSpec = spec as ChartSpec;
  const datasets = chartSpec.datasets.map((ds, i) => ({
    label: ds.label,
    data: ds.data,
    backgroundColor: ds.color || PALETTE[i % PALETTE.length],
    borderColor: ds.borderColor || ds.color || PALETTE[i % PALETTE.length],
    borderWidth: chartSpec.type === "line" ? 2 : 1,
    pointRadius: chartSpec.type === "scatter" ? 4 : 3,
  }));

  const isPieType = chartSpec.type === "pie" || chartSpec.type === "doughnut";

  new Chart(canvas, {
    type: chartSpec.type,
    data: { labels: chartSpec.labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      plugins: {
        title: chartSpec.title
          ? { display: true, text: chartSpec.title, color: textColor, font: { size: 14 } }
          : undefined,
        legend: { labels: { color: textColor } },
        tooltip: { enabled: true },
      },
      scales: !isPieType
        ? {
            x: {
              ticks: { color: textColor },
              grid: { color: gridColor },
              stacked: chartSpec.options?.stacked,
              title: chartSpec.options?.x_label
                ? { display: true, text: chartSpec.options.x_label, color: textColor }
                : undefined,
            },
            y: {
              ticks: { color: textColor },
              grid: { color: gridColor },
              stacked: chartSpec.options?.stacked,
              title: chartSpec.options?.y_label
                ? { display: true, text: chartSpec.options.y_label, color: textColor }
                : undefined,
            },
          }
        : undefined,
    },
  });
}

function renderHeatmap(
  canvas: HTMLCanvasElement,
  spec: HeatmapSpec,
  theme: "light" | "dark",
): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const rows = spec.data.length;
  const cols = spec.data[0]?.length || 0;
  if (rows === 0 || cols === 0) return;

  const textColor = theme === "dark" ? "#cdd6f4" : "#1a1a2e";
  const labelFont = "11px JetBrains Mono, monospace";

  // Measure label widths
  ctx.font = labelFont;
  const yLabelWidth = Math.max(...spec.y_labels.map((l) => ctx.measureText(l).width)) + 10;
  const titleHeight = spec.title ? 28 : 0;
  const xLabelHeight = 20;

  const cellW = Math.max(40, Math.min(60, (canvas.width - yLabelWidth - 20) / cols));
  const cellH = Math.max(28, Math.min(40, (canvas.height - titleHeight - xLabelHeight - 20) / rows));

  // Resize canvas to fit
  canvas.width = yLabelWidth + cols * cellW + 20;
  canvas.height = titleHeight + rows * cellH + xLabelHeight + 20;

  const dpr = window.devicePixelRatio || 1;
  canvas.style.width = canvas.width + "px";
  canvas.style.height = canvas.height + "px";
  canvas.width *= dpr;
  canvas.height *= dpr;
  ctx.scale(dpr, dpr);

  const drawW = canvas.width / dpr;
  const drawH = canvas.height / dpr;

  // Find data range
  let minVal = Infinity, maxVal = -Infinity;
  for (const row of spec.data) {
    for (const v of row) {
      if (v < minVal) minVal = v;
      if (v > maxVal) maxVal = v;
    }
  }
  const range = maxVal - minVal || 1;

  // Title
  if (spec.title) {
    ctx.fillStyle = textColor;
    ctx.font = "bold 14px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(spec.title, drawW / 2, 18);
  }

  const startX = yLabelWidth;
  const startY = titleHeight;

  // Draw cells
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const val = spec.data[r][c];
      const norm = (val - minVal) / range;

      // Blue color scale
      const intensity = Math.round(norm * 200 + 55);
      ctx.fillStyle =
        spec.color_scale === "red"
          ? `rgba(${intensity}, ${55}, ${55}, 0.9)`
          : `rgba(${55}, ${Math.round(norm * 100 + 80)}, ${intensity}, 0.9)`;

      ctx.fillRect(startX + c * cellW, startY + r * cellH, cellW - 1, cellH - 1);

      // Cell value text
      ctx.fillStyle = norm > 0.5 ? "#ffffff" : textColor;
      ctx.font = labelFont;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(
        val.toFixed(2),
        startX + c * cellW + cellW / 2,
        startY + r * cellH + cellH / 2,
      );
    }
  }

  // Y-axis labels
  ctx.fillStyle = textColor;
  ctx.font = labelFont;
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let r = 0; r < rows; r++) {
    ctx.fillText(spec.y_labels[r], startX - 6, startY + r * cellH + cellH / 2);
  }

  // X-axis labels
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (let c = 0; c < cols; c++) {
    ctx.fillText(spec.x_labels[c], startX + c * cellW + cellW / 2, startY + rows * cellH + 4);
  }
}
