import { createMemo } from "solid-js";
import { sanitizeSvg } from "../../../lib/html-sanitizer";

export function RichSvgSegment(props: { content: string }) {
  const safe = createMemo(() => sanitizeSvg(props.content));
  return <div class="rich-output-svg" innerHTML={safe()} />;
}
