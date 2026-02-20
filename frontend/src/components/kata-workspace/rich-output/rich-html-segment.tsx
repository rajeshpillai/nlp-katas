import { createMemo } from "solid-js";
import { sanitizeHtml } from "../../../lib/html-sanitizer";

export function RichHtmlSegment(props: { content: string }) {
  const safe = createMemo(() => sanitizeHtml(props.content));
  return <div class="rich-output-html" innerHTML={safe()} />;
}
