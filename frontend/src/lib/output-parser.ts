export type OutputSegmentType = "text" | "html" | "svg" | "image" | "chart";

export interface OutputSegment {
  type: OutputSegmentType;
  content: string;
}

const RICH_PATTERN = /<!--NLP_RICH:(\w+)-->\n?([\s\S]*?)<!--\/NLP_RICH-->/g;
const VALID_TYPES = new Set<OutputSegmentType>(["html", "svg", "image", "chart"]);

export function parseOutput(stdout: string): OutputSegment[] {
  const segments: OutputSegment[] = [];
  let lastIndex = 0;

  for (const match of stdout.matchAll(RICH_PATTERN)) {
    // Plain text before this marker
    if (match.index! > lastIndex) {
      const text = stdout.slice(lastIndex, match.index!);
      if (text.trim()) {
        segments.push({ type: "text", content: text });
      }
    }

    const richType = match[1] as OutputSegmentType;
    const richContent = match[2].trim();

    if (VALID_TYPES.has(richType)) {
      segments.push({ type: richType, content: richContent });
    } else {
      // Unknown type — treat as plain text
      segments.push({ type: "text", content: match[0] });
    }

    lastIndex = match.index! + match[0].length;
  }

  // Trailing plain text
  if (lastIndex < stdout.length) {
    const text = stdout.slice(lastIndex);
    if (text.trim()) {
      segments.push({ type: "text", content: text });
    }
  }

  return segments;
}
