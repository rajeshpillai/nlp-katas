import { createMemo } from "solid-js";
import "./markdown-content.css";

interface Props {
  markdown: string;
}

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function parseMarkdown(md: string): string {
  let html = md;

  // Code blocks (must be first to protect contents)
  html = html.replace(/```(\w+)?\n([\s\S]*?)```/g, (_match, lang, code) => {
    return `<pre class="md-code-block"><code class="md-code-lang-${lang || "text"}">${escapeHtml(code.trimEnd())}</code></pre>`;
  });

  // Horizontal rules
  html = html.replace(/^---$/gm, '<hr class="md-hr" />');

  // Headers
  html = html.replace(/^#### (.+)$/gm, '<h4 class="md-h4">$1</h4>');
  html = html.replace(/^### (.+)$/gm, '<h3 class="md-h3">$1</h3>');
  html = html.replace(/^## (.+)$/gm, '<h2 class="md-h2">$1</h2>');
  html = html.replace(/^# (.+)$/gm, '<h1 class="md-h1">$1</h1>');

  // Blockquotes
  html = html.replace(/^> (.+)$/gm, '<blockquote class="md-blockquote">$1</blockquote>');

  // Bold and italic
  html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\*(.+?)\*/g, "<em>$1</em>");

  // Inline code (but not inside code blocks)
  html = html.replace(/`([^`]+)`/g, '<code class="md-inline-code">$1</code>');

  // Unordered lists
  html = html.replace(/^- (.+)$/gm, '<li class="md-li">$1</li>');
  html = html.replace(
    /(<li class="md-li">[\s\S]*?<\/li>)/g,
    '<ul class="md-ul">$1</ul>',
  );
  // Merge adjacent ul tags
  html = html.replace(/<\/ul>\s*<ul class="md-ul">/g, "");

  // Ordered lists
  html = html.replace(/^\d+\.\s+(.+)$/gm, '<li class="md-oli">$1</li>');
  html = html.replace(
    /(<li class="md-oli">[\s\S]*?<\/li>)/g,
    '<ol class="md-ol">$1</ol>',
  );
  html = html.replace(/<\/ol>\s*<ol class="md-ol">/g, "");

  // Paragraphs (lines not already wrapped)
  html = html.replace(
    /^(?!<[a-z])((?!^\s*$).+)$/gm,
    '<p class="md-p">$1</p>',
  );

  return html;
}

export function MarkdownContent(props: Props) {
  const rendered = createMemo(() => parseMarkdown(props.markdown));

  return <div class="md-content" innerHTML={rendered()} />;
}
