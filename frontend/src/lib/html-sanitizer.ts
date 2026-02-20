const ALLOWED_TAGS = new Set([
  "table", "thead", "tbody", "tfoot", "tr", "th", "td", "caption",
  "div", "span", "p", "br", "hr",
  "strong", "em", "b", "i", "u", "code", "pre",
  "ul", "ol", "li",
  "h1", "h2", "h3", "h4", "h5", "h6",
]);

const ALLOWED_ATTRS = new Set([
  "class", "style", "colspan", "rowspan", "title",
]);

const SVG_TAGS = new Set([
  "svg", "g", "rect", "circle", "ellipse", "line", "polyline",
  "polygon", "path", "text", "tspan", "defs", "linearGradient",
  "radialGradient", "stop", "clipPath", "mask", "use", "title",
]);

const SVG_ATTRS = new Set([
  "viewBox", "width", "height", "x", "y", "cx", "cy", "r", "rx", "ry",
  "d", "fill", "stroke", "stroke-width", "stroke-dasharray",
  "transform", "opacity", "font-size", "font-family", "text-anchor",
  "dominant-baseline", "class", "style", "id", "points",
  "x1", "y1", "x2", "y2", "offset", "stop-color", "stop-opacity",
  "gradientUnits", "gradientTransform", "xmlns",
]);

function sanitizeNode(
  node: Node,
  allowedTags: Set<string>,
  allowedAttrs: Set<string>,
): void {
  const toRemove: Node[] = [];

  for (const child of Array.from(node.childNodes)) {
    if (child.nodeType === Node.ELEMENT_NODE) {
      const el = child as Element;
      const tag = el.tagName.toLowerCase();

      if (!allowedTags.has(tag)) {
        toRemove.push(child);
        continue;
      }

      // Strip disallowed attributes and event handlers
      for (const attr of Array.from(el.attributes)) {
        if (!allowedAttrs.has(attr.name) || attr.name.startsWith("on")) {
          el.removeAttribute(attr.name);
        }
      }

      // Strip javascript: from style values
      if (el.hasAttribute("style")) {
        const style = el.getAttribute("style") || "";
        if (style.toLowerCase().includes("javascript:") || style.toLowerCase().includes("expression(")) {
          el.removeAttribute("style");
        }
      }

      sanitizeNode(child, allowedTags, allowedAttrs);
    }
  }

  for (const n of toRemove) {
    n.parentNode?.removeChild(n);
  }
}

export function sanitizeHtml(html: string): string {
  const doc = new DOMParser().parseFromString(html, "text/html");
  sanitizeNode(doc.body, ALLOWED_TAGS, ALLOWED_ATTRS);
  return doc.body.innerHTML;
}

export function sanitizeSvg(svg: string): string {
  const doc = new DOMParser().parseFromString(svg, "image/svg+xml");
  const allTags = new Set([...SVG_TAGS]);
  const allAttrs = new Set([...SVG_ATTRS]);
  sanitizeNode(doc.documentElement, allTags, allAttrs);
  return new XMLSerializer().serializeToString(doc.documentElement);
}
