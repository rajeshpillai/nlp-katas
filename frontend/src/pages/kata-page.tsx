import { createResource, createSignal, Show } from "solid-js";
import { useParams } from "@solidjs/router";
import { fetchKataContent } from "../lib/api-client";
import { MarkdownContent } from "../components/markdown-content/markdown-content";
import { KataWorkspace } from "../components/kata-workspace/kata-workspace";
import "./kata-page.css";

type Tab = "concept" | "code";

export function KataPage() {
  const params = useParams<{ trackId: string; phaseId: string; kataId: string }>();
  const [activeTab, setActiveTab] = createSignal<Tab>("concept");

  const [content] = createResource(
    () => ({ trackId: params.trackId, phaseId: Number(params.phaseId), kataId: params.kataId }),
    (args) => fetchKataContent(args.trackId, args.phaseId, args.kataId),
  );

  return (
    <div class="kata-page">
      <div class="kata-tabs">
        <button
          class="kata-tab"
          classList={{ "kata-tab--active": activeTab() === "concept" }}
          onClick={() => setActiveTab("concept")}
        >
          Concept
        </button>
        <button
          class="kata-tab"
          classList={{ "kata-tab--active": activeTab() === "code" }}
          onClick={() => setActiveTab("code")}
        >
          Code
        </button>
      </div>
      <div class="kata-content">
        <Show when={content.loading}>
          <p class="kata-loading">Loading...</p>
        </Show>
        <Show when={content.error}>
          <p class="kata-error">Failed to load kata content.</p>
        </Show>
        <Show when={content()}>
          <Show when={activeTab() === "concept"}>
            <MarkdownContent markdown={content()!} />
          </Show>
          <Show when={activeTab() === "code"}>
            <KataWorkspace
              markdown={content()!}
              kataId={params.kataId}
            />
          </Show>
        </Show>
      </div>
    </div>
  );
}
