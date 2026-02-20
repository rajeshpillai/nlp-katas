import { For, Show, createSignal, type Resource } from "solid-js";
import { useNavigate, useParams } from "@solidjs/router";
import type { KatasResponse } from "../../lib/api-client";
import { kataPath } from "../../routes";
import "./sidebar.css";

interface Props {
  trackId: string;
  katasData: Resource<KatasResponse>;
}

export function Sidebar(props: Props) {
  const params = useParams();
  const navigate = useNavigate();
  const [expandedPhases, setExpandedPhases] = createSignal<Set<number>>(new Set([0]));

  const togglePhase = (phase: number) => {
    setExpandedPhases((prev) => {
      const next = new Set(prev);
      if (next.has(phase)) next.delete(phase);
      else next.add(phase);
      return next;
    });
  };

  const katasByPhase = () => {
    const data = props.katasData();
    if (!data) return {};
    const grouped: Record<number, typeof data.katas> = {};
    for (const kata of data.katas) {
      if (!grouped[kata.phase]) grouped[kata.phase] = [];
      grouped[kata.phase].push(kata);
    }
    return grouped;
  };

  const phases = () => props.katasData()?.phases ?? {};

  return (
    <nav class="sidebar">
      <Show when={props.katasData()} fallback={<p class="sidebar-loading">Loading...</p>}>
        <For each={Object.entries(phases()).sort(([a], [b]) => Number(a) - Number(b))}>
          {([phaseNum, phaseName]) => {
            const phase = Number(phaseNum);
            const katas = () => katasByPhase()[phase] ?? [];
            const isExpanded = () => expandedPhases().has(phase);

            return (
              <div class="sidebar-phase">
                <button
                  class="sidebar-phase-header"
                  onClick={() => togglePhase(phase)}
                >
                  <span class="sidebar-phase-arrow" classList={{ "sidebar-phase-arrow--open": isExpanded() }}>
                    &#9654;
                  </span>
                  <span class="sidebar-phase-label">
                    Phase {phase}
                  </span>
                  <span class="sidebar-phase-name">{phaseName as string}</span>
                </button>
                <Show when={isExpanded()}>
                  <ul class="sidebar-katas">
                    <For each={katas()}>
                      {(kata) => (
                        <li>
                          <button
                            class="sidebar-kata"
                            classList={{ "sidebar-kata--active": params.kataId === kata.id }}
                            onClick={() => navigate(kataPath(props.trackId, kata.phase, kata.id))}
                          >
                            {kata.title}
                          </button>
                        </li>
                      )}
                    </For>
                  </ul>
                </Show>
              </div>
            );
          }}
        </For>
      </Show>
    </nav>
  );
}
