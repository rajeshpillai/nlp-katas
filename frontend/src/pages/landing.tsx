import { createResource, For, Show } from "solid-js";
import { useNavigate } from "@solidjs/router";
import { fetchTracks, type Track } from "../lib/api-client";
import { ThemeToggle } from "../components/layout/theme-toggle";
import { trackPath } from "../routes";
import "./landing.css";

export function Landing() {
  const [tracks] = createResource(fetchTracks);
  const navigate = useNavigate();

  return (
    <div class="landing">
      <header class="landing-header">
        <h1 class="landing-title">NLP Katas</h1>
        <p class="landing-subtitle">
          Learn NLP as a representation and modeling discipline
        </p>
        <ThemeToggle />
      </header>
      <main class="landing-tracks">
        <Show when={tracks()} fallback={<p class="landing-loading">Loading tracks...</p>}>
          <For each={tracks()}>
            {(track: Track) => (
              <button
                class="track-card"
                classList={{ "track-card--disabled": track.status !== "active" }}
                disabled={track.status !== "active"}
                onClick={() => navigate(trackPath(track.id))}
              >
                <h2 class="track-card-title">{track.name}</h2>
                <p class="track-card-description">{track.description}</p>
                <Show when={track.status !== "active"}>
                  <span class="track-card-badge">Coming Soon</span>
                </Show>
              </button>
            )}
          </For>
        </Show>
      </main>
    </div>
  );
}
