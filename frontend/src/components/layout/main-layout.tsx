import { createSignal, type Resource } from "solid-js";
import { useNavigate } from "@solidjs/router";
import type { KatasResponse } from "../../lib/api-client";
import { Sidebar } from "./sidebar";
import { ThemeToggle } from "./theme-toggle";
import "./main-layout.css";

interface Props {
  trackId: string;
  katasData: Resource<KatasResponse>;
  children?: any;
}

export function MainLayout(props: Props) {
  const navigate = useNavigate();
  const [sidebarOpen, setSidebarOpen] = createSignal(true);

  return (
    <div class="main-layout">
      <div
        class="main-layout-sidebar"
        classList={{ "main-layout-sidebar--collapsed": !sidebarOpen() }}
      >
        <Sidebar trackId={props.trackId} katasData={props.katasData} />
      </div>
      <div class="main-layout-content">
        <header class="main-layout-header">
          <div class="main-layout-header-left">
            <button
              class="main-layout-menu-btn"
              onClick={() => setSidebarOpen((prev) => !prev)}
              aria-label={sidebarOpen() ? "Close sidebar" : "Open sidebar"}
            >
              {sidebarOpen() ? "\u2715" : "\u2630"}
            </button>
            <button class="main-layout-home" onClick={() => navigate("/")}>
              NLP Katas
            </button>
          </div>
          <ThemeToggle />
        </header>
        <main class="main-layout-main">
          {props.children}
        </main>
      </div>
    </div>
  );
}
