import { type Resource } from "solid-js";
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

  return (
    <div class="main-layout">
      <Sidebar trackId={props.trackId} katasData={props.katasData} />
      <div class="main-layout-content">
        <header class="main-layout-header">
          <button class="main-layout-home" onClick={() => navigate("/")}>
            NLP Katas
          </button>
          <ThemeToggle />
        </header>
        <main class="main-layout-main">
          {props.children}
        </main>
      </div>
    </div>
  );
}
