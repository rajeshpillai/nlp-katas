import { Router, Route } from "@solidjs/router";
import { ThemeProvider } from "./context/theme-context";
import { Landing } from "./pages/landing";
import { TrackPage } from "./pages/track-page";
import { KataPage } from "./pages/kata-page";
import { ROUTES } from "./routes";

export default function App() {
  return (
    <ThemeProvider>
      <Router>
        <Route path={ROUTES.HOME} component={Landing} />
        <Route path={ROUTES.TRACK} component={TrackPage}>
          <Route path={ROUTES.KATA} component={KataPage} />
          <Route path="/" component={() => <div class="track-placeholder">Select a kata from the sidebar</div>} />
        </Route>
      </Router>
    </ThemeProvider>
  );
}
