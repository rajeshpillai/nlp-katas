import { useTheme } from "../../context/theme-context";
import "./theme-toggle.css";

export function ThemeToggle() {
  const { theme, toggle } = useTheme();

  return (
    <button class="theme-toggle" onClick={toggle} aria-label="Toggle theme">
      {theme() === "light" ? "\u263E" : "\u2600"}
    </button>
  );
}
