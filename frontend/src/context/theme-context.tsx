import {
  createContext,
  createSignal,
  useContext,
  onMount,
  type ParentProps,
} from "solid-js";

type Theme = "light" | "dark";

interface ThemeContextValue {
  theme: () => Theme;
  toggle: () => void;
}

const ThemeContext = createContext<ThemeContextValue>();

export function ThemeProvider(props: ParentProps) {
  const [theme, setTheme] = createSignal<Theme>("light");

  onMount(() => {
    const saved = localStorage.getItem("nlp-katas-theme") as Theme | null;
    const preferred =
      saved ||
      (window.matchMedia("(prefers-color-scheme: dark)").matches
        ? "dark"
        : "light");
    setTheme(preferred);
    document.documentElement.classList.toggle("dark", preferred === "dark");
  });

  const toggle = () => {
    const next = theme() === "light" ? "dark" : "light";
    setTheme(next);
    localStorage.setItem("nlp-katas-theme", next);
    document.documentElement.classList.toggle("dark", next === "dark");
  };

  return (
    <ThemeContext.Provider value={{ theme, toggle }}>
      {props.children}
    </ThemeContext.Provider>
  );
}

export function useTheme(): ThemeContextValue {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error("useTheme must be used within ThemeProvider");
  return ctx;
}
