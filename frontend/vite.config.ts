import { defineConfig } from "vite";
import solidPlugin from "vite-plugin-solid";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [solidPlugin(), tailwindcss()],
  server: {
    port: 3000,
    proxy: {
      "/api/execute/rust-nlp": {
        target: "http://localhost:8001",
        changeOrigin: true,
        rewrite: (path: string) => path.replace("/api/execute/rust-nlp", "/api/execute"),
      },
      "/api/execute/python-nlp": {
        target: "http://localhost:8000",
        changeOrigin: true,
        rewrite: (path: string) => path.replace("/api/execute/python-nlp", "/api/execute"),
      },
      "/api/tracks/python-nlp": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
      "/api": {
        target: "http://localhost:8001",
        changeOrigin: true,
      },
    },
  },
  build: {
    target: "esnext",
  },
});
