import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In dev the API and WebSocket are proxied to the FastAPI server so the
// browser only ever talks to one origin (no CORS, no mixed-origin WS).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://localhost:8000", changeOrigin: true },
      "/ws": { target: "ws://localhost:8000", ws: true },
    },
  },
  build: { outDir: "dist", sourcemap: false },
});
