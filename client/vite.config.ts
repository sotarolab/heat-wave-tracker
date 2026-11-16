import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Served by the FastAPI app at the root (service/app.py mounts client/dist).
// In development, /api is proxied to the Python service on port 8051.
export default defineConfig({
  plugins: [react()],
  base: '/',
  server: { proxy: { '/api': { target: 'http://localhost:8051', changeOrigin: true } } },
  build: { outDir: 'dist', emptyOutDir: true },
})
