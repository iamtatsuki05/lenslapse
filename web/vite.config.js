import { existsSync } from 'node:fs'
import { defineConfig } from 'vite'
import serveStatic from 'serve-static'

// Dev-only: serve converted checkpoints from a local directory under /models/ so the live probe
// works without uploading weights anywhere. Production deployments point the app at a HF Hub repo
// instead (see src/live.js HF_DEFAULT), so model files are never part of the built site.
const MODELS_DIR = process.env.LENSLAPSE_MODELS_DIR

function devModels() {
  return {
    name: 'lenslapse-dev-models',
    configureServer(server) {
      if (MODELS_DIR && existsSync(MODELS_DIR)) {
        server.middlewares.use('/models', serveStatic(MODELS_DIR, { fallthrough: false }))
      }
    },
    configurePreviewServer(server) {
      if (MODELS_DIR && existsSync(MODELS_DIR)) {
        server.middlewares.use('/models', serveStatic(MODELS_DIR, { fallthrough: false }))
      }
    },
  }
}

// base './' so the built site works from any path (GitHub Pages project sites live under /<repo>/).
export default defineConfig({
  base: './',
  plugins: [devModels()],
  // onnxruntime-web must load its own .wasm/.mjs assets at runtime; pre-bundling breaks the paths.
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
  build: {
    target: 'es2022',
    chunkSizeWarningLimit: 2000,
  },
})
