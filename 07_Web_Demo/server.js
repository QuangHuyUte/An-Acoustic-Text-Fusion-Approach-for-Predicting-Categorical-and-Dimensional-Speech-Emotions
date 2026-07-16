const http = require('http');
const fs = require('fs');
const path = require('path');
const os = require('os');
const { spawn } = require('child_process');

const PORT = Number(process.env.PORT || 5174);
const PUBLIC_DIR = path.join(__dirname, 'public');
const BACKEND_SCRIPT = path.join(__dirname, 'backend', '03b_infer.py');
const BACKEND_SERVICE_SCRIPT = path.join(__dirname, 'backend', '03b_service.py');
const MODEL_REGISTRY_PATH = path.join(__dirname, 'models', 'live_model_registry.json');
const MAX_JSON_BODY_BYTES = Number(process.env.MAX_JSON_BODY_BYTES || 35 * 1024 * 1024);
const MODEL_TIMEOUT_MS = Number(process.env.MODEL_03B_TIMEOUT_MS || 300000);
const USE_MODEL_SERVICE = process.env.USE_MODEL_03B_SERVICE !== '0';
const MODEL_SERVICE_HOST = process.env.MODEL_03B_SERVICE_HOST || '127.0.0.1';
const MODEL_SERVICE_PORT = Number(process.env.MODEL_03B_SERVICE_PORT || 8765);
const MODEL_SERVICE_URL = `http://${MODEL_SERVICE_HOST}:${MODEL_SERVICE_PORT}`;

let modelServiceProcess = null;
let modelServiceStarting = null;

const mimeTypes = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.webp': 'image/webp',
  '.wav': 'audio/wav',
  '.mp3': 'audio/mpeg'
};

function sendJson(res, statusCode, payload) {
  res.writeHead(statusCode, { 'Content-Type': 'application/json; charset=utf-8' });
  res.end(JSON.stringify(payload));
}

function readJsonBody(req, maxBytes = MAX_JSON_BODY_BYTES) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    req.on('data', (chunk) => {
      size += chunk.length;
      if (size > maxBytes) {
        reject(new Error(`Request body too large. Limit is ${Math.round(maxBytes / 1024 / 1024)} MB.`));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on('end', () => {
      try {
        const raw = Buffer.concat(chunks).toString('utf8');
        resolve(raw ? JSON.parse(raw) : {});
      } catch (error) {
        reject(new Error(`Invalid JSON payload: ${error.message}`));
      }
    });
    req.on('error', reject);
  });
}

function safeFileStem(value) {
  return String(value || `speech-${Date.now()}`)
    .replace(/[^a-zA-Z0-9_-]/g, '_')
    .slice(0, 80);
}

function findPythonCommand() {
  return process.env.PYTHON || process.env.PYTHON_EXE || 'python';
}

function parseBackendJson(stdout) {
  const lines = stdout
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  for (let index = lines.length - 1; index >= 0; index -= 1) {
    try {
      return JSON.parse(lines[index]);
    } catch (_error) {
      // Model libraries may print progress before the final JSON line.
    }
  }
  throw new Error('03B backend did not return JSON.');
}

function readModelRegistry() {
  try {
    const raw = fs.readFileSync(MODEL_REGISTRY_PATH, 'utf8');
    return JSON.parse(raw);
  } catch (error) {
    return {
      version: 'missing',
      defaultProfileId: '03b_frozen_text_5fold',
      profiles: [],
      error: error.message
    };
  }
}

function httpJsonRequest({ method = 'GET', requestPath = '/', payload = undefined, timeoutMs = 5000 }) {
  return new Promise((resolve, reject) => {
    const raw = payload === undefined ? null : Buffer.from(JSON.stringify(payload), 'utf8');
    const req = http.request({
      hostname: MODEL_SERVICE_HOST,
      port: MODEL_SERVICE_PORT,
      path: requestPath,
      method,
      headers: raw ? {
        'Content-Type': 'application/json; charset=utf-8',
        'Content-Length': String(raw.length)
      } : {}
    }, (res) => {
      const chunks = [];
      res.on('data', (chunk) => chunks.push(chunk));
      res.on('end', () => {
        const body = Buffer.concat(chunks).toString('utf8');
        try {
          const json = body ? JSON.parse(body) : {};
          if (res.statusCode >= 400) {
            reject(new Error(json.error || json.message || `Model service returned HTTP ${res.statusCode}.`));
            return;
          }
          resolve(json);
        } catch (error) {
          reject(new Error(`Invalid model service response: ${error.message}`));
        }
      });
    });

    req.setTimeout(timeoutMs, () => {
      req.destroy(new Error(`Model service request timed out after ${Math.round(timeoutMs / 1000)} seconds.`));
    });
    req.on('error', reject);
    if (raw) req.write(raw);
    req.end();
  });
}

async function checkModelService(timeoutMs = 1000) {
  return httpJsonRequest({ requestPath: '/health', timeoutMs });
}

async function waitForModelService(timeoutMs = 20000) {
  const deadline = Date.now() + timeoutMs;
  let lastError = null;
  while (Date.now() < deadline) {
    try {
      return await checkModelService(1000);
    } catch (error) {
      lastError = error;
      await new Promise((resolve) => setTimeout(resolve, 500));
    }
  }
  throw lastError || new Error('Model service did not become ready.');
}

async function startModelService() {
  if (!USE_MODEL_SERVICE) return null;
  if (!fs.existsSync(BACKEND_SERVICE_SCRIPT)) {
    throw new Error(`Missing backend service script: ${BACKEND_SERVICE_SCRIPT}`);
  }
  if (modelServiceStarting) return modelServiceStarting;

  modelServiceStarting = (async () => {
    try {
      return await checkModelService(800);
    } catch (_existingServiceError) {
      // No existing service answered; start the bundled persistent backend.
    }

    const python = findPythonCommand();
    modelServiceProcess = spawn(python, [
      BACKEND_SERVICE_SCRIPT,
      '--host',
      MODEL_SERVICE_HOST,
      '--port',
      String(MODEL_SERVICE_PORT),
      '--prewarm'
    ], {
      cwd: __dirname,
      windowsHide: true,
      env: {
        ...process.env,
        PYTHONIOENCODING: 'utf-8',
        MODEL_03B_VERBOSE: process.env.MODEL_03B_VERBOSE || '0',
        MODEL_03B_SERVICE_LOG: process.env.MODEL_03B_SERVICE_LOG || '0'
      }
    });

    modelServiceProcess.stdout.on('data', (chunk) => {
      process.stdout.write(`[03B service] ${chunk.toString('utf8')}`);
    });
    modelServiceProcess.stderr.on('data', (chunk) => {
      process.stderr.write(`[03B service] ${chunk.toString('utf8')}`);
    });
    modelServiceProcess.on('exit', (code) => {
      if (modelServiceProcess) {
        console.log(`03B model service exited with code ${code}.`);
      }
      modelServiceProcess = null;
      modelServiceStarting = null;
    });

    return waitForModelService(30000);
  })();

  return modelServiceStarting;
}

function stopModelService() {
  if (modelServiceProcess) {
    const child = modelServiceProcess;
    modelServiceProcess = null;
    child.kill();
  }
}

function run03BInferenceViaCli({ audioPath, transcript, modelProfileId, autoTranscribe = true }) {
  return new Promise((resolve, reject) => {
    if (!fs.existsSync(BACKEND_SCRIPT)) {
      reject(new Error(`Missing backend script: ${BACKEND_SCRIPT}`));
      return;
    }

    const python = findPythonCommand();
    const args = [
      BACKEND_SCRIPT,
      '--audio',
      audioPath,
      '--transcript',
      transcript || '',
      '--model-profile',
      modelProfileId || '03b_frozen_text_5fold'
    ];
    if (autoTranscribe === false) args.push('--no-auto-transcribe');
    const child = spawn(python, args, {
      cwd: __dirname,
      windowsHide: true,
      env: {
        ...process.env,
        PYTHONIOENCODING: 'utf-8'
      }
    });

    let stdout = '';
    let stderr = '';
    let settled = false;
    const timer = setTimeout(() => {
      if (settled) return;
      settled = true;
      child.kill('SIGKILL');
      reject(new Error(`03B inference timed out after ${Math.round(MODEL_TIMEOUT_MS / 1000)} seconds.`));
    }, MODEL_TIMEOUT_MS);

    child.stdout.on('data', (chunk) => {
      stdout += chunk.toString('utf8');
    });
    child.stderr.on('data', (chunk) => {
      stderr += chunk.toString('utf8');
    });
    child.on('error', (error) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      reject(error);
    });
    child.on('close', (code) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      try {
        const result = parseBackendJson(stdout);
        if (code !== 0 || result.ok === false) {
          reject(new Error(result.error || stderr || `03B backend exited with code ${code}.`));
          return;
        }
        resolve({
          ...result,
          service: {
            ...(result.service || {}),
            mode: 'one-shot-python-cli'
          },
          backendLogs: stderr ? stderr.slice(-2000) : undefined
        });
      } catch (error) {
        reject(new Error(`${error.message}${stderr ? ` Backend stderr: ${stderr.slice(-2000)}` : ''}`));
      }
    });
  });
}

async function run03BInference(payload) {
  if (USE_MODEL_SERVICE) {
    await startModelService();
    return httpJsonRequest({
      method: 'POST',
      requestPath: '/analyze',
      payload: {
        audioWavBase64: payload.audioWavBase64,
        transcript: payload.transcript || '',
        sessionId: payload.sessionId || '',
        autoTranscribe: payload.autoTranscribe !== false,
        modelProfileId: payload.modelProfileId || '03b_frozen_text_5fold'
      },
      timeoutMs: MODEL_TIMEOUT_MS
    });
  }

  const tmpDir = path.join(os.tmpdir(), 'speech-demo-03b');
  fs.mkdirSync(tmpDir, { recursive: true });
  const audioPath = path.join(tmpDir, `${safeFileStem(payload.sessionId)}.wav`);
  fs.writeFileSync(audioPath, Buffer.from(payload.audioWavBase64, 'base64'));
  try {
    return await run03BInferenceViaCli({
      audioPath,
      transcript: payload.transcript || '',
      modelProfileId: payload.modelProfileId || '03b_frozen_text_5fold',
      autoTranscribe: payload.autoTranscribe !== false
    });
  } finally {
    fs.unlink(audioPath, () => {});
  }
}

const server = http.createServer((req, res) => {
  if (req.method === 'GET' && req.url === '/api/health') {
    checkModelService(800)
      .then((serviceHealth) => {
        sendJson(res, 200, {
          ok: true,
          app: 'Speech Emotion Presentation Feedback Demo',
          mode: 'browser-audio-analysis-with-persistent-03b-backend',
          modelIntegration: fs.existsSync(BACKEND_SERVICE_SCRIPT) ? '03B persistent backend service configured' : '03B backend service missing',
          modelService: serviceHealth,
          timeoutSeconds: Math.round(MODEL_TIMEOUT_MS / 1000)
        });
      })
      .catch((error) => {
        sendJson(res, 200, {
          ok: true,
          app: 'Speech Emotion Presentation Feedback Demo',
          mode: 'browser-audio-analysis-with-03b-backend',
          modelIntegration: fs.existsSync(BACKEND_SCRIPT) ? '03B one-shot backend script configured' : '03B backend script missing',
          modelService: {
            ok: false,
            url: MODEL_SERVICE_URL,
            message: error.message
          },
          timeoutSeconds: Math.round(MODEL_TIMEOUT_MS / 1000)
        });
      });
    return;
  }

  if (req.method === 'GET' && req.url === '/api/model-profiles') {
    const registry = readModelRegistry();
    sendJson(res, 200, {
      ok: true,
      registry
    });
    return;
  }

  if (req.method === 'POST' && req.url === '/api/analyze-03b') {
    readJsonBody(req)
      .then(async (payload) => {
        const audioWavBase64 = payload.audioWavBase64 || payload.audio || '';
        if (!audioWavBase64) {
          sendJson(res, 400, {
            ok: false,
            source: '03B backend',
            message: 'Missing audioWavBase64 in request payload.'
          });
          return;
        }

        let result;
        try {
          result = await run03BInference({
            audioWavBase64,
            transcript: payload.transcript || '',
            sessionId: payload.sessionId || '',
            autoTranscribe: payload.autoTranscribe !== false,
            modelProfileId: payload.modelProfileId || '03b_frozen_text_5fold'
          });
        } catch (serviceError) {
          const tmpDir = path.join(os.tmpdir(), 'speech-demo-03b');
          fs.mkdirSync(tmpDir, { recursive: true });
          const audioPath = path.join(tmpDir, `${safeFileStem(payload.sessionId)}.wav`);
          fs.writeFileSync(audioPath, Buffer.from(audioWavBase64, 'base64'));
          try {
            result = await run03BInferenceViaCli({
              audioPath,
              transcript: payload.transcript || '',
              modelProfileId: payload.modelProfileId || '03b_frozen_text_5fold',
              autoTranscribe: payload.autoTranscribe !== false
            });
            result.serviceFallback = serviceError.message;
          } finally {
            fs.unlink(audioPath, () => {});
          }
        }

        sendJson(res, 200, {
          ...result,
          received: {
            duration: payload.duration,
            sampleRate: payload.sampleRate,
            hasTranscript: Boolean(payload.transcript),
            modelProfileId: payload.modelProfileId || '03b_frozen_text_5fold'
          }
        });
      })
      .catch((error) => {
        sendJson(res, 503, {
          ok: false,
          source: '03B backend unavailable',
          message: error.message,
          fallback: 'Frontend will use browser-side analysis for this demo run.'
        });
      });
    return;
  }

  let requestPath = decodeURIComponent(req.url.split('?')[0]);
  if (requestPath === '/') requestPath = '/index.html';

  const filePath = path.normalize(path.join(PUBLIC_DIR, requestPath));
  if (!filePath.startsWith(PUBLIC_DIR)) {
    res.writeHead(403, { 'Content-Type': 'text/plain; charset=utf-8' });
    res.end('Forbidden');
    return;
  }

  fs.readFile(filePath, (err, data) => {
    if (err) {
      res.writeHead(404, { 'Content-Type': 'text/html; charset=utf-8' });
      res.end('<h1>404</h1><p>Demo asset not found.</p>');
      return;
    }

    const ext = path.extname(filePath).toLowerCase();
    res.writeHead(200, {
      'Content-Type': mimeTypes[ext] || 'application/octet-stream',
      'Cache-Control': 'no-store'
    });
    res.end(data);
  });
});

server.listen(PORT, () => {
  console.log(`Speech demo running at http://localhost:${PORT}`);
  if (USE_MODEL_SERVICE) {
    startModelService()
      .then(() => httpJsonRequest({ method: 'POST', requestPath: '/prewarm', timeoutMs: MODEL_TIMEOUT_MS }).catch(() => null))
      .catch((error) => {
        console.warn(`03B persistent service is not ready yet: ${error.message}`);
      });
  }
});

process.on('SIGINT', () => {
  stopModelService();
  process.exit(0);
});
process.on('SIGTERM', () => {
  stopModelService();
  process.exit(0);
});
process.on('exit', stopModelService);
