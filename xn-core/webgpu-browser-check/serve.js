// Static server for caps.html and the harness. Binds all interfaces so a phone
// on the same network can reach it -- but note WebGPU needs a secure context, so
// a LAN address will report "not a secure context". Use `adb reverse` (localhost
// counts as secure) or an https tunnel.
const http = require('http'), fs = require('fs'), path = require('path');
const root = __dirname, port = Number(process.argv[2] || 8734);
http.createServer((req, res) => {
  const p = decodeURIComponent(req.url.split('?')[0]);
  const f = path.join(root, p === '/' ? 'caps.html' : p);
  if (!f.startsWith(root) || !fs.existsSync(f) || fs.statSync(f).isDirectory()) {
    res.writeHead(404); res.end('not found'); return;
  }
  const ext = path.extname(f);
  res.writeHead(200, { 'Content-Type':
    ext === '.html' ? 'text/html' : ext === '.js' ? 'text/javascript'
    : ext === '.wasm' ? 'application/wasm' : ext === '.json' ? 'application/json' : 'text/plain' });
  res.end(fs.readFileSync(f));
}).listen(port, '0.0.0.0', () => console.error(`serving ${root} on 0.0.0.0:${port}`));
