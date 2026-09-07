// Serves this directory over HTTPS so a phone can run the checks.
//
// WebGPU requires a secure context, so a plain http:// LAN address will not do --
// `navigator.gpu` is simply absent there, however capable the device is. HTTPS
// with a self-signed certificate does count as secure once the warning is
// accepted, which needs no USB cable and no extra tooling.
//
//   node serve.js [port]        # default 8734
//
// The certificate is generated on first run into .tls/ (gitignored), with the
// machine's current addresses as subject-alt-names so it survives a DHCP change.
// Binding is on all interfaces, which is the point, but do note that anything on
// the network can reach it while it runs.
const http = require('http');
const https = require('https');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { execFileSync } = require('child_process');

const root = __dirname;
const port = Number(process.argv[2] || 8734);
const tlsDir = path.join(root, '.tls');

function localAddresses() {
  return Object.values(os.networkInterfaces())
    .flat()
    .filter(i => i && i.family === 'IPv4' && !i.internal)
    .map(i => i.address);
}

function ensureCert() {
  const key = path.join(tlsDir, 'key.pem');
  const cert = path.join(tlsDir, 'cert.pem');
  const addrs = localAddresses();
  const stamp = path.join(tlsDir, 'addrs');
  const want = addrs.join(',');
  if (fs.existsSync(key) && fs.existsSync(cert) && fs.existsSync(stamp)
      && fs.readFileSync(stamp, 'utf8') === want) {
    return { key: fs.readFileSync(key), cert: fs.readFileSync(cert) };
  }
  fs.mkdirSync(tlsDir, { recursive: true });
  // Chrome needs the address in a subjectAltName; a CN alone is ignored.
  const san = ['DNS:localhost', 'IP:127.0.0.1', ...addrs.map(a => `IP:${a}`)].join(',');
  const conf = path.join(tlsDir, 'openssl.cnf');
  fs.writeFileSync(conf, [
    '[req]', 'distinguished_name=dn', 'x509_extensions=v3', 'prompt=no',
    '[dn]', 'CN=xn-webgpu-check',
    '[v3]', `subjectAltName=${san}`, 'basicConstraints=CA:FALSE',
    'keyUsage=digitalSignature,keyEncipherment', 'extendedKeyUsage=serverAuth',
  ].join('\n'));
  execFileSync('openssl', [
    'req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', '365',
    '-keyout', key, '-out', cert, '-config', conf,
  ], { stdio: 'ignore' });
  fs.writeFileSync(stamp, want);
  console.error(`generated a self-signed certificate for ${san}`);
  return { key: fs.readFileSync(key), cert: fs.readFileSync(cert) };
}

const TYPES = {
  '.html': 'text/html', '.js': 'text/javascript', '.wasm': 'application/wasm',
  '.json': 'application/json', '.wgsl': 'text/plain', '.css': 'text/css',
};

function handler(req, res) {
  if (req.method === 'POST') {
    let b = '';
    req.on('data', c => (b += c));
    req.on('end', () => {
      res.writeHead(204);
      res.end();
      console.log(b);
    });
    return;
  }
  const rel = decodeURIComponent(req.url.split('?')[0]);
  const f = path.join(root, rel === '/' ? 'index.html' : rel);
  if (!f.startsWith(root) || !fs.existsSync(f) || fs.statSync(f).isDirectory()) {
    res.writeHead(404, { 'Content-Type': 'text/plain' });
    res.end('not found: ' + rel);
    return;
  }
  res.writeHead(200, { 'Content-Type': TYPES[path.extname(f)] || 'text/plain' });
  res.end(fs.readFileSync(f));
}

const { key, cert } = ensureCert();
https.createServer({ key, cert }, handler).listen(port, '0.0.0.0', () => {
  console.error(`\nhttps on 0.0.0.0:${port} — serving ${root}\n`);
  for (const a of ['localhost', ...localAddresses()]) {
    console.error(`  https://${a}:${port}/`);
  }
  console.error('\nOn the phone Chrome will warn about the certificate: tap');
  console.error('Advanced then Proceed. HTTPS is what makes WebGPU available at all.\n');
});
// Plain http alongside, purely so a phone that lands on it gets told why the
// page reports no WebGPU rather than silently failing.
http.createServer(handler).listen(port + 1, '0.0.0.0', () => {
  console.error(`http on 0.0.0.0:${port + 1} (insecure; WebGPU will be absent)`);
});
