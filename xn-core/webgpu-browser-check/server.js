const http = require('http'), fs = require('fs'), path = require('path');
const srv = http.createServer((req, res) => {
  if (req.method === 'POST') {
    let b = ''; req.on('data', c => (b += c));
    req.on('end', () => { res.writeHead(204); res.end(); console.log(b); srv.close(); process.exit(0); });
    return;
  }
  const url = req.url === '/' ? '/index.html' : req.url;
  const f = path.join(__dirname, decodeURIComponent(url));
  if (!fs.existsSync(f)) { res.writeHead(404); res.end(); return; }
  const type = f.endsWith('.html') ? 'text/html' : f.endsWith('.json') ? 'application/json' : 'text/plain';
  res.writeHead(200, { 'Content-Type': type });
  res.end(fs.readFileSync(f));
});
srv.listen(8732, () => console.error('listening'));
setTimeout(() => { console.log(JSON.stringify({error:'timeout'})); process.exit(1); }, 90000);
