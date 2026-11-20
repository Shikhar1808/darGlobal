// The page turns model text into HTML; this checks it escapes everything it inserts. Run: node --test tests/
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const page = readFileSync(new URL('../app/static/index.html', import.meta.url), 'utf8');
const code = page.slice(page.indexOf('// ---- reply formatting'), page.indexOf('function add('));
const { render } = new Function(code + 'return { render };')();

test('model text cannot inject HTML', () => {
  const html = render('<script>alert(1)</script> **<img src=x onerror=alert(1)>**');
  assert.ok(!html.includes('<script') && !html.includes('<img'));
  assert.ok(html.includes('&lt;script&gt;') && html.includes('<strong>&lt;img'));
});

test('lists, bold and links', () => {
  const html = render('Two homes:\n- **Villa** https://wasalt.sa/a.\n- [Flat](https://wasalt.sa/b)\n1. first');
  assert.ok(html.includes('<ul><li><strong>Villa</strong> <a href="https://wasalt.sa/a" target="_blank" rel="noopener">https://wasalt.sa/a</a>.</li>'));
  assert.ok(html.includes('<a href="https://wasalt.sa/b" target="_blank" rel="noopener">Flat</a>'));
  assert.ok(html.includes('</ul><ol><li>first</li></ol>'));
});

test('a link can only be http(s)', () => {
  assert.ok(!render('[x](javascript:alert(1))').includes('<a'));
  assert.ok(!render('[x](https://a.com" onmouseover="alert(1))').includes('onmouseover="'));
});
