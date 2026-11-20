// DarGlobal company pages (about, FAQ, why invest, investor, contact, hospitality, partners...).
// DarGlobal is behind bot protection (D-004), so this runs in a normal browser tab on darglobal.co.uk
// (DevTools console), as a visitor, about 1 page per second. Output is cleaned by hand into
// data/darglobal_company.json (one short record per FAQ answer or page; marketing filler dropped).
const clean = s => s.replace(/\s+/g, ' ').trim();

// FAQ: accordion items are in the server HTML even when closed
async function faq() {
  const doc = new DOMParser().parseFromString(await fetch('/faq').then(r => r.text()), 'text/html');
  const items = [...doc.querySelectorAll('[class*="faqItem__"]')].map(it => {
    const q = clean(it.querySelector('[class*="faqItemLabel"]').textContent);
    return { q, a: clean(it.textContent).slice(q.length).trim() };
  });
  return [...new Map(items.map(x => [x.q, x])).values()];  // the page repeats some questions
}

// Other pages: headings, paragraphs, list items and short leaf text, without header/footer/nav
async function page(path) {
  const doc = new DOMParser().parseFromString(await fetch(path).then(r => r.text()), 'text/html');
  doc.querySelectorAll('header, footer, nav, script, style, noscript, form, select, [class*="faqItem__"]').forEach(e => e.remove());
  const out = [], seen = new Set();
  doc.querySelectorAll('h1, h2, h3, h4, h5, p, li, div, span').forEach(e => {
    const block = /^(DIV|SPAN)$/.test(e.tagName);
    if (block ? e.children.length : e.querySelector('p, li, h1, h2, h3, h4, h5')) return;
    const t = clean(e.textContent);
    if (t.length > 2 && !seen.has(t) && (!block || t.includes(' '))) { seen.add(t); out.push((/^H/.test(e.tagName) ? '## ' : '') + t); }
  });
  return { url: location.origin + path, title: clean(doc.title), text: out.join('\n') };
}

const PATHS = ['/about', '/why-invest', '/investor', '/get-in-touch', '/luxury-golf-communities', '/hospitality',
  '/exclusive-membership', '/commercial', '/tokenization', '/one-of-one',
  ...['aston-martin', 'elie-saab', 'fendi-casa', 'lamborghini', 'marriott-residences-aida-oman', 'missoni', 'mouawad',
      'pagani', 'the-trump-organization', 'w-hotels', 'paramount'].map(p => '/partners/' + p)];

const result = { faq: await faq(), pages: [] };
for (const p of PATHS) {
  result.pages.push(await page(p));
  await new Promise(r => setTimeout(r, 1000));
}
console.log(JSON.stringify(result));
