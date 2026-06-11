"""Batch URL scraper — Claude runs these via jshook page_evaluate one-by-one.

Strategy: for each video page:
  1. Navigate to page
  2. Run extract JS -> get MP4 URL
  3. Save to scraped_urls.jsonl
  4. Next page

Hanime: source[src] contains vdownload.hembed.com/XXXX-480p.mp4?secure=...
Iwara: script contains fileUrl = "https://xxx.iwara.tv/view?hash=..."
"""

HANIME_EXTRACT = """(() => {
  const sources = [];
  document.querySelectorAll('source[src]').forEach(s => {
    const src = s.getAttribute('src');
    if (src && (src.includes('.mp4') || src.includes('vdownload'))) sources.push(src);
  });
  return JSON.stringify({ sources: [...new Set(sources)] });
})()"""

IWARA_EXTRACT = """(() => {
  const sources = [];
  document.querySelectorAll('video source, video').forEach(el => {
    const src = el.src || el.getAttribute('src');
    if (src && src.includes('.mp4')) sources.push(src);
  });
  [...document.querySelectorAll('script')].forEach(s => {
    const m = s.textContent.match(/fileUrl["']?\s*[:=]\s*["']([^"']+\\.mp4[^"']*)["']/g);
    if (m) m.forEach(x => {
      const u = x.match(/["']([^"']+\\.mp4[^"']*)["']/);
      if (u) sources.push(u[1]);
    });
  });
  return JSON.stringify({ sources: [...new Set(sources)].filter(s => s.includes('/view?hash=')) });
})()"""

print("=== HANIME EXTRACT JS (paste in console on hanime video page) ===")
print(HANIME_EXTRACT)
print()
print("=== IWARA EXTRACT JS (paste in console on iwara video page) ===")
print(IWARA_EXTRACT)
print()
print("=== COMMAND FORMAT ===")
print("After each extract, save to: D:/cumhub/anti-nsfw-yolo/data-collector/manifests/crawl/scraped_urls.jsonl")
print("Format: {\"domain\":\"anime2d\",\"video_id\":\"406536\",\"url\":\"https://...\"}")
