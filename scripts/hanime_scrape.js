"""
IMPORTANT: Paste this ENTIRE script into the browser console on hanime1.me.
It auto-scrapes all video pages, extracts MP4 URLs, and downloads them to console.
Results are printed as JSON — copy them to a file.
"""

(async () => {
  const UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/148.0.0.0 Safari/537.36";
  const results = [];
  const DELAY = 1500;
  const MAX = 50;

  // Get video IDs from queue
  const ids = ["406536","406533","406535","406534","406526","406532","406538","406537","406531","406505","406504","434","433","432","405932","405852","405851","405849","405847","431","405850","405846","404983","404980","404988","404987","404986","404982","404979","404990","404989","166761","404480","404917","404477","143654","404479","404471","166763","166762","404721","166752","166751","166746","166750","166749","166748","166747","157878","157877"];

  for (let i = 0; i < Math.min(ids.length, MAX); i++) {
    const vid = ids[i];
    const url = `https://hanime1.me/watch?v=${vid}`;

    try {
      const resp = await fetch(url, { credentials: "include" });
      const html = await resp.text();

      // Extract mp4 source URL
      const srcMatch = html.match(/<source\s+src="(https?:\/\/vdownload[^"]+\.mp4[^"]*)"/);
      if (srcMatch) {
        results.push({
          domain: "anime2d",
          video_id: vid,
          url: srcMatch[1],
        });
        console.log(`[${i+1}/${ids.length}] ${vid}: OK — ${srcMatch[1].substring(0, 60)}...`);
      } else {
        console.log(`[${i+1}/${ids.length}] ${vid}: NO MP4 FOUND`);
      }
    } catch(e) {
      console.log(`[${i+1}/${ids.length}] ${vid}: ERROR — ${e.message}`);
    }

    if (i < ids.length - 1) await new Promise(r => setTimeout(r, DELAY));
  }

  console.log("=== DONE ===");
  console.log(JSON.stringify(results));
})();