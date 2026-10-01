import { build } from 'esbuild';
import { mkdir, readFile, writeFile, cp } from 'node:fs/promises';
import { createHash } from 'node:crypto';
await mkdir('dist', { recursive: true });
await cp('public','dist',{recursive:true});
await build({entryPoints:['src/main.tsx'],bundle:true,external:['/fonts/*'],outdir:'dist',minify:true,sourcemap:true,jsx:'automatic',define:{'process.env.NODE_ENV':'"production"'}});
const version = createHash('sha256').update(await readFile('dist/main.js')).update(await readFile('dist/main.css')).digest('hex').slice(0,12);
const html = (await readFile('index.html','utf8')).replace('/main.js',`/main.js?v=${version}`).replace('/main.css',`/main.css?v=${version}`);
await writeFile('dist/index.html',html);
