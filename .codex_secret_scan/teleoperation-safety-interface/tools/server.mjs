import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { dirname, extname, join, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const port = Number(process.env.PORT || 4173);
const host = "127.0.0.1";

const contentTypes = {
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".md": "text/markdown; charset=utf-8",
  ".svg": "image/svg+xml"
};

function isInsideRoot(filePath) {
  return filePath === root || filePath.startsWith(`${root}${sep}`);
}

async function resolveFile(pathname) {
  const requested = pathname === "/" ? "/index.html" : pathname;
  const filePath = resolve(root, `.${decodeURIComponent(requested)}`);

  if (!isInsideRoot(filePath)) {
    return { status: 403 };
  }

  const fileStat = await stat(filePath);
  if (fileStat.isDirectory()) {
    return { status: 200, filePath: join(filePath, "index.html") };
  }

  return { status: 200, filePath };
}

const server = createServer(async (request, response) => {
  try {
    const url = new URL(request.url || "/", `http://${request.headers.host}`);
    const result = await resolveFile(url.pathname);

    if (result.status !== 200 || !result.filePath) {
      response.writeHead(result.status);
      response.end("Forbidden");
      return;
    }

    const body = await readFile(result.filePath);
    const type = contentTypes[extname(result.filePath)] || "application/octet-stream";
    response.writeHead(200, { "Content-Type": type });
    response.end(body);
  } catch (error) {
    const status = error.code === "ENOENT" ? 404 : 500;
    response.writeHead(status, { "Content-Type": "text/plain; charset=utf-8" });
    response.end(status === 404 ? "Not found" : "Server error");
  }
});

server.listen(port, host, () => {
  console.log(`Teleoperation Safety Interface running at http://${host}:${port}`);
});
