"""Loopback GET-only gateway; serves no local files and only two public data repos."""
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from urllib.parse import urlsplit,parse_qs
import urllib.request

def allowed(url):
    parsed=urlsplit(url)
    return (parsed.scheme=='https' and parsed.hostname=='huggingface.co' and not parsed.username and not parsed.password
        and parsed.port in (None,443) and any(parsed.path.startswith(prefix) for prefix in [
            '/datasets/lizb6626/Arb-Objaverse/resolve/','/datasets/allenai/objaverse/resolve/']))

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def do_GET(self):
        values=parse_qs(urlsplit(self.path).query);url=values.get('url',[''])[0]
        if not allowed(url):self.send_error(403,'Public dataset URL not allowed');return
        headers={'Accept-Encoding':'identity','User-Agent':'FaceFlux-public-data/1.0'}
        if self.headers.get('Range'):headers['Range']=self.headers['Range']
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=40) as remote:
                self.send_response(remote.status)
                for key in ['Content-Type','Content-Length','Content-Range','ETag','Last-Modified']:
                    if remote.headers.get(key):self.send_header(key,remote.headers[key])
                self.send_header('Connection','close');self.end_headers()
                remaining=64*1024**2
                while remaining:
                    chunk=remote.read(min(65536,remaining))
                    if not chunk:break
                    self.wfile.write(chunk);remaining-=len(chunk)
        except (BrokenPipeError,ConnectionResetError):pass
        except Exception:
            self.close_connection=True

if __name__=='__main__':ThreadingHTTPServer(('127.0.0.1',18765),Handler).serve_forever()
