"""Strict, bounded HTTP byte access for inspecting large public data archives."""
import io
import re
import urllib.request


class HTTPRangeReader(io.RawIOBase):
    def __init__(self, url, size, *, budget=128*1024*1024,
                 max_request=16*1024*1024, timeout=25, opener=None):
        super().__init__()
        if size <= 0 or budget <= 0 or max_request <= 0:
            raise ValueError('Positive size, budget and request cap required')
        self.url, self.size = url, size
        self.budget, self.max_request, self.timeout = budget, max_request, timeout
        self.position = self.downloaded = self.requests = 0
        self.opener = opener or urllib.request.urlopen

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        target = offset + {io.SEEK_SET: 0, io.SEEK_CUR: self.position,
                           io.SEEK_END: self.size}[whence]
        if target < 0: raise ValueError('Negative seek')
        self.position = target
        return target

    def read(self, count=-1):
        if self.closed: raise ValueError('Reader closed')
        count = max(0, self.size-self.position) if count < 0 else min(count, max(0,self.size-self.position))
        if not count: return b''
        if count > self.max_request or self.downloaded+count > self.budget:
            raise ValueError('HTTP range budget or request cap exceeded')
        start, end = self.position, self.position+count-1
        # Query varies with the range to avoid caches reusing a different range.
        sep = '&' if '?' in self.url else '?'
        url = self.url + sep + f'research_range={start}-{end}'
        request = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}',
            'Accept-Encoding':'identity','User-Agent':'FaceFlux-Research/1.0'})
        with self.opener(request, timeout=self.timeout) as response:
            actual = response.headers.get('Content-Range','')
            match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',actual)
            if response.status != 206 or not match or tuple(map(int,match.groups())) != (start,end,self.size):
                raise RuntimeError('Server did not honor the exact byte range; full download refused')
            raw = response.read(count+1)
            self.downloaded += len(raw)
            self.requests += 1
            if len(raw) != count: raise RuntimeError('Truncated or oversized HTTP range')
        self.position += count
        return raw

    def readinto(self, buffer):
        raw = self.read(len(buffer))
        buffer[:len(raw)] = raw
        return len(raw)


class CachedRangeReader(io.RawIOBase):
    """Cache small HDF5 metadata reads without fetching image/depth arrays."""
    def __init__(self, reader, block_size=64*1024):
        super().__init__()
        self.reader, self.block_size, self.position, self.blocks = reader, block_size, 0, {}
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position
    def seek(self, offset, whence=io.SEEK_SET):
        target=offset+{io.SEEK_SET:0,io.SEEK_CUR:self.position,io.SEEK_END:self.reader.size}[whence]
        if target<0: raise ValueError('Negative seek')
        self.position=target
        return target
    def read(self,count=-1):
        count=max(0,self.reader.size-self.position) if count<0 else min(count,max(0,self.reader.size-self.position))
        parts=[]
        while count:
            block=self.position//self.block_size
            offset=self.position%self.block_size
            if block not in self.blocks:
                self.reader.seek(block*self.block_size)
                self.blocks[block]=self.reader.read(min(self.block_size,self.reader.size-block*self.block_size))
            raw=self.blocks[block][offset:offset+count]
            parts.append(raw);self.position+=len(raw);count-=len(raw)
        return b''.join(parts)
    def readinto(self,buffer):
        raw=self.read(len(buffer));buffer[:len(raw)]=raw;return len(raw)
