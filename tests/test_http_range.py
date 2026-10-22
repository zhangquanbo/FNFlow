import io
import pytest
from src.http_range import HTTPRangeReader,CachedRangeReader

class Response(io.BytesIO):
    def __init__(self,raw,status,content_range):
        super().__init__(raw); self.status=status;self.headers={'Content-Range':content_range}

def test_range_seek_and_wrong_cache_are_checked():
    reader=HTTPRangeReader('https://example.test/large.zip',100,
        opener=lambda req,timeout:Response(b'abc',206,'bytes 97-99/100'))
    reader.seek(-3,io.SEEK_END)
    assert reader.read()==b'abc' and reader.tell()==100
    reader.seek(0)
    with pytest.raises(RuntimeError,match='exact byte range'):reader.read(3)

def test_full_download_and_excess_budget_are_refused():
    reader=HTTPRangeReader('https://example.test/large.zip',100,budget=4,
        opener=lambda req,timeout:Response(b'abc',200,'bytes 0-2/100'))
    with pytest.raises(ValueError,match='budget'):reader.read(5)
    with pytest.raises(RuntimeError,match='full download refused'):reader.read(3)

def test_small_metadata_reads_reuse_blocks_and_handle_eof():
    raw=io.BytesIO(b'0123456789');raw.size=10
    reader=CachedRangeReader(raw,block_size=4)
    reader.seek(3)
    buffer=bytearray(4)
    assert reader.readinto(buffer)==4 and buffer==b'3456'
    assert len(reader.blocks)==2
    reader.seek(1);assert reader.read(2)==b'12' and len(reader.blocks)==2
    reader.seek(-1,io.SEEK_END);assert reader.read(5)==b'9' and reader.read()==b''
