import io,zipfile
import pytest
from scripts.download_arb_training import read_zip_member

def test_partial_member_decoder_validates_crc_and_matches_zipfile():
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED) as z:z.writestr('asset/normal.exr',b'normal bytes'*100)
    raw=buffer.getvalue()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:info=z.getinfo('asset/normal.exr')
    source=io.BytesIO(raw);source.size=len(raw)
    assert read_zip_member(source,info)==b'normal bytes'*100
    info.CRC^=1
    with pytest.raises(RuntimeError,match='CRC'):read_zip_member(source,info)
