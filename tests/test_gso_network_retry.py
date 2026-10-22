import io,json,urllib.error
import pytest
from scripts import prepare_gso_holdout as intake


def test_metadata_tls_timeout_retries_same_source_without_changing_content(monkeypatch):
    calls=[]
    def request(url,timeout):
        calls.append(url)
        if len(calls)==1:raise urllib.error.URLError(TimeoutError('TLS handshake timed out'))
        return io.BytesIO(b'{"version":7,"license_name":"CC-BY 4.0"}')
    monkeypatch.setattr(intake.urllib.request,'urlopen',request);monkeypatch.setattr(intake.time,'sleep',lambda n:None)
    assert intake.fetch_json('https://official.example/model')=={'version':7,'license_name':'CC-BY 4.0'}
    assert calls==['https://official.example/model']*2


def test_bad_metadata_is_not_replaced_or_treated_as_transient_network_error(monkeypatch):
    calls=[]
    def request(url,timeout):calls.append(url);return io.BytesIO(b'not JSON')
    monkeypatch.setattr(intake.urllib.request,'urlopen',request);monkeypatch.setattr(intake.time,'sleep',lambda n:None)
    with pytest.raises(json.JSONDecodeError):intake.fetch_json('https://official.example/model')
    assert len(calls)==1
