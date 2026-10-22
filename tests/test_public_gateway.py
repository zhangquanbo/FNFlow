from scripts.public_dataset_gateway import allowed

def test_gateway_exposes_only_public_dataset_reads():
    assert allowed('https://huggingface.co/datasets/allenai/objaverse/resolve/main/glbs/object.glb')
    for url in ['file:///private/data','http://127.0.0.1/x','https://example.com/x',
                'https://huggingface.co/api/whoami','https://fixture-user@huggingface.co/datasets/allenai/objaverse/resolve/x']:
        assert not allowed(url)
