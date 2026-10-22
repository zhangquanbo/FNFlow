from scripts.prepare_training_probe import select_probe


def test_training_probe_is_order_independent_training_only_and_frozen_across_methods():
    rows=[{'path':f'/cache/{i}.npz','split':'train' if i<10 else 'val'} for i in range(20)]
    first=select_probe(rows,'hypersim',4)
    assert first==select_probe(list(reversed(rows)),'hypersim',4)
    assert all(r['split']=='train' for r in first)
    assert len(first)==4
