"""Safety edges for importing independently authorized local files."""
import pytest

from test_restricted_motions import fixture, helper


def test_later_missing_reference_causes_no_partial_import(tmp_path):
    module = helper()
    content, record = fixture(tmp_path)
    source = tmp_path / 'authorized'
    source.mkdir()
    (source / 'sample.npz').write_bytes(content)
    second = dict(record, destination='motions/second.npz')
    target = tmp_path / 'target'
    with pytest.raises(ValueError, match='exact'):
        module.import_authorized(source, target, [record, second])
    assert not target.exists()


def test_existing_different_destination_is_preserved(tmp_path):
    module = helper()
    content, record = fixture(tmp_path)
    source = tmp_path / 'authorized'
    source.mkdir()
    (source / 'sample.npz').write_bytes(content)
    target = tmp_path / 'target'
    existing = target / record['destination']
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b'preserve existing content')
    with pytest.raises(FileExistsError, match='overwrite'):
        module.import_authorized(source, target, [record])
    assert existing.read_bytes() == b'preserve existing content'


def test_outside_destination_is_rejected(tmp_path):
    module = helper()
    _, record = fixture(tmp_path)
    record['destination'] = '../outside.npz'
    with pytest.raises(ValueError, match='inside'):
        module.check(tmp_path / 'target', [record])
