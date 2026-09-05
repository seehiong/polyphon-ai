import pytest

from polyphon.voicedb import base as voicedb_base

DEFAULT_DB_PATH = "~/.polyphon/voicedb"


@pytest.fixture(autouse=True)
def isolated_voicedb(tmp_path, monkeypatch):
    """Redirect the default VoiceDB location to a temp dir for every test.

    VoiceDB defaults to ~/.polyphon/voicedb, which is real user data. Any test
    that constructs one without an explicit path (directly, or indirectly via a
    server endpoint) would otherwise enrol fake speakers into the developer's
    own database, where they persist after the run and break `--identify` with
    an embedding-size mismatch.
    """
    original_init = voicedb_base.VoiceDB.__init__

    def init_with_temp_default(self, db_path=DEFAULT_DB_PATH, *args, **kwargs):
        if str(db_path) == DEFAULT_DB_PATH:
            db_path = tmp_path / "voicedb"
        original_init(self, db_path, *args, **kwargs)

    monkeypatch.setattr(voicedb_base.VoiceDB, "__init__", init_with_temp_default)
    yield
