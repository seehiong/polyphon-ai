import os

from polyphon.config import find_env_file, load_env, parse_env_file


def test_parse_env_file_handles_comments_blanks_and_quotes():
    parsed = parse_env_file(
        "\n".join(
            [
                "# a comment",
                "",
                "PLAIN=value",
                "QUOTED='single'",
                'DQUOTED="double"',
                "SPACED  =  padded  ",
                "export EXPORTED=exported-value",
                "WITH_EQUALS=a=b=c",
                "NOT_A_PAIR",
            ]
        )
    )
    assert parsed["PLAIN"] == "value"
    assert parsed["QUOTED"] == "single"
    assert parsed["DQUOTED"] == "double"
    assert parsed["SPACED"] == "padded"
    assert parsed["EXPORTED"] == "exported-value"
    # Only the first '=' separates key from value.
    assert parsed["WITH_EQUALS"] == "a=b=c"
    assert "NOT_A_PAIR" not in parsed


def test_parse_env_file_preserves_inner_quotes():
    parsed = parse_env_file("""MSG="he said \"hi\"" """)
    assert parsed["MSG"].startswith("he said")


def test_find_env_file_walks_up_from_subdirectory(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("KEY=from-root\n", encoding="utf-8")
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)

    monkeypatch.delenv("POLYPHON_ENV_FILE", raising=False)
    found = find_env_file(start=nested)
    assert found == tmp_path / ".env"


def test_find_env_file_respects_explicit_override(tmp_path, monkeypatch):
    custom = tmp_path / "custom.env"
    custom.write_text("KEY=custom\n", encoding="utf-8")
    monkeypatch.setenv("POLYPHON_ENV_FILE", str(custom))
    assert find_env_file() == custom


def test_find_env_file_returns_none_when_absent(tmp_path, monkeypatch):
    monkeypatch.delenv("POLYPHON_ENV_FILE", raising=False)
    # An empty tmp_path has no .env, but parents might; point the override at a
    # nonexistent file to assert the miss path explicitly.
    monkeypatch.setenv("POLYPHON_ENV_FILE", str(tmp_path / "nope.env"))
    assert find_env_file() is None


def test_load_env_does_not_override_real_environment(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("POLYPHON_TEST_KEY=from-file\n", encoding="utf-8")
    monkeypatch.setenv("POLYPHON_ENV_FILE", str(env_file))
    monkeypatch.setenv("POLYPHON_TEST_KEY", "from-real-env")

    load_env(force=True)

    # Real environment variables win; this is what lets CI override the file.
    assert os.environ["POLYPHON_TEST_KEY"] == "from-real-env"


def test_load_env_sets_missing_values(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("POLYPHON_TEST_UNSET=from-file\n", encoding="utf-8")
    monkeypatch.setenv("POLYPHON_ENV_FILE", str(env_file))
    monkeypatch.delenv("POLYPHON_TEST_UNSET", raising=False)

    loaded = load_env(force=True)

    assert loaded == env_file
    assert os.environ["POLYPHON_TEST_UNSET"] == "from-file"


def test_load_env_missing_file_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv("POLYPHON_ENV_FILE", str(tmp_path / "absent.env"))
    assert load_env(force=True) is None


def test_find_env_file_falls_back_to_user_home(tmp_path, monkeypatch):
    from pathlib import Path

    monkeypatch.delenv("POLYPHON_ENV_FILE", raising=False)
    fake_home = tmp_path / "home"
    global_env = fake_home / ".polyphon" / ".env"
    global_env.parent.mkdir(parents=True)
    global_env.write_text("KEY=global\n", encoding="utf-8")

    monkeypatch.setattr(Path, "home", lambda: fake_home)

    isolated_dir = tmp_path / "other" / "project"
    isolated_dir.mkdir(parents=True)

    found = find_env_file(start=isolated_dir)
    assert found == global_env
