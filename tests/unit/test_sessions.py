"""Server-minted sessions (SPEC §3.6)."""

from datetime import UTC, datetime, timedelta

from farmhub.core.sessions import SessionStore

NOW = datetime(2026, 9, 20, 12, 0, tzinfo=UTC)


def test_a_session_is_minted_for_a_new_conversation() -> None:
    store = SessionStore()
    session = store.resolve(satellite="kitchen", conversation_id="ha-1", now=NOW)
    assert session.id.startswith("s-")
    assert session.satellite == "kitchen"


def test_the_same_conversation_from_the_same_satellite_continues() -> None:
    store = SessionStore()
    first = store.resolve(satellite="kitchen", conversation_id="ha-1", now=NOW)
    second = store.resolve(satellite="kitchen", conversation_id="ha-1", now=NOW)
    assert second.id == first.id


def test_a_conversation_id_alone_cannot_reach_another_satellites_session() -> None:
    """SPEC §3.6: HA's conversation_id is a lookup key, never trusted alone.

    The workshop satellite quoting the kitchen's conversation id must not inherit the
    kitchen's session, and with it the kitchen's scope.
    """
    store = SessionStore()
    kitchen = store.resolve(satellite="kitchen", conversation_id="ha-1", now=NOW)
    workshop = store.resolve(satellite="workshop", conversation_id="ha-1", now=NOW)
    assert workshop.id != kitchen.id
    assert workshop.satellite == "workshop"


def test_an_unidentified_caller_cannot_reach_an_identified_session() -> None:
    store = SessionStore()
    kitchen = store.resolve(satellite="kitchen", conversation_id="ha-1", now=NOW)
    anonymous = store.resolve(satellite=None, conversation_id="ha-1", now=NOW)
    assert anonymous.id != kitchen.id
    assert anonymous.satellite is None


def test_a_request_without_a_conversation_id_always_gets_a_fresh_session() -> None:
    store = SessionStore()
    first = store.resolve(satellite="kitchen", conversation_id=None, now=NOW)
    second = store.resolve(satellite="kitchen", conversation_id=None, now=NOW)
    assert first.id != second.id


def test_a_stale_session_is_expired_rather_than_resumed() -> None:
    store = SessionStore(ttl_s=60)
    first = store.resolve(satellite="kitchen", conversation_id="ha-1", now=NOW)
    later = store.resolve(
        satellite="kitchen", conversation_id="ha-1", now=NOW + timedelta(seconds=61)
    )
    assert later.id != first.id
    assert store.get(first.id) is None


def test_activity_keeps_a_session_alive() -> None:
    store = SessionStore(ttl_s=60)
    first = store.resolve(satellite="kitchen", conversation_id="ha-1", now=NOW)
    for offset in (30, 60, 90):
        current = store.resolve(
            satellite="kitchen", conversation_id="ha-1", now=NOW + timedelta(seconds=offset)
        )
        assert current.id == first.id


def test_the_store_is_bounded_so_unidentified_traffic_cannot_exhaust_memory() -> None:
    store = SessionStore(max_sessions=10)
    for i in range(50):
        store.resolve(satellite=None, conversation_id=f"c{i}", now=NOW + timedelta(seconds=i))
    assert len(store) <= 10


def test_returning_the_minted_session_id_continues_that_session() -> None:
    """The real Home Assistant round trip (SPEC §3.6).

    The integration sends FarmHub's minted session id back as the conversation id on
    the next turn. Without this, every turn after the first minted a new session and
    continuity never survived one exchange.
    """
    store = SessionStore()
    first = store.resolve(satellite="kitchen", conversation_id="ha-1", now=NOW)
    second = store.resolve(satellite="kitchen", conversation_id=first.id, now=NOW)
    assert second.id == first.id


def test_a_session_id_from_another_satellite_is_not_honoured() -> None:
    """The binding still holds when the lookup key is a session id."""
    store = SessionStore()
    kitchen = store.resolve(satellite="kitchen", conversation_id="ha-1", now=NOW)
    workshop = store.resolve(satellite="workshop", conversation_id=kitchen.id, now=NOW)
    assert workshop.id != kitchen.id
    assert workshop.satellite == "workshop"


def test_an_unidentified_caller_cannot_resume_a_session_by_its_id() -> None:
    store = SessionStore()
    kitchen = store.resolve(satellite="kitchen", conversation_id="ha-1", now=NOW)
    anonymous = store.resolve(satellite=None, conversation_id=kitchen.id, now=NOW)
    assert anonymous.id != kitchen.id
    assert anonymous.satellite is None
