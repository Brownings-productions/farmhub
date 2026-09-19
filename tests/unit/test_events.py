from farmhub.core.events import Event, EventBus


async def test_publish_delivers_to_subscribers() -> None:
    bus = EventBus()
    seen: list[Event] = []

    async def handler(event: Event) -> None:
        seen.append(event)

    bus.subscribe("a", handler)
    await bus.publish("a", {"x": 1})
    await bus.publish("b", {"x": 2})
    assert seen == [Event("a", {"x": 1})]


async def test_failing_handler_does_not_break_publisher_or_other_handlers() -> None:
    bus = EventBus()
    seen: list[str] = []

    async def bad(event: Event) -> None:
        raise RuntimeError("boom")

    async def good(event: Event) -> None:
        seen.append(event.topic)

    bus.subscribe("a", bad)
    bus.subscribe("a", good)
    await bus.publish("a")
    assert seen == ["a"]


async def test_unsubscribe() -> None:
    bus = EventBus()
    seen: list[str] = []

    async def handler(event: Event) -> None:
        seen.append(event.topic)

    unsubscribe = bus.subscribe("a", handler)
    unsubscribe()
    unsubscribe()  # idempotent
    await bus.publish("a")
    assert seen == []
