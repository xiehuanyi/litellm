"""
Batching tests for NewRelicMetricsLogger: flush-window interval computation,
dimension-bucket aggregation, the 4xx-drop vs 5xx/network-requeue policy, the
retry-queue cap, and the stop flag that ends the periodic flush loop.
"""

import asyncio
import gzip
import json
from unittest.mock import AsyncMock, patch

import pytest
from httpx import Request, Response

from litellm.integrations.newrelic.newrelic_metrics import (
    NewRelicMetricsLogger,
    build_metric_payload,
)
from litellm.types.integrations.newrelic import (
    NEWRELIC_METRIC_COMPLETION_TOKENS,
    NEWRELIC_METRIC_COST_USD,
    NEWRELIC_METRIC_ENDPOINT_BY_REGION,
    NEWRELIC_METRIC_PROMPT_TOKENS,
    NEWRELIC_METRIC_REQUEST_DURATION_MS,
    NEWRELIC_METRIC_REQUESTS,
    NEWRELIC_METRIC_TOTAL_TOKENS,
    NewRelicMetricRecord,
)


def _record(
    team_id="team-a",
    team_alias=None,
    model="gpt-4o",
    model_group=None,
    status="success",
    response_cost=0.5,
    prompt_tokens=10,
    completion_tokens=20,
    total_tokens=30,
    duration_ms=100.0,
) -> NewRelicMetricRecord:
    return NewRelicMetricRecord(
        team_id=team_id,
        team_alias=team_alias if team_alias is not None else f"{team_id}-alias",
        model_group=model_group if model_group is not None else f"{model}-group",
        model=model,
        custom_llm_provider="openai",
        status=status,
        response_cost=response_cost,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        duration_ms=duration_ms,
    )


def _standard_logging_object(team_id="team-a", response_cost=0.25) -> dict:
    return {
        "metadata": {"user_api_key_team_id": team_id, "user_api_key_team_alias": f"{team_id}-alias"},
        "model_group": "gpt-4o-group",
        "model": "gpt-4o",
        "custom_llm_provider": "openai",
        "status": "success",
        "response_cost": response_cost,
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
        "response_time": 0.1,
    }


def _make_logger(**kwargs) -> NewRelicMetricsLogger:
    with patch("asyncio.create_task"):
        return NewRelicMetricsLogger(newrelic_api_key="test-key", **kwargs)


def _response(status_code: int, text: str = "") -> Response:
    return Response(status_code, request=Request("POST", "https://example.com"), text=text)


def _metrics_by_name(payload, name):
    return [m for m in payload[0]["metrics"] if m["name"] == name]


class TestBuildMetricPayload:
    def test_interval_and_timestamp_reflect_flush_window(self):
        payload = build_metric_payload((_record(),), window_start=1_000.0, now=1_007.5)

        assert payload[0]["common"]["timestamp"] == 1_000_000
        assert payload[0]["common"]["interval.ms"] == 7_500

    def test_interval_is_at_least_one_ms(self):
        payload = build_metric_payload((_record(),), window_start=1_000.0, now=1_000.0)

        assert payload[0]["common"]["interval.ms"] == 1

    def test_single_record_metric_values(self):
        payload = build_metric_payload(
            (_record(response_cost=0.5, prompt_tokens=10, completion_tokens=20, total_tokens=30, duration_ms=100.0),),
            window_start=1_000.0,
            now=1_005.0,
        )

        by_name = {m["name"]: m for m in payload[0]["metrics"]}
        assert by_name[NEWRELIC_METRIC_REQUESTS]["value"] == 1.0
        assert by_name[NEWRELIC_METRIC_REQUESTS]["type"] == "count"
        assert by_name[NEWRELIC_METRIC_COST_USD]["value"] == 0.5
        assert by_name[NEWRELIC_METRIC_PROMPT_TOKENS]["value"] == 10.0
        assert by_name[NEWRELIC_METRIC_COMPLETION_TOKENS]["value"] == 20.0
        assert by_name[NEWRELIC_METRIC_TOTAL_TOKENS]["value"] == 30.0
        duration = by_name[NEWRELIC_METRIC_REQUEST_DURATION_MS]
        assert duration["type"] == "summary"
        assert duration["value"] == {"count": 1, "sum": 100.0, "min": 100.0, "max": 100.0}
        assert by_name[NEWRELIC_METRIC_REQUESTS]["attributes"] == {
            "team_id": "team-a",
            "team_alias": "team-a-alias",
            "model_group": "gpt-4o-group",
            "model": "gpt-4o",
            "custom_llm_provider": "openai",
            "status": "success",
        }

    def test_aggregates_across_dimension_buckets(self):
        """Two teams x two models in one queue land in the right bucket sums.

        team_alias and model_group are held constant so bucketing provably keys on
        team_id and model themselves, not on correlated fields.
        """
        shared = {"team_alias": "shared-alias", "model_group": "shared-group"}
        records = (
            _record(team_id="team-a", model="gpt-4o", response_cost=0.1, total_tokens=10, duration_ms=50.0, **shared),
            _record(team_id="team-a", model="gpt-4o", response_cost=0.2, total_tokens=20, duration_ms=150.0, **shared),
            _record(
                team_id="team-a", model="claude-4", response_cost=0.4, total_tokens=40, duration_ms=200.0, **shared
            ),
            _record(team_id="team-b", model="gpt-4o", response_cost=0.8, total_tokens=80, duration_ms=300.0, **shared),
        )
        payload = build_metric_payload(records, window_start=1_000.0, now=1_005.0)

        cost_by_bucket = {
            (m["attributes"]["team_id"], m["attributes"]["model"]): m["value"]
            for m in _metrics_by_name(payload, NEWRELIC_METRIC_COST_USD)
        }
        assert cost_by_bucket == {
            ("team-a", "gpt-4o"): pytest.approx(0.3),
            ("team-a", "claude-4"): pytest.approx(0.4),
            ("team-b", "gpt-4o"): pytest.approx(0.8),
        }

        requests_by_bucket = {
            (m["attributes"]["team_id"], m["attributes"]["model"]): m["value"]
            for m in _metrics_by_name(payload, NEWRELIC_METRIC_REQUESTS)
        }
        assert requests_by_bucket == {
            ("team-a", "gpt-4o"): 2.0,
            ("team-a", "claude-4"): 1.0,
            ("team-b", "gpt-4o"): 1.0,
        }

        duration_by_bucket = {
            (m["attributes"]["team_id"], m["attributes"]["model"]): m["value"]
            for m in _metrics_by_name(payload, NEWRELIC_METRIC_REQUEST_DURATION_MS)
        }
        assert duration_by_bucket[("team-a", "gpt-4o")] == {"count": 2, "sum": 200.0, "min": 50.0, "max": 150.0}

    def test_status_is_a_bucket_dimension(self):
        records = (
            _record(status="success", response_cost=0.1),
            _record(status="failure", response_cost=0.0),
        )
        payload = build_metric_payload(records, window_start=1_000.0, now=1_005.0)

        statuses = {m["attributes"]["status"] for m in _metrics_by_name(payload, NEWRELIC_METRIC_REQUESTS)}
        assert statuses == {"success", "failure"}

    def test_empty_attribute_values_are_omitted(self):
        record = NewRelicMetricRecord(
            team_id="",
            team_alias="",
            model_group="",
            model="gpt-4o",
            custom_llm_provider="openai",
            status="success",
            response_cost=0.0,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            duration_ms=0.0,
        )
        payload = build_metric_payload((record,), window_start=1_000.0, now=1_005.0)

        attributes = payload[0]["metrics"][0]["attributes"]
        assert "team_id" not in attributes
        assert "team_alias" not in attributes
        assert "model_group" not in attributes


class TestQueueAndFlush:
    @pytest.mark.asyncio
    async def test_log_event_queues_record_from_standard_logging_object(self):
        logger = _make_logger()

        await logger.async_log_success_event(
            kwargs={"standard_logging_object": _standard_logging_object()},
            response_obj={},
            start_time=None,
            end_time=None,
        )

        assert len(logger.log_queue) == 1
        record = logger.log_queue[0]
        assert record.team_id == "team-a"
        assert record.response_cost == 0.25
        assert record.duration_ms == pytest.approx(100.0)

    @pytest.mark.asyncio
    async def test_failure_event_queues_record(self):
        logger = _make_logger()

        slo = _standard_logging_object()
        slo["status"] = "failure"
        await logger.async_log_failure_event(
            kwargs={"standard_logging_object": slo},
            response_obj={},
            start_time=None,
            end_time=None,
        )

        assert len(logger.log_queue) == 1
        assert logger.log_queue[0].status == "failure"

    @pytest.mark.asyncio
    async def test_threshold_flush_uses_flush_queue(self):
        logger = _make_logger()
        logger.batch_size = 1
        logger.flush_queue = AsyncMock()

        await logger.async_log_success_event(
            kwargs={"standard_logging_object": _standard_logging_object()},
            response_obj={},
            start_time=None,
            end_time=None,
        )

        logger.flush_queue.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_flush_queue_updates_last_flush_time_on_success(self):
        logger = _make_logger()
        logger.log_queue = [_record()]
        logger.last_flush_time = 0
        logger.async_client.post = AsyncMock(return_value=_response(202))

        await logger.flush_queue()

        assert logger.log_queue == []
        assert logger.last_flush_time > 0

    @pytest.mark.asyncio
    async def test_flush_queue_does_not_update_last_flush_time_on_requeue(self):
        logger = _make_logger()
        logger.log_queue = [_record()]
        logger.last_flush_time = 123.0
        logger.async_client.post = AsyncMock(return_value=_response(500))

        await logger.flush_queue()

        assert logger.last_flush_time == 123.0

    @pytest.mark.asyncio
    async def test_sent_payload_window_starts_at_last_flush_time(self):
        logger = _make_logger()
        logger.log_queue = [_record()]
        logger.last_flush_time = 2_000.0
        logger.async_client.post = AsyncMock(return_value=_response(202))

        with patch("litellm.integrations.newrelic.newrelic_metrics.time.time", return_value=2_010.0):
            await logger.async_send_batch()

        sent = logger.async_client.post.await_args.kwargs
        body = json.loads(gzip.decompress(sent["data"]).decode("utf-8"))
        assert body[0]["common"]["timestamp"] == 2_000_000
        assert body[0]["common"]["interval.ms"] == 10_000
        assert sent["headers"]["Api-Key"] == "test-key"
        assert sent["headers"]["Content-Encoding"] == "gzip"
        assert sent["url"] == NEWRELIC_METRIC_ENDPOINT_BY_REGION["us"]


class TestFlushConcurrency:
    @pytest.mark.asyncio
    async def test_records_appended_during_flush_await_survive(self):
        """A record appended by a concurrent request while the POST is in flight
        must survive the flush, not be clobbered by a queue replacement."""
        logger = _make_logger()
        logger.log_queue = [_record(team_id="team-a")]
        interleaved = _record(team_id="team-interleaved")

        async def _post_appending_mid_flight(**kwargs):
            logger.log_queue.append(interleaved)
            return _response(202)

        logger.async_client.post = AsyncMock(side_effect=_post_appending_mid_flight)

        await logger.async_send_batch()

        assert logger.log_queue == [interleaved]
        body = json.loads(gzip.decompress(logger.async_client.post.await_args.kwargs["data"]).decode("utf-8"))
        team_ids = {m["attributes"]["team_id"] for m in body[0]["metrics"]}
        assert team_ids == {"team-a"}

    @pytest.mark.asyncio
    async def test_records_appended_during_failed_flush_await_survive_requeue(self):
        """The requeue path must also preserve interleaved records: batch is
        prepended in place, never assigned over the live queue."""
        logger = _make_logger()
        original = _record(team_id="team-a")
        logger.log_queue = [original]
        interleaved = _record(team_id="team-interleaved")

        async def _post_appending_mid_flight(**kwargs):
            logger.log_queue.append(interleaved)
            return _response(500)

        logger.async_client.post = AsyncMock(side_effect=_post_appending_mid_flight)

        await logger.async_send_batch()

        assert logger.log_queue == [original, interleaved]


class TestErrorPolicy:
    @pytest.mark.asyncio
    async def test_4xx_drops_batch(self):
        logger = _make_logger()
        logger.log_queue = [_record(), _record(team_id="team-b")]
        logger.async_client.post = AsyncMock(return_value=_response(400, text="bad request"))

        await logger.async_send_batch()

        assert logger.log_queue == []
        assert logger.async_client.post.await_count == 1

    @pytest.mark.asyncio
    async def test_403_drops_batch_and_names_permanent_credential_failure(self):
        logger = _make_logger()
        logger.log_queue = [_record()]
        logger.async_client.post = AsyncMock(return_value=_response(403))

        with patch("litellm.integrations.newrelic.newrelic_metrics.verbose_logger") as mock_logger:
            await logger.async_send_batch()

        assert logger.log_queue == []
        warning_text = " ".join(str(arg) for call in mock_logger.warning.call_args_list for arg in call.args)
        assert "permanent credential failure" in warning_text

    @pytest.mark.asyncio
    async def test_5xx_requeues_batch(self):
        records = [_record(), _record(team_id="team-b")]
        logger = _make_logger()
        logger.log_queue = list(records)
        logger.async_client.post = AsyncMock(return_value=_response(500))

        await logger.async_send_batch()

        assert logger.log_queue == records

    @pytest.mark.asyncio
    async def test_network_error_requeues_batch(self):
        records = [_record()]
        logger = _make_logger()
        logger.log_queue = list(records)
        logger.async_client.post = AsyncMock(side_effect=ConnectionError("boom"))

        await logger.async_send_batch()

        assert logger.log_queue == records

    @pytest.mark.asyncio
    async def test_requeue_is_capped_dropping_oldest(self):
        logger = _make_logger()
        logger.max_queue_size = 3
        oldest = _record(team_id="oldest")
        rest = [_record(team_id=f"team-{i}") for i in range(3)]
        logger.log_queue = [oldest, *rest]
        logger.async_client.post = AsyncMock(return_value=_response(500))

        await logger.async_send_batch()

        assert logger.log_queue == rest

    @pytest.mark.asyncio
    async def test_requeued_records_are_resent_with_new_records(self):
        logger = _make_logger()
        logger.log_queue = [_record()]
        logger.async_client.post = AsyncMock(return_value=_response(500))

        await logger.async_send_batch()
        logger.log_queue.append(_record(team_id="team-b"))
        logger.async_client.post = AsyncMock(return_value=_response(202))

        await logger.async_send_batch()

        sent = logger.async_client.post.await_args.kwargs
        body = json.loads(gzip.decompress(sent["data"]).decode("utf-8"))
        team_ids = {m["attributes"]["team_id"] for m in body[0]["metrics"]}
        assert team_ids == {"team-a", "team-b"}
        assert logger.log_queue == []


class TestStopFlag:
    @pytest.mark.asyncio
    async def test_stop_ends_periodic_flush_loop(self):
        logger = _make_logger()
        logger.flush_interval = 0.01
        logger.flush_queue = AsyncMock()

        task = asyncio.create_task(logger.periodic_flush())
        await asyncio.sleep(0.05)
        assert not task.done()

        logger.stop()
        await asyncio.wait_for(task, timeout=1.0)

        assert task.done()

    @pytest.mark.asyncio
    async def test_stopped_logger_exits_after_one_final_drain(self):
        logger = _make_logger()
        logger.flush_interval = 0.01
        logger.flush_queue = AsyncMock()
        logger._stopped = True

        await asyncio.wait_for(logger.periodic_flush(), timeout=1.0)

        logger.flush_queue.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_eviction_drains_queued_records(self):
        """Eviction must post what is already queued, not silently discard it."""
        from litellm.litellm_core_utils.specialty_caches.dynamic_logging_cache import (
            DynamicLoggingCache,
        )

        cache = DynamicLoggingCache()
        logger = _make_logger()
        logger.log_queue = [_record(), _record(team_id="team-b")]
        logger.async_client.post = AsyncMock(return_value=_response(202))
        credentials = {"newrelic_api_key": "test-key", "newrelic_region": None}
        cache.set_cache(credentials=credentials, service_name="newrelic", logging_obj=logger)

        key = cache.get_cache_key(args={**credentials, "service_name": "newrelic"})
        cache.cache._remove_key(key)
        for _ in range(10):
            await asyncio.sleep(0)

        logger.async_client.post.assert_awaited_once()
        body = json.loads(gzip.decompress(logger.async_client.post.await_args.kwargs["data"]).decode("utf-8"))
        team_ids = {m["attributes"]["team_id"] for m in body[0]["metrics"]}
        assert team_ids == {"team-a", "team-b"}
        assert logger.log_queue == []

    @pytest.mark.asyncio
    async def test_dynamic_logging_cache_eviction_calls_stop(self):
        from litellm.litellm_core_utils.specialty_caches.dynamic_logging_cache import (
            DynamicLoggingCache,
        )

        cache = DynamicLoggingCache()
        logger = _make_logger()
        credentials = {"newrelic_api_key": "test-key", "newrelic_region": None}
        cache.set_cache(credentials=credentials, service_name="newrelic", logging_obj=logger)

        key = cache.get_cache_key(args={**credentials, "service_name": "newrelic"})
        cache.cache._remove_key(key)

        assert logger._stopped is True
        assert cache.get_cache(credentials=credentials, service_name="newrelic") is None


@pytest.mark.asyncio
async def test_append_after_eviction_drain_self_flushes():
    """An in-flight callback holding an evicted (stopped) logger still delivers
    its record: with no periodic loop left, the append itself drains."""
    logger = _make_logger()
    with patch.object(
        logger.async_client, "post", new=AsyncMock(return_value=_response(202))
    ) as mock_post:
        logger.stop()
        await logger.async_log_success_event(
            {"standard_logging_object": _standard_logging_object()}, None, None, None
        )
    assert mock_post.await_count >= 1, "record appended after stop() must be flushed, not stranded"
    assert logger.log_queue == []


@pytest.mark.asyncio
async def test_final_drain_retries_transient_failure_then_delivers():
    """A transient 5xx during the eviction drain must not strand the last
    batch: the final drain retries on its own (no periodic loop is left)."""
    logger = _make_logger()
    responses = [_response(500), _response(500), _response(202)]
    post_mock = AsyncMock(side_effect=responses)
    with patch.object(logger, "async_client") as client, patch("asyncio.sleep", new=AsyncMock()):
        client.post = post_mock
        await logger._log_async_event(standard_logging_object=_standard_logging_object())
        await logger._final_drain()
    assert post_mock.await_count == 3
    assert logger.log_queue == []
