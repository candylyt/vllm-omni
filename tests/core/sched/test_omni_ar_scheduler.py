from vllm.v1.request import RequestStatus

from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler


class _FakeRequest:
    def __init__(self):
        self.output_token_ids = []
        self.status = None
        self.stop_reason = None

    def append_output_token_ids(self, token_id):
        self.output_token_ids.append(token_id)


def test_moss_tts_delay_audio_end_stops_and_trims_spec_tokens():
    scheduler = OmniARScheduler.__new__(OmniARScheduler)
    scheduler._moss_tts_delay_stop_token_id = 151653

    request = _FakeRequest()
    new_token_ids, stopped = scheduler._update_request_with_output(
        request,
        [151653, 42, 43],
    )

    assert stopped
    assert new_token_ids == [151653]
    assert request.output_token_ids == [151653]
    assert request.status == RequestStatus.FINISHED_STOPPED
    assert request.stop_reason == 151653
