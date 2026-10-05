#
# Voice pipeline — Prosper AI Software Engineer Challenge
#
# The runnable voice agent: WebRTC transport + ElevenLabs STT/TTS + OpenAI LLM,
# driven by a Pipecat Flows node graph. This file is generic — it loads an agent
# definition (JSON) via AgentBuilder and runs it. Swapping the agent is a data
# change (edit/replace the JSON), not a code change.
#
#   agents/<active>.json  ->  AgentBuilder  ->  Pipecat Flows graph  ->  FlowManager
#
# Run:  python bot.py   then open http://localhost:7860/builder  (or /client)
#
# The builder UI (builder_api.py) is mounted on the same server. It picks which
# agent the next call runs, and shows a live trace fed by TraceTap below.
#

import os
import time
from pathlib import Path

import certifi

# python.org macOS builds ship without a CA bundle; NLTK (used by pipecat's text
# aggregation) downloads data at import time and fails SSL verification without this.
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    MetricsFrame,
    TranscriptionFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import LLMUsageMetricsData, TTFBMetricsData
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.elevenlabs.stt import ElevenLabsRealtimeSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.workers.runner import WorkerRunner
from pipecat_flows import FlowManager

import builder_api
from agent_builder import AgentBuilder
from catalog.tools import get_engine
from catalog.vocab import stt_keyterms
from sim.textflow import PRICES

# Load .env next to this file, so the bot runs the same from the repo root or backend/.
load_dotenv(Path(__file__).parent / ".env", override=True)



class TraceTap(FrameProcessor):
    """Pass-through processor that mirrors transcripts and LLM token usage into
    the builder's live trace panel."""

    def __init__(self, source: str, model: str = ""):
        super().__init__()
        self.source = source
        self.model = model
        self._buf = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if self.source == "user" and isinstance(frame, TranscriptionFrame):
            builder_api.emit({"type": "user", "text": frame.text})
        elif self.source == "llm":
            if isinstance(frame, LLMTextFrame):
                self._buf.append(frame.text)
            elif isinstance(frame, LLMFullResponseEndFrame) and self._buf:
                builder_api.emit({"type": "assistant", "text": "".join(self._buf).strip()})
                self._buf = []
            elif isinstance(frame, MetricsFrame):
                for d in frame.data:
                    if isinstance(d, LLMUsageMetricsData):
                        u = d.value
                        inp, cached_p, out = PRICES.get(self.model, PRICES["gpt-4.1"])
                        cached = u.cache_read_input_tokens or 0
                        cost = ((u.prompt_tokens - cached) * inp + cached * cached_p + u.completion_tokens * out) / 1e6
                        builder_api.emit({"type": "usage", "prompt": u.prompt_tokens, "cached": cached,
                                          "completion": u.completion_tokens, "cost": cost})
        await self.push_frame(frame, direction)


class LatencyTap(FrameProcessor):
    """Measures voice-to-voice latency per turn (caller stops speaking -> bot audio
    starts, server side) and per-service TTFB, and streams both to the trace."""

    def __init__(self):
        super().__init__()
        self._user_stopped_at = None
        self._ttfb = {}

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, VADUserStoppedSpeakingFrame):
            self._user_stopped_at = time.monotonic()
            self._ttfb = {}
        elif isinstance(frame, MetricsFrame):
            for d in frame.data:
                if isinstance(d, TTFBMetricsData) and d.value:
                    kind = "llm" if "LLM" in d.processor else "tts" if "TTS" in d.processor else "stt" if "STT" in d.processor else d.processor
                    # LLM may run several times per turn (tool calls): keep the sum.
                    self._ttfb[kind] = round(self._ttfb.get(kind, 0) + d.value, 3)
        elif isinstance(frame, BotStartedSpeakingFrame) and self._user_stopped_at is not None:
            latency = time.monotonic() - self._user_stopped_at
            self._user_stopped_at = None
            builder_api.emit({"type": "latency", "seconds": round(latency, 2), "ttfb": self._ttfb})
            logger.info(f"[latency] voice-to-voice {latency:.2f}s ttfb={self._ttfb}")
        await self.push_frame(frame, direction)


transport_params = {
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
}


async def run_bot(
    transport: BaseTransport, runner_args: RunnerArguments, builder: AgentBuilder
) -> None:
    config = builder.config
    logger.info(f"Starting '{config.name}' with {len(config.nodes)} nodes")

    # Bias STT towards the catalog's rare words (surnames, location names, test names).
    stt = ElevenLabsRealtimeSTTService(
        api_key=os.environ["ELEVENLABS_API_KEY"],
        settings=ElevenLabsRealtimeSTTService.Settings(keyterms=stt_keyterms(get_engine().c)),
    )
    tts = ElevenLabsTTSService(
        api_key=os.environ["ELEVENLABS_API_KEY"],
        # Flash is ElevenLabs' lowest-latency model (~75ms model time).
        settings=ElevenLabsTTSService.Settings(voice=config.voice_id, model="eleven_flash_v2_5"),
    )
    llm = OpenAILLMService(api_key=os.environ["OPENAI_API_KEY"], model=config.model)

    context = LLMContext()
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            TraceTap("user"),
            context_aggregator.user(),
            llm,
            TraceTap("llm", config.model),
            tts,
            transport.output(),
            LatencyTap(),
            context_aggregator.assistant(),
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    flow_manager = FlowManager(
        llm=llm,
        context_aggregator=context_aggregator,
        worker=worker,
        transport=transport,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected — starting flow at initial node")
        builder_api.emit({"type": "call_start", "agent": config.name})
        await flow_manager.initialize(await builder.build_initial_node())

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        builder_api.emit({"type": "call_end"})
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Entry point invoked by the Pipecat dev runner (and Pipecat Cloud)."""
    transport = await create_transport(runner_args, transport_params)
    # Load the agent fresh per call, so edits saved in the builder apply immediately.
    builder = AgentBuilder.from_json(builder_api.active_agent_path(), on_event=builder_api.emit)
    await run_bot(transport, runner_args, builder)


if __name__ == "__main__":
    from pipecat.runner.run import app, main

    builder_api.register(app)
    from catalog.jev import jev_available
    if not jev_available():
        logger.warning("TYPESAFE_API_KEY not set: Jev-based matchers (hybrid, jev, cascade) will "
                       "fall back to the local fuzzy matcher. Lookups show 'fuzzy_fallback' in the trace.")
    main()
