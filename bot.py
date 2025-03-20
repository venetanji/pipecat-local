#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import sys

import aiohttp
from dotenv import load_dotenv
from loguru import logger

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.frames.frames import Frame, TranscriptionFrame
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai import OpenAILLMService
from pipecat.services.xtts import XTTSService
from pipecat.services.whisper import WhisperSTTService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

class TranscriptionLogger(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            print(f"Transcription: {frame.text}")


async def main():
    async with aiohttp.ClientSession() as session:
        transport = LocalAudioTransport(LocalAudioTransportParams(audio_in_enabled=True, audio_out_enabled=True))

        stt = WhisperSTTService(no_speech_prob=0.4, model="large-v3")

        tl = TranscriptionLogger()


        tts = XTTSService(
            aiohttp_session=session,
            voice_id="Claribel Dervla",
            base_url="http://localhost:8000",
        )

        llm = OpenAILLMService(api_key='ollama', model="gemma3:12b", base_url="http://localhost:11434/v1")

        messages = [
            {
                "role": "system",
                "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way.",
            },
        ]

        context = OpenAILLMContext(messages)
        context_aggregator = llm.create_context_aggregator(context)

        pipeline = Pipeline(
            [
                transport.input(),  # Transport user input
                stt,
                context_aggregator.user(),  # User responses
                llm,  # LLM
                tts,  # TTS
                transport.output(),  # Transport bot output
                context_aggregator.assistant(),  # Assistant spoken responses
            ]
        )

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
                enable_usage_metrics=True,
                report_only_initial_ttfb=True,
            ),
        )


        runner = PipelineRunner(handle_sigint=False if sys.platform == "win32" else True)

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
