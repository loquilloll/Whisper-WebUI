import pytest
import asyncio
import httpx
import subprocess
import time
import sys
import random
from io import BytesIO
from fastapi import UploadFile

from backend.db.task.models import TaskStatus
from backend.tests.test_task_status import wait_for_task_completion
from backend.tests.test_backend_config import (
    get_client,
    setup_test_file,
    get_upload_file_instance,
    calculate_wer,
    TEST_PIPELINE_PARAMS,
    TEST_ANSWER,
)


@pytest.fixture(scope="module")
def uvicorn_server():
    host = "127.0.0.1"
    port = 8000
    base_url = f"http://{host}:{port}"
    # Assuming /docs or /openapi.json is a valid endpoint for health check
    # If your app has a specific health check endpoint, prefer that.
    health_check_url = f"{base_url}/docs"

    # Start the uvicorn server as a subprocess
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "backend.main:app",
            "--host",
            host,
            "--port",
            str(port),
            "--log-level",
            "debug",
        ],
    )

    max_wait_time = 30  # seconds to wait for server to start
    poll_interval = 0.5  # seconds between health checks
    start_time = time.time()

    server_ready = False
    print(f"Waiting for Uvicorn server to start at {base_url}...")
    while time.time() - start_time < max_wait_time:
        try:
            # Use a synchronous httpx client for the health check within the fixture
            with httpx.Client() as client:
                response = client.get(health_check_url, timeout=1.0)
            if response.status_code == 200:
                server_ready = True
                print(f"Uvicorn server started successfully.")
                break
        except httpx.ConnectError:
            # Server not yet accepting connections
            time.sleep(poll_interval)
        except Exception as e:
            # Other errors during health check (e.g., timeout, unexpected status code)
            print(f"Health check attempt failed: {e}")
            time.sleep(poll_interval)

    if not server_ready:
        stdout, stderr = process.communicate()  # Get output for debugging
        process.terminate()
        process.wait()
        raise RuntimeError(
            f"Uvicorn server failed to start at {base_url} within {max_wait_time} seconds.\n"
            f"STDOUT: {stdout.decode(errors='ignore')}\n"
            f"STDERR: {stderr.decode(errors='ignore')}"
        )

    yield base_url

    # Terminate the server after the tests
    print("Terminating Uvicorn server...")
    process.terminate()
    try:
        process.wait(timeout=5)  # Wait for graceful termination
        print("Uvicorn server terminated gracefully.")
    except subprocess.TimeoutExpired:
        print("Uvicorn server did not terminate gracefully, killing...")
        process.kill()
        process.wait()
        print("Uvicorn server killed.")


@pytest.mark.parametrize("pipeline_params", [TEST_PIPELINE_PARAMS])
def test_transcription_endpoint(get_upload_file_instance, pipeline_params: dict):
    client = get_client()
    file_content = BytesIO(get_upload_file_instance.file.read())
    get_upload_file_instance.file.seek(0)

    response = client.post(
        "/transcription",
        files={"file": (get_upload_file_instance.filename, file_content, "audio/mpeg")},
        params=pipeline_params,
    )

    assert response.status_code == 201
    assert response.json()["status"] == TaskStatus.QUEUED
    task_identifier = response.json()["identifier"]
    assert isinstance(task_identifier, str) and task_identifier

    completed_task = wait_for_task_completion(identifier=task_identifier)

    assert (
        completed_task is not None
    ), f"Task with identifier {task_identifier} did not complete within the expected time."

    result = completed_task.json()["result"]
    assert result, "Transcription text is empty"

    wer = calculate_wer(TEST_ANSWER, result[0]["text"].strip().replace(",", "").replace(".", ""))
    assert wer < 0.1, f"WER is too high, it's {wer}"


@pytest.mark.asyncio
async def test_concurrent_async_transcription_requests_via_uvicorn(uvicorn_server):
    # dispatch all transcription requests immediately

    # -- override file_paths with your 5 .m4a files --
    file_paths = [
        "/mnt/FC1AFB6C1AFB2276/OBS/2025-05-22_14-34-35_part-0_audio.m4a",
        "/mnt/FC1AFB6C1AFB2276/OBS/2025-05-22_15-00-53_part-1_audio.m4a",
        "/mnt/FC1AFB6C1AFB2276/OBS/2025-05-22_15-01-10_part-2_audio.m4a",
        "/mnt/FC1AFB6C1AFB2276/OBS/2025-05-22_15-24-00_part-3_audio.m4a",
        "/mnt/FC1AFB6C1AFB2276/OBS/2025-05-22_15-30-01_part-4_audio.m4a",
    ]

    # disable HTTPX timeouts (or bump to whatever you needs)
    async with httpx.AsyncClient(base_url=uvicorn_server, timeout=None) as client:
        posts = []
        file_objs = []

        for path in file_paths:
            f = open(path, "rb")
            file_objs.append(f)
            # random jitter before each POST (0–1000 ms)
            delay_ms = random.randint(1000, 3000)
            await asyncio.sleep(delay_ms / 1000)
            # override the Whisper model size to large-v3-turbo
            params = {**TEST_PIPELINE_PARAMS, "model_size": "large-v3-turbo"}
            posts.append(client.post("/transcription/", files={"file": f}, params=params))

        # kick off all uploads
        responses = await asyncio.gather(*posts)

        # now close the files
        for f in file_objs:
            f.close()

        # ensure all submissions succeeded (201 Created)
        for r in responses:
            assert r.status_code == 201

        uuids = [r.json()["identifier"] for r in responses]

        # poll each task until it's completed or fail
        completed = set()

        while len(completed) < len(uuids):
            for uid in uuids:
                if uid in completed:
                    continue

                status_resp = await client.get(f"/task/{uid}")
                status = status_resp.json().get("status")
                print(f"Task {uid} status: {status}", flush=True)
                # once we see a terminal state, mark this uid done so we exit the loop
                if status in {TaskStatus.COMPLETED, TaskStatus.FAILED}:
                    completed.add(uid)

            await asyncio.sleep(1)

        # fetch and verify each transcript
        for uid in uuids:
            resp = await client.get(f"/task/{uid}")
            assert resp.status_code == 200

            result = resp.json()["result"]
            assert result, "Transcription text is empty"

            wer = calculate_wer(TEST_ANSWER, result[0]["text"].strip().replace(",", "").replace(".", ""))
            assert wer < 0.3, f"HIGH WER={wer} for {uid}"
