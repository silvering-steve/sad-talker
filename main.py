import io
import os
import base64

import zipfile
from typing import List
from uuid import uuid4, UUID

from fastapi_sessions.backends.implementations import InMemoryBackend
from fastapi_sessions.session_verifier import SessionVerifier
from pydantic import BaseModel

import uvicorn
from starlette.responses import JSONResponse, StreamingResponse
from fastapi import FastAPI, Body, Response, Depends, HTTPException
from fastapi_sessions.frontends.implementations import SessionCookie, CookieParameters

from elevenlabs import generate, Voice, VoiceSettings, set_api_key
from openai import OpenAI

from src.gradio_demo import SadTalker

from huggingface_hub import snapshot_download

app = FastAPI()

# Init Model
client = OpenAI(api_key="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")
set_api_key("28df0b632b692bda62e579cb616a5266")

snapshot_download(repo_id='vinthony/SadTalker-V002rc', local_dir='./checkpoints', local_dir_use_symlinks=True)
sad_talker = SadTalker(lazy_load=True)


# Init Session
class SessionData(BaseModel):
    username: str
    chat: List


class BasicVerifier(SessionVerifier[UUID, SessionData]):
    def __init__(
            self,
            *,
            identifier: str,
            auto_error: bool,
            backend: InMemoryBackend[UUID, SessionData],
            auth_http_exception: HTTPException,
    ):
        self._identifier = identifier
        self._auto_error = auto_error
        self._backend = backend
        self._auth_http_exception = auth_http_exception

    @property
    def identifier(self):
        return self._identifier

    @property
    def backend(self):
        return self._backend

    @property
    def auto_error(self):
        return self._auto_error

    @property
    def auth_http_exception(self):
        return self._auth_http_exception

    def verify_session(self, model: SessionData) -> bool:
        """If the session exists, it is valid"""
        return True


cookie_params = CookieParameters()

cookie = SessionCookie(
    cookie_name="cookie",
    identifier="general_verifier",
    auto_error=True,
    secret_key="DONOTUSE",
    cookie_params=cookie_params,
)
backend = InMemoryBackend[UUID, SessionData]()

verifier = BasicVerifier(
    identifier="general_verifier",
    auto_error=True,
    backend=backend,
    auth_http_exception=HTTPException(status_code=403, detail="invalid session"),
)


# Route App
@app.get("/")
def check_health():
    return JSONResponse(
        content={
            "message": "Healthy"
        }
    )


@app.get("/api/v1/download/{name}")
async def download(name: str):
    zip_bytes_io = io.BytesIO()
    with zipfile.ZipFile(zip_bytes_io, 'w', zipfile.ZIP_DEFLATED) as zipped:
        for dirname, subdirs, files in os.walk(f"sheets/{name}"):
            zipped.write(dirname)
            for filename in files:
                zipped.write(os.path.join(dirname, filename))

    response = StreamingResponse(
        iter([zip_bytes_io.getvalue()]),
        media_type="application/x-zip-compressed",
        headers={"Content-Disposition": f"attachment;filename=output.zip",
                 "Content-Length": str(zip_bytes_io.getbuffer().nbytes)}
    )
    zip_bytes_io.close()

    return response


@app.post("/api/v1/create/{name}")
async def create(name: str, response: Response):
    session = uuid4()
    data = SessionData(
        username=name,
        chat=[
            {"role": "system", "content": "You are a helpful assistant."},
        ]
    )

    await backend.create(session, data)
    cookie.attach_to_response(response, session)

    return f"{name} session created"


@app.post("/api/v1/chat", dependencies=[Depends(cookie)])
async def chat(text: str = Body(...), chat_id: str = Body(...), voice_id: str = Body(...),
               session_data: SessionData = Depends(verifier), session_id: UUID = Depends(cookie)):
    session_data.chat.append(
        {"role": "user", "content": text}
    )

    message = client.chat.completions.create(
        model=chat_id,
        messages=session_data.chat
    ).choices[0].message.content

    session_data.chat.append(
        {"role": "assistant", "content": message}
    )

    audio = base64.b64encode(
        generate(
            text=message,
            voice=Voice(
                voice_id=voice_id,
                model="eleven_multilingual_v2",
                settings=VoiceSettings(
                    stability=.8,
                    similarity_boost=.8,
                    style=.2,
                    use_speaker_boost=True
                )
            )
        )
    ).decode('utf-8')

    await backend.update(session_id, session_data)

    return JSONResponse(
        content={
            "message": message,
            "audio": audio
        }
    )


@app.get("/check_session", dependencies=[Depends(cookie)])
async def check_session(session_data: SessionData = Depends(verifier)):
    return session_data


@app.delete("/delete_session")
async def del_session(response: Response, session_id: UUID = Depends(cookie)):
    await backend.delete(session_id)
    cookie.delete_from_response(response)

    return "deleted session"


if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
